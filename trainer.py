import os
from pickletools import uint8
from re import X
import time
import datetime
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from typing import Iterable, Optional

from sklearn.metrics import roc_auc_score

from timm.utils import accuracy
from timm.utils import NativeScaler, get_state_dict, ModelEma
from einops import rearrange, repeat

from pathlib import Path
import wandb

import warnings
warnings.filterwarnings("ignore")

from ops import rot_rand, aug_rand, jig_rand, rot_rand_v2, aug_rand_v2
from loss import Loss
import utils

def maybe_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_schedule,
                 start_epoch,
                 args: argparse.ArgumentParser,
                ):

        """Initialization."""
        self.device = args.device
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.start_epoch = start_epoch
        self.args = args

    def train_one_epoch(self, model, loss_function, data_loader, optimizer, 
                        lr_schedule, epoch, fp16_scaler, args):
                        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

            # move images to gpu
            images = [im['image'].cuda(non_blocking=True) for im in images]
            # atlases = [im['seg'].cuda(non_blocking=True) for im in images]
            atlases = None
            locs = None
            glofeats = None
            locfeats = None
            glo_x = images[0]
            loc_x1 = images[1:4]
            loc_x2 = images[4:]
            x1, rot1 = rot_rand(args, glo_x)
            x2, rot2 = rot_rand(args, glo_x)
            x3, rot3 = rot_rand(args, loc_x1)
            x4, rot4 = rot_rand(args, loc_x2)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x3_augment = aug_rand(args, x3)
            x4_augment = aug_rand(args, x4)
            x1_augment = x1_augment
            x2_augment = x2_augment
            x3_augment = x3_augment
            x4_augment = x4_augment
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                rot1_p, _     , rec_x1, contrastive1_p, atlas1_p, glofeat1_p, _          = model(x1_augment)
                rot2_p, _     , rec_x2, contrastive2_p, atlas2_p, glofeat2_p, _          = model(x2_augment)
                rot3_p, _     , rec_x3, _             , atlas3_p, _         , locfeat3_p = model(x3_augment)
                rot4_p, loc_p , rec_x4, _             , atlas4_p, _         , locfeat4_p = model(x4_augment)
                rot_p = torch.cat([rot1_p, rot2_p, rot3_p, rot4_p], dim=0)
                rots  = torch.cat([rot1, rot2, rot3, rot4], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2, rec_x3, rec_x4], dim=0)
                imgs = torch.cat([x1, x2, x3, x4], dim=0)
                atlas_p = torch.cat([atlas1_p, atlas2_p, atlas3_p, atlas4_p], dim=0)
                glofeat_p = torch.cat([glofeat1_p, glofeat2_p], dim=0)
                locfeat_p = torch.cat([locfeat3_p, locfeat4_p], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, loc_p, locs, imgs_recon, imgs, contrastive1_p, contrastive2_p,
                                                   atlas_p, atlases, glofeat_p, glofeats, locfeat_p, locfeats)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # student update
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad:
                    param_norms = utils.clip_gradients(model, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(model, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            acc_rot = accuracy(rot_p, rots)[0]
            acc_loc = accuracy(loc_p, locs)[0]
            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
