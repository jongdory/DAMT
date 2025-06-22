import os
import sys
import numpy as np
import random
import argparse
import datetime
import time
import math
import SimpleITK as sitk
import scipy.stats as stats
# import tensorflow # needs to call tensorflow before torch, otherwise crush
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from timm.scheduler import create_scheduler
from timm.utils import accuracy

from datasets import get_brain_dataet
# from trainer import Trainer
from models import SSLHead_Swin
from monai import transforms
from monai.losses import DiceLoss, DiceCELoss
from loss import Loss, Contrast
from ops import rot_rand, aug_rand, get_atlas_mask, get_feature_mask
import utils
import warnings
warnings.filterwarnings(action='ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_argparser():
    parser = argparse.ArgumentParser(description='Argparser')

    # Model parameters
    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--input-size', default=256, type=int, help='images input size')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Dataset parameters
    parser.add_argument('--epochs', type=int, default = 301, help = '# of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--nb-classes', type=int, default=1000, help='# of classes')
    parser.add_argument('--seed', type=int, default = 42, help = 'random seed')
    parser.add_argument('--batch-size', type=int, default = 4, help = 'batch size')
    
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
            local views to generate. Set this parameter to 0 to disable multi-crop training.
            When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--loc_patch_crops_number', type=int, default=1, help="""Number of small
            location patch views to generate. Set this parameter to 0 to disable multi-crop training.
            When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)        
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")    
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--project', type=str, default="self-supervised-learning")
    parser.add_argument('--data-path', type=str, default='/store8/01.Database/01.Brain/')
    parser.add_argument('--data-type', type=str, default="OG")
    parser.add_argument('--name', type=str, default="ssl")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--in-channels", type=int, default=1, help="in channels")
    parser.add_argument('--type', type=str, default="rot")
    parser.add_argument('--model', type=str, default="swin")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--output_dir', default='./runs_dict',
                            help='path where to save, empty for no saving')

    args = parser.parse_args()
    
    return args


def maybe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def save_nifti(image, prefix, index):
    
    path = f"save/{index}"
    maybe_mkdir(path)

    if prefix == "glo_par_P" or prefix == "loc_par_P":
        image = torch.argmax(image, axis=0)

    image = image.detach().cpu().numpy().squeeze()
    image = np.float32(image)
    img = sitk.GetImageFromArray(image)

    sitk.WriteImage(img, f"{path}/{prefix}.nii.gz")

def remove_zerotensor(tensor_p, tensor):
    tensor_p_list = []
    tensor_list = []
    for i in range(tensor.shape[0]):
        if tensor[i].sum() != 0:
            tensor_list.append(tensor[i])
            tensor_p_list.append(tensor_p[i])

    tensor_p = torch.stack(tensor_p_list) if tensor_p_list else torch.tensor([])
    tensor = torch.stack(tensor_list) if tensor_list else torch.tensor([])

    return tensor_p.cuda(non_blocking=True), tensor.cuda(non_blocking=True)


def main():
    args = get_argparser()
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    
    transform = DataAugmentation(args.local_crops_number, args.loc_patch_crops_number)
    dataset = get_brain_dataet(args=args, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    ## create a data loader
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        # drop_last=True,
    )

    # Network initialize
    model = SSLHead_Swin(args).cuda()
    model = nn.parallel.DistributedDataParallel(model, 
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)
    
    ############## Load from checkpoint if exists, otherwise from model ###############
    loss_function = Loss(args.batch_size_per_gpu, args).cuda()
    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256, #256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss_function=loss_function,
    )
    start_epoch = to_restore["epoch"]
    print(start_epoch)
    
    ############## TRAINING ###############
    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(model, loss_function, data_loader, optimizer, 
                        lr_schedule, wd_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, loss_function, data_loader, optimizer, 
                        lr_schedule, wd_schedule, epoch, fp16_scaler, args):

        loss_rot = torch.nn.CrossEntropyLoss()
        loss_loc = torch.nn.CrossEntropyLoss()
        loss_contrast = Contrast(args, args.batch_size_per_gpu)
        loss_atlas = DiceCELoss(to_onehot_y=True, softmax=True) 
        loss_feat = torch.nn.L1Loss()
        loss_texture = torch.nn.L1Loss() # torch.nn.MSELoss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        for it, (images, atlases, masks, radiomics, features, loc_trues) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            images = [im.cuda(non_blocking=True) for im in images]
            masks = [mask.cuda(non_blocking=True) for mask in masks]

            glo_atlas = atlases[0]
            loc_atlas = torch.cat(atlases[1:], dim=0)

            glo_mask = masks[0]
            loc_mask = torch.cat(masks[1:], dim=0)

            glo_x = images[0]

            loc_x1 = torch.cat(images[-args.loc_patch_crops_number:], dim=0)
            x2, _ , _    = rot_rand(args, glo_x, glo_x)

            loc_x1 = torch.cat(images[-args.loc_patch_crops_number:], dim=0)
            x1, a1, rot1 = rot_rand(args, glo_x, glo_atlas)
            x2, _ , _    = rot_rand(args, glo_x, glo_atlas)
            x3, a3, rot2 = rot_rand(args, loc_x1, loc_atlas)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # global forward
                hidden_states_out1, cls_token1 = model.module.encode(x1)
                hidden_states_out2, cls_token2 = model.module.encode_mask(x2, glo_mask)
                rot1_p = model.module.forward_rot(cls_token1)
                texture_p = model.module.forward_texture(hidden_states_out1[4])
                glo_feat_p = model.module.forward_global(hidden_states_out1[4])
                glo_atlas_p = model.module.forward_decoder(hidden_states_out1)
                contrastive1_p = model.module.forward_contrastive(cls_token1)
                contrastive2_p = model.module.forward_contrastive(cls_token2)
                # local forward
                hidden_states_out3, cls_token3 = model.module.encode(x3)
                hidden_states_out4, _          = model.module.encode_mask(loc_x1, loc_mask)
                rot2_p = model.module.forward_rot(cls_token3)
                loc_p = model.module.forward_loc(cls_token3)
                loc_atlas_p = model.module.forward_decoder(hidden_states_out3)

                mim_loss = model.module.forward_mim(x2, glo_mask, hidden_states_out2[4]) + \
                           model.module.forward_mim(loc_x1, loc_mask, hidden_states_out4[4])

                glo_atlas_p, glo_atlas = remove_zerotensor(glo_atlas_p, a1)
                loc_atlas_p, loc_atlas = remove_zerotensor(loc_atlas_p, a3)
                texture_p, glo_radi = remove_zerotensor(texture_p, glo_radi)
                glo_feat_p, glo_feat = remove_zerotensor(glo_feat_p, glo_feat)

                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rot = torch.cat([rot1, rot2], dim=0)

                # # compute loss
                rot_loss = loss_rot(rot_p, rot)
                loc_loss = loss_loc(loc_p, loc_trues)
                contrastive_loss = loss_contrast(contrastive1_p, contrastive2_p)
                glo_atlas_loss = loss_atlas(glo_atlas_p, glo_atlas) if glo_atlas.sum() != 0 else 0
                loc_atlas_loss = loss_atlas(loc_atlas_p, loc_atlas) if loc_atlas.sum() != 0 else 0
                atlas_loss = 0.5 * (glo_atlas_loss + loc_atlas_loss)
                feat_loss = 5 * loss_feat(glo_feat_p, glo_feat) if glo_feat.sum() != 0 else 0
                texture_loss = 5 * loss_texture(texture_p, glo_radi) if glo_radi.sum() != 0 else 0
                loss = rot_loss + loc_loss +  contrastive_loss + feat_loss + texture_loss + atlas_loss + mim_loss


            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("\nAveraged stats:", metric_logger, "\n")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class MaskGenerator:
    def __init__(self, input_size=128, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        return mask
    

class DataAugmentation(object):
    def __init__(self, local_crops_number, loc_patch_crops_number):
        self.load_image = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
                transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
                transforms.Lambdad(keys=["image"], func=lambda x: x[0, :, :, :]),
                transforms.AddChanneld(keys=["image"]),
                transforms.EnsureTyped(keys=["image"]),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True),
                transforms.Spacingd(keys=["image", "label"], pixdim=(1.25, 1.25, 1.25), mode ="nearest", allow_missing_keys=True),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128,128,128),allow_missing_keys=True),
                transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.05, upper=99.95, b_min=0, b_max=1)
            ]
        )
        
        color_jitter = transforms.Compose(
            [
                transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
            ]
        )
        # global crop
        self.global_transfo = transforms.Compose([
            transforms.RandSpatialCropd( keys=["image", "label"], roi_size=(128, 128, 128), random_size=False, allow_missing_keys=True), # 128
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.loc_patch_crops_number = loc_patch_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(56, 56, 56), random_size=False, allow_missing_keys=True), # 56
            transforms.Resized(keys=["image", "label"], spatial_size=(64, 64, 64), mode = "nearest", allow_missing_keys=True),
        ])

        self.glo_mask_generator = MaskGenerator(
            input_size=128,
            mask_patch_size=16,
            model_patch_size=2,
            mask_ratio=0.75,
        )

        self.loc_mask_generator = MaskGenerator(
            input_size=64,
            mask_patch_size=16,
            model_patch_size=2,
            mask_ratio=0.75,
        )

    
    def get_randlocation(self, num):
        list = []
        loc = np.random.randint(0,9)
        for i in range(num):
            while loc in list:
                loc = np.random.randint(0,9)
            list.append(loc)

        return list

    def __call__(self, image):
        image = self.load_image(image)
        crops = []
        crops.append(self.global_transfo(image.copy()))
        crops.append(self.local_transfo(image.copy()))
        images = [crop['image'] for crop in crops]
        masks = [self.glo_mask_generator(), self.loc_mask_generator()] #+ [self.loc_mask_generator() for i in range(self.loc_patch_crops_number)]

        return images, masks



if __name__ == "__main__":
    main()