import random
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import torch
from einops import rearrange


def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(args, x_s, a_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    a_aug = a_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        a = a_s[i]
        orientation = np.random.randint(0, 10)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
            a = a.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
            a = a.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
            a = a.rot90(3, (2, 3))
        elif orientation == 4:
            x = x.rot90(1, (1, 3))
            a = a.rot90(1, (1, 3))
        elif orientation == 5:
            x = x.rot90(2, (1, 3))
            a = a.rot90(2, (1, 3))
        elif orientation == 6:
            x = x.rot90(3, (1, 3))
            a = a.rot90(3, (1, 3))
        elif orientation == 7:
            x = x.rot90(1, (1, 2))
            a = a.rot90(1, (1, 2))
        elif orientation == 8:
            x = x.rot90(2, (1, 2))
            a = a.rot90(2, (1, 2))
        elif orientation == 9:
            x = x.rot90(3, (1, 2))
            a = a.rot90(3, (1, 2))
        x_aug[i] = x
        a_aug[i] = a
        x_rot[i] = orientation

    return x_aug, a_aug, x_rot


def rot_rand_v2(args, x_s, permutations):
    img_n = x_s.size()[0]
    x_aug = torch.cat((x_s,x_s,x_s,x_s)).detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n*4).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        perm = np.random.randint(len(permutations))
        for j, ori in enumerate(permutations[perm]):
            if ori == 0:
                pass
            elif ori == 1:
                x = x.rot90(1, dims=[1,2])
            elif ori == 2:
                x = x.rot90(2, dims=[1,2])
            elif ori == 3:
                x = x.rot90(3, dims=[1,2])
            x_aug[i*4 + j] = x
            x_rot[i*4 + j] = ori

    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug


def aug_rand_v2(args, x_s):
    li = [0,1,2,3]
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)

    for i in range(0, img_n, 4):
        li_a = random.sample(li, 4)
        li_b = list(set(li) - set(li_a))

        for j in li_a:
            x = x_s[i+j]
            orientation = np.random.randint(0, 4)
            if orientation == 0:
                pass
            elif orientation == 1:
                x = x.rot90(1, dims=[1,2])
            elif orientation == 2:
                x = x.rot90(2, dims=[1,2])
            elif orientation == 3:
                x = x.rot90(3, dims=[1,2])

            x_aug[i+j] = patch_rand_drop(args, x)
            idx_rnd = randint(0, img_n)
            if idx_rnd != (i+j):
                x_aug[i+j] = patch_rand_drop(args, x_aug[i+j], x_aug[idx_rnd])
            x_rot[i+j] = orientation


    return x_aug, x_rot


def jig_rand(args, x_s, permutations):

    x_s = rearrange(x_s, "(b p) c h w -> b p c h w", p=4)
    
    img_n = x_s.shape[0]
    x_jig = []

    for i in range(img_n):
        x = x_s[i]
        tiles = [x_tiles for x_tiles in x]

        perm = np.random.randint(len(permutations))
        data = [tiles[permutations[perm][t]] for t in range(4)]
        jig = torch.tensor([[permutations[perm][t]] for t in range(4)]).to(args.device)
        data = torch.stack(data, 0)
        
        x_s[i] = data
        x_jig.append(jig)

    x_jig = torch.stack(x_jig, 0)
    x_jig = rearrange(x_jig, "b p c -> (b p) c ", p=4).squeeze()
    x_aug = rearrange(x_s, "b p c h w -> (b p) c h w", p=4)

    return x_aug, x_jig

def get_atlas_mask(args, atlas):
    img_n = atlas.size()[0]
    device = torch.device(f"cuda:{args.local_rank}")
    contents = torch.zeros((img_n,120)).to(device)
    masks = torch.ones((img_n,120)).to(device)
    atlas_masks = torch.ones_like(atlas)
    for i in range(img_n):
        for j in range(120):
            contents[i,j] = torch.numel(atlas[atlas==j])
            if contents[i,j] < 100:
                masks[i,j] = 0
            if contents[i,0] == 1:
                masks[i] = masks[i] * 0
                atlas_masks[i] = atlas_masks[i] * 0
                break
    return atlas_masks, masks

def get_feature_mask(args, mask):
    img_n = mask.size()[0]
    device = torch.device(f"cuda:{args.local_rank}")
    fmask = torch.zeros((img_n,138))
    for i in range(img_n):
        fmask[i] = torch.tensor([1] + [w for w in mask[i][51:85] for i in range(2)] + \
                                [1] + [w for w in mask[i][86:] for i in range(2)])
    return fmask.to(device)

