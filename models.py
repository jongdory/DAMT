import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.utils import ensure_tuple_rep

from swin_unetr import SwinTransformer
from timm.models.layers import trunc_normal_
from typing import Optional, Sequence, Tuple, Union



class SSLHead_Swin(nn.Module):
    def __init__(self, args, dim=768):
        super(SSLHead_Swin, self).__init__()
        feature_size = 48
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.SwinViT = SwinTransformer(
            in_chans=args.in_channels,
            embed_dim=feature_size, # args.feature_size 48
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            spatial_dims=3,
        ).to(args.device)
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 10)
        self.location_pre = nn.Identity()
        self.location_head = nn.Linear(dim, 9)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.feature_pre = nn.Identity()
        self.texture_pre = nn.Identity()
        self.glo_feat_head = nn.Sequential(
            nn.Conv3d(dim, 32, 1),
            nn.InstanceNorm3d(32),
            nn.Flatten(),
            nn.Linear(2048, 138),
            nn.ReLU(),
        )
        self.loc_feat_head = nn.Sequential(
            nn.Conv3d(dim, 256, 1),
            nn.InstanceNorm3d(256),
            nn.Flatten(),
            nn.Linear(2048, 138),
            nn.ReLU(),
        )
        self.texture_head = nn.Sequential(
            nn.Conv3d(dim, 32, 1),
            nn.InstanceNorm3d(32),
            nn.Flatten(),
            nn.Linear(2048, 72),
            nn.ReLU(),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_size))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        self.decoder = ConvDecoder(spatial_dims=3, feature_size=feature_size,norm_name="instance", dim=dim)
        self.decoder_mim = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=32768, kernel_size=1),
            PixelShuffle3d(32),
        )
        self.patch_size = 2

        self.conv =  nn.Conv3d(dim // 16, 1, kernel_size=1, stride=1)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=120)

    def encode(self, x):
        hidden_states_out = self.SwinViT(x.contiguous())
        x4 = hidden_states_out[4]
        cls_token = self.avgpool(x4)
        cls_token = cls_token.flatten(start_dim=1)
        
        return hidden_states_out, cls_token
    
    def encode_mask(self, x, mask):
        z = self.SwinViT.patch_embed(x)
        B, C, H, W, D = z.shape
        z = z.flatten(start_dim=2).transpose(1, 2)

        assert mask is not None
        B, L, _ = z.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        z = z * (1. - w) + mask_tokens * w
        z = z.reshape(B, C, H, W, D)

        z0 = self.SwinViT.pos_drop(z)
        z0_out = self.SwinViT.proj_out(z0.contiguous())
        z1 = self.SwinViT.layers1[0](z.contiguous())
        z1_out = self.SwinViT.proj_out(z1.contiguous())
        z2 = self.SwinViT.layers2[0](z1.contiguous())
        z2_out = self.SwinViT.proj_out(z2.contiguous())
        z3 = self.SwinViT.layers3[0](z2.contiguous())
        z3_out = self.SwinViT.proj_out(z3.contiguous())
        z4 = self.SwinViT.layers4[0](z3.contiguous())
        z4_out = self.SwinViT.proj_out(z4.contiguous())
        
        hidden_states_out = [z0_out, z1_out, z2_out, z3_out, z4_out]
        cls_token = self.avgpool(z4)
        cls_token = cls_token.flatten(start_dim=1)

        return hidden_states_out, cls_token
    
    def forward_mim(self, x, mask, z):
        x_rec = self.decoder_mim(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).unsqueeze(1).contiguous()

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        return loss
    
    
    def forward_rot(self, cls_token):
        x_rot_pre = self.rotation_pre(cls_token)
        x_rot = self.rotation_head(x_rot_pre)
        return x_rot
    
    def forward_loc(self, cls_token):
        x_loc_pre = self.location_pre(cls_token)
        x_loc = self.location_head(x_loc_pre)
        return x_loc
    
    def forward_contrastive(self, cls_token):
        x_contrastive_pre = self.contrastive_pre(cls_token)
        x_contrastive = self.contrastive_head(x_contrastive_pre)
        return x_contrastive
    
    def forward_texture(self, x4):
        x_texture = self.texture_pre(x4)
        x_texture = self.texture_head(x_texture)
        return x_texture
    
    def forward_global(self, x4):
        x_glo_feat = self.feature_pre(x4)
        x_glo_feat = self.glo_feat_head(x_glo_feat)
        return x_glo_feat
    
    def forward_local(self, x4):
        x_loc_feat = self.feature_pre(x4)
        x_loc_feat = self.loc_feat_head(x_loc_feat)
        return x_loc_feat
    
    def forward_decoder(self, hidden_states_out):
        x_upsample = self.decoder(hidden_states_out)    
        # x_rec = self.conv(x_upsample)
        x_atlas = self.out(x_upsample)
        return x_atlas

    def forward(self, x, type):
        hidden_states_out, cls_token = self.encode(x)
        if type == "global":
            x_rot = self.forward_rot(cls_token)
            x_contrastive = self.forward_contrastive(cls_token)
            x_texture = self.forward_texture(hidden_states_out[4])
            x_glo_feat = self.forward_global(hidden_states_out[4])
            x_rec, x_atlas = self.forward_decoder(hidden_states_out)
            return x_rot, x_contrastive, x_texture, x_glo_feat, x_rec, x_atlas
        elif type == "global2":
            x_rot = self.forward_rot(cls_token)
            x_contrastive = self.forward_contrastive(cls_token)
            return x_rot, x_contrastive
        elif type == "local":
            x_loc = self.forward_loc(cls_token)
            x_loc_feat = self.forward_local(hidden_states_out[4])
            x_rec, x_atlas = self.forward_decoder(hidden_states_out)
            return x_loc, x_loc_feat, x_rec, x_atlas
        else:
            raise ValueError("Type must be global or local")
        
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
        
class ConvDecoder(nn.Module):
    def __init__(self, spatial_dims=3, feature_size=64,norm_name="instance", dim=1024):
        super(ConvDecoder, self).__init__()
        self.encoder10 = EncoderBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

        self.decoder4 = nn.Sequential(
            nn.Conv3d(dim , dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv3d(dim // 2 , dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(dim // 4 , dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv3d(dim // 8 , dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

    def forward(self, hidden_states_out):
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = torch.cat((self.decoder5(dec4), hidden_states_out[3]), dim=1)
        dec2 = torch.cat((self.decoder4(dec3), hidden_states_out[2]), dim=1)
        dec1 = torch.cat((self.decoder3(dec2), hidden_states_out[1]), dim=1)
        dec0 = torch.cat((self.decoder2(dec1), hidden_states_out[0]), dim=1)
        out = self.decoder1(dec0)

        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv(inp)
        out = self.norm(out)
        out = self.lrelu(out)

        return out


class TrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: Union[Sequence[int], int],
    ) -> None:


        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, inp):
        out = self.transp_conv(inp)

        return out




def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]





class JigsawHead(nn.Module):
    def __init__(
        self,
        in_dims: int,
        hid_dims: int=128,
        out_dims: int=1000,
    ) -> None:

        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_dims, hid_dims)
        self.classifier = nn.Linear(hid_dims * 16, out_dims)

    def forward(self, x):
        bs = x.shape[0]

        # 256 256 768 -> 256 768 16 16
        x = rearrange(x, "b (i j) e-> b e i j", i=16, j=16) # batch, patch, embedding

        # 256 768 16 16 -> 256 768 16 4 4
        x = rearrange(x, "b e (i w) (j h) -> b e (i j) w h", i=4, j=4) 
        # 256 768 16 4 4 -> 256 768 16
        x = self.avgpool(x).squeeze() 
        # 256 768 16 -> 256 16 768
        x = rearrange(x, "b e p -> b p e") 
        # 256 16 768 -> 256 16 128
        x = self.fc(x) 
        # 256 16 128 -> 256 2048
        x = self.flatten(x) 
        # 256 2048 -> 256 1000
        x = self.classifier(x.view(bs, -1))

        return x

class RotationHead(nn.Module):
    def __init__(
        self,
        in_dims: int,
        hid_dims: int,
        out_dims: int,
    ) -> None:

        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_dims, hid_dims)
        self.classifier = nn.Linear(hid_dims * 16, out_dims)

    def forward(self, x):
        bs = x.shape[0]
        # 256 768 16 16 -> 256 768 16 4 4
        x = rearrange(x, "b e (i w) (j h) -> b e (i j) w h", i=4, j=4) 
        # 256 768 16 4 4 -> 256 768 16
        x = self.avgpool(x).squeeze() 
        # 256 768 16 -> 256 16 768
        x = rearrange(x, "b e p -> b p e") 
        # 256 16 768 -> 256 16 128
        x = self.fc(x) 
        # 256 16 128 -> 256 2048
        x = self.flatten(x) 
        # 256 2048 -> 256 1000
        x = self.classifier(x.view(bs, -1))

        return x
    


