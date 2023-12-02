# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import yaml
from .swin import SwinTransformer
from .swin_adapters import SwinTransformerWithAdapters

from .swin_PHM import SwinTransformerPHM
from .swin_compacter import SwinTransformerCompacter
from .swin_LowRank import SwinTransformerLowRank
from .swin_LoRa import SwinTransformerLoRa
from .swin_adapters_layer import SwinTransformerWithAdaptersLayerWise
from .swin_ssf import SwinTransformerSSF

from .vit_adapters import  ViTwithAdapters
from .cvt import get_cls_model
from .cvt_adapters import get_cls_model_adapters
from .t2t import T2t_vit_14
from .t2t_adapters import T2t_vit_14_Adapters

from .swin_vpt import PromptedSwinTransformer
from .swin_factTT import SwinTransformer_FactTK, WindowAttention, Mlp

from .resnet import ResNet50
from .vit import ViT
from .residual_resnet import ResidualResNet

import os

import torch
import torch.nn as nn
import math

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False)
    elif model_type == "cvt":
        with open(r'configs/cvt_13_224.yaml') as file:
            config_cvt = yaml.load(file, Loader=yaml.FullLoader)
            model = get_cls_model(config, config_cvt["MODEL"]["CVT"])
    elif model_type == "cvt_adapters":
        with open(r'configs/cvt_13_224.yaml') as file:
            config_cvt = yaml.load(file, Loader=yaml.FullLoader)
            model = get_cls_model_adapters(config, config_cvt["MODEL"]["CVT"])
    elif model_type == "t2t":
        model = T2t_vit_14(
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=False,
            use_abs=False,
        )
    elif model_type == "t2t_adapters":
        model = T2t_vit_14_Adapters(
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=False,
            use_abs=False,
            type_adapter=config.TRAIN.TYPE_ADAPTERS,
            param_ratio=config.TRAIN.SIZE_ADAPTERS
        )
    elif model_type == 'resnet50':
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=False,
            use_abs=False,
            pretrained_bool = config.MODEL.FINETUNE
        )
    elif model_type == "vit":
        model = ViT(model_type="sup_vitb16_224", img_size=224, num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == "vit_adapters":
        model = ViTwithAdapters(
            model_type="sup_vitb16_224", img_size=224, num_classes=config.MODEL.NUM_CLASSES,
            type_adapter=config.TRAIN.TYPE_ADAPTERS,
            adapters_size=config.TRAIN.SIZE_ADAPTERS
        )
    elif model_type == "residualResnet26":
        current = os.getcwd()
        pre_model_path = os.path.join(current, "pretrained", "resnet26-timm.pth")
        model = ResidualResNet(
            num_classes     = config.MODEL.NUM_CLASSES,
            use_drloc       = config.TRAIN.USE_DRLOC,
            pretrained_path = pre_model_path,
            drloc_mode=False,
            use_abs=False,
            pretrained_bool = config.MODEL.FINETUNE
        )
    elif model_type == "swin_adapters":
        model = SwinTransformerWithAdapters(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            ratio_param=config.TRAIN.SIZE_ADAPTERS,
            )
    elif model_type == "swin_ssf":
        model = SwinTransformerSSF(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            ratio_param=config.TRAIN.SIZE_ADAPTERS,
            )

    elif model_type == "swin_compacter":
        model = SwinTransformerCompacter(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            adapter_state=config.TRAIN.USE_ADAPTERS,
            ratio_param=config.TRAIN.SIZE_ADAPTERS,
            )
    elif model_type == "swin_PHM":
        model = SwinTransformerPHM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            adapter_state=config.TRAIN.USE_ADAPTERS,
            ratio_param=config.TRAIN.SIZE_ADAPTERS,
            )
    elif model_type == "swin_LowRank":
        model = SwinTransformerLowRank(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            adapter_state=config.TRAIN.USE_ADAPTERS,
            ratio_param=config.TRAIN.SIZE_ADAPTERS,
            )
    elif model_type == "swin_LoRa":
        model = SwinTransformerLoRa(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=False,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=False
            )
    elif model_type == "swin_adapters_layer":
        model = SwinTransformerWithAdaptersLayerWise(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            type_adapters=config.TRAIN.TYPE_ADAPTERS,
            ratio_param=[[14, 0], [28, 0], [7, 18, 6, 0, 3, 0], [12, 8]],
            )

    elif model_type == "swin_vpt":
        model = PromptedSwinTransformer(
            prompt_config=config.MODEL.PROMPT,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False,
            )
    if model_type == 'swin_fact_tk':
        model = SwinTransformer_FactTK(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=False,
            use_abs=False)

        def fact_forward_attn(self, x, mask):
            B_, N, C = x.shape

            idx = int(math.log(C / 96, 2))
            FacTu = model.FacTu[idx]
            FacTv = model.FacTv[idx]
            qkv = self.qkv(x)
            q = FacTv(self.dp(self.q_FacTs(FacTu(x))))
            k = FacTv(self.dp(self.k_FacTs(FacTu(x))))
            v = FacTv(self.dp(self.v_FacTs(FacTu(x))))

            qkv += torch.cat([q, k, v], dim=2) * self.s
            qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            proj = self.proj(x)
            proj += FacTv(self.dp(self.proj_FacTs(FacTu(x)))) * self.s
            x = self.proj_drop(proj)
            return x

        def fact_forward_mlp(self, x):
            B, N, C = x.shape
            idx = int(math.log(C / 96, 2))
            FacTu = model.FacTu[idx]
            FacTv = model.FacTv[idx]
            h = self.fc1(x)  # B n 4c
            # print(x.size(), h.size())
            h += FacTv(self.dp(self.fc1_FacTs(FacTu(x))).reshape(
                B, N, 4, self.dim)).reshape(
                B, N, 4 * C) * self.s
            x = self.act(h)
            x = self.drop(x)
            h = self.fc2(x)
            x = x.reshape(B, N, 4, C)
            h += FacTv(self.dp(self.fc2_FacTs(FacTu(x).reshape(
                B, N, 4 * self.dim)))) * self.s
            x = self.drop(h)
            return x

        def set_FacT(model, dim=8, s=1):
            if type(model) == SwinTransformer_FactTK:
                model.FacTu = [nn.Linear(96 * expand, dim, bias=False) for expand in [1, 2, 4, 8]]
                model.FacTv = [nn.Linear(dim, 96 * expand, bias=False) for expand in [1, 2, 4, 8]]
                for weight in model.FacTv:
                    nn.init.zeros_(weight.weight)
                model.FacTu = nn.ModuleList(model.FacTu)
                model.FacTv = nn.ModuleList(model.FacTv)
            for _ in model.children():
                if type(_) == WindowAttention:
                    _.q_FacTs = nn.Linear(dim, dim, bias=False)
                    _.k_FacTs = nn.Linear(dim, dim, bias=False)
                    _.v_FacTs = nn.Linear(dim, dim, bias=False)
                    _.proj_FacTs = nn.Linear(dim, dim, bias=False)
                    _.dp = nn.Dropout(0.1)
                    _.s = s
                    # _.dim = dim
                    bound_method = fact_forward_attn.__get__(_, _.__class__)
                    setattr(_, 'forward', bound_method)
                elif type(_) == Mlp:
                    _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
                    _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
                    _.dim = dim
                    _.s = s
                    _.dp = nn.Dropout(0.1)
                    bound_method = fact_forward_mlp.__get__(_, _.__class__)
                    setattr(_, 'forward', bound_method)
                elif len(list(_.children())) != 0:
                    set_FacT(_, dim, s)


        set_FacT(model, dim=32, s=0.1)

    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
