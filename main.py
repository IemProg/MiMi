# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models

import collections.abc as container_abcs
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_best,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor
)

from drloc import cal_selfsupervised_loss, SymKlCriterion

# import EarlyStopping
from pytorchtools import EarlyStopping

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dsets_type', type=str, help='path to dataset', default="decathlon")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--exp_name', type=str, help='Experiment name with in output folder')
    parser.add_argument('--model_type', type=str, help='Model type')

    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O2', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--finetune', type=int, default = 0, help='Finetunning mode')

    parser.add_argument("--transform", action='store_true', help="Use data transformation")
    parser.add_argument('--OPT', type=str, default='adamw', choices=['SGD', 'adamw'], help='optimizer')
    parser.add_argument('--WD', type=float, default = 0.05, help='weight decay')
    parser.add_argument('--LR', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--WU', type=int, default=20, help='warm-up epochs')
    parser.add_argument('--WU_LR', type=float, default=5e-7, help='warm-up epochs')
    parser.add_argument('--pretrained_model', type=str, help='Experiment name with in output folder')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    parser.add_argument("--use_drloc", action='store_true', help="Use Dense Relative localization loss")
    parser.add_argument("--drloc_mode", type=str, default="l1", choices=["l1", "ce", "cbr"])
    parser.add_argument("--lambda_drloc", type=float, default=0.5, help="weight of Dense Relative localization loss")
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--use_multiscale", action='store_true')
    parser.add_argument("--ape", action="store_true", help="using absolute position embedding")
    parser.add_argument("--rpe", action="store_false", help="using relative position embedding")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--ssl_warmup_epochs", type=int, default=20)
    parser.add_argument("--total_epochs", type=int, default=100)

    parser.add_argument("--type_adapters", type=str, default="parallel")
    parser.add_argument("--size_adapters", type=int, default=32)
    #parser.add_argument('--param_ratios', help='delimited list input', type=str)
    parser.add_argument('--param_ratios', help='delimited list input', type=str, action='append', nargs='+')

    # Pruning
    parser.add_argument("--prune_layer", type=str, default="parallel_mlp")

    # PRUNING AMOUNT
    parser.add_argument("--prune_type", type=str, default="magnitude")
    parser.add_argument("--prune_struct", type=str, default="structured")
    parser.add_argument("--prune_amount", type=float, default=0.2)
    parser.add_argument("--delta_loss", type=float, default=0.1)
    parser.add_argument("--weighted", type=int, default=0, help="StructLAMP++")
    parser.add_argument("--scaling", type=int, default=1, help="StructLAMP++")
    parser.add_argument("--range", type=int, default=1)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # Debugging mode
    parser.add_argument("--debug", default=False, help="Debugging Mode. Default = False.")

    args, unparsed = parser.parse_known_args()
    return args  # , config

def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight

def main(config):
    # dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = \
    #                                 build_loader_split(config, val_size=0.2, seed=42)

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"\t\t Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    logger.info(f"\t\t FINETUNNING Argument: {config.MODEL.FINETUNE}")
    logger.info(f"\t\t DRLOC Argument: {config.TRAIN.USE_DRLOC}")
    current = os.getcwd()
    if config.MODEL.FINETUNE == 1:
        logger.info(f"\t\t Finetunning the model: {config.MODEL.TYPE}")
        if config.MODEL.TYPE == "resnet50":
            model = build_model(config)
            model.requires_grad = True
        elif config.MODEL.TYPE == "vit":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".npz")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.hidden_size, 21843)
            model.head = classifier

            checkpoint = np.load(pre_model_path)
            msg = model.load_from(checkpoint)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.hidden_size, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")
        elif config.MODEL.TYPE == "cvt":
            logger.info("\t\t Finetunning CvT-13 224x224")
            pre_model_path = os.path.join(current, "pretrained", "CvT-13-224x224-IN-1k" + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.dim_embed, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=True)
            logger.info(f"Checkpoint for CvT-13-224*224 finetunning")
            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.dim_embed, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
        else:
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')

            # TODO: make it dynamic
            msg = model.load_state_dict(checkpoint['model'], strict=True)
            #msg = model.load_state_dict(checkpoint, strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            for param in model.parameters():
                param.requires_grad = True
            logger.info("\t\t Finetunning {} Transformer".format(config.MODEL.TYPE))
    elif config.MODEL.FINETUNE == 3:
        if config.MODEL.TYPE == "swin_adapters" or config.MODEL.TYPE == "swin_adapters_layer":
            logger.info("\t\t Finetunning Swin Transformer Using Houlsbi Adapters")
            logger.info(f"\t\t TYPE OF ADAPTERS: {config.TRAIN.TYPE_ADAPTERS}")
            logger.info(f"\t\t SIZE OF ADAPTERS: {config.TRAIN.SIZE_ADAPTERS}")
            logger.info(f"\t\t STAGEs SIZE OF ADAPTERS: {config.MODEL.SWIN.PARAM_RATIOS}")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint['model'], strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if "parallel_mlp" in str(name):
                    param.requires_grad = True
                if "norm3" in str(name):
                    param.requires_grad = True
                if "norm4" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True

        elif config.MODEL.TYPE == "swin_ssf":
            logger.info("\t\t Finetunning Swin Transformer Using SSF")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint['model'], strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if "ssf_" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t Number of trainable params: {n_parameters}")

        elif config.MODEL.TYPE == "swin_vpt":
            logger.info("\t\t Finetunning Swin Transformer Using VPT")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint['model'], strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if "prompt_" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t Number of trainable params: {n_parameters}")

        elif config.MODEL.TYPE == "swin_fact_tk":
            logger.info("\t\t Finetunning Swin Transformer Using Fact TK")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint['model'], strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if 'FacT' in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t Number of trainable params: {n_parameters}")


        elif config.MODEL.TYPE == "vit_adapters":
            logger.info("\t\t Finetunning ViT Transformer Using Houlsbi Adapters")
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".npz")
            model = build_model(config)
            classifier = nn.Linear(model.hidden_size, 21843)
            model.head = classifier

            checkpoint = np.load(pre_model_path)
            msg = model.load_from(checkpoint)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.hidden_size, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if "parallel_mlp" in str(name):
                    param.requires_grad = True
                if "norm3" in str(name):
                    param.requires_grad = True
                if "norm4" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
    elif config.MODEL.FINETUNE == 6:
        if config.MODEL.TYPE == "swin":
            logger.info(f"\t\t Finetunning FitBit Model ")
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)

            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            #msg = model.load_state_dict(checkpoint, strict=False)

            net1_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint["model"].items()}
            net1_dict.update(pretrained_dict)
            msg = model.load_state_dict(net1_dict)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t Number of params: {n_parameters}")

            for param in model.parameters():
                param.requires_grad = False

            # Parameters to be optimized
            for param in model.head.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                if 'bias' in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
        elif config.MODEL.TYPE == "vit":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".npz")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.hidden_size, 21843)
            model.head = classifier

            checkpoint = np.load(pre_model_path)
            msg = model.load_from(checkpoint)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.hidden_size, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for param in model.parameters():
                param.requires_grad = False

            # Parameters to be optimized
            for param in model.head.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                if 'bias' in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True

    elif config.MODEL.FINETUNE == 10:
        if config.MODEL.TYPE == "swin":
            logger.info(f"\t\t FINETUNE type: {config.MODEL.FINETUNE}")
            logger.info(f"\t\t Finetuning Only head classifiers")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)

            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier
            checkpoint = torch.load(pre_model_path, map_location='cpu')

            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=True)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t Initial Number of params: {n_parameters}")

            for param in model.parameters():
                param.requires_grad = False

            # Parameters to be optimized
            for param in model.head.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
        elif config.MODEL.TYPE == "vit":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".npz")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.hidden_size, 21843)
            model.head = classifier

            checkpoint = np.load(pre_model_path)
            msg = model.load_from(checkpoint)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.hidden_size, config.MODEL.NUM_CLASSES)
            model.head = classifier

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for param in model.parameters():
                param.requires_grad = False

            # Parameters to be optimized
            for param in model.head.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
    else:
        raise NotImplementedError(f"Finetune type {config.MODEL.FINETUNE} not implemented")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\t\t After Number of params: {n_parameters}")
    model.cuda()

    # logger.info(str(model))
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if not config.DEBUG:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if hasattr(model_without_ddp, 'flops'):
        from ptflops import get_model_complexity_info
        import re

        # Model thats already available
        macs, params_ = get_model_complexity_info(model_without_ddp, (3, 224, 224), as_strings=True,
                                                  print_per_layer_stat=True, verbose=True)
        # Extract the numerical value
        flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
        # Extract the unit
        flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

        print('Computational complexity: {:<8}'.format(macs))
        print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
        print('Number of parameters: {:<8}'.format(params_))

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # supervised criterion
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion_sup = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_sup = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_sup = torch.nn.CrossEntropyLoss()

    # self-supervised criterion
    criterion_ssup = cal_selfsupervised_loss

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        # light
        # dataset_val.cache.reset()
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("\t\tStart training")
    start_time = time.time()
    # logs for training
    logs_dict = {"loss": [], "loss_test": [], "loss_val": [], "epoch_time": [], "loss_avg": [],
                 "acc1_test": [], "acc5_test": [], "acc1_val": [], "acc5_val": [],
                 "acc1_train": [], "acc5_train": [], "params": n_parameters}

    init_lambda_drloc = 0.0

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 5
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for phase in range(config.TRAIN.RANGE):
        # We apply same training process (annealing)
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            epoch = (phase * 100) + epoch
            if not config.DEBUG:
                data_loader_train.sampler.set_epoch(epoch)

            loss, epoch_time, loss_meter_avg = train_one_epoch(config, model, criterion_sup, criterion_ssup,
                data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, init_lambda_drloc
            )

            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            # Validation on training dataset
            acc1_train, acc5_train, loss_train = validate(config, data_loader_train, model, mixup_fn, mode="train")
            acc1, acc5, loss_val = validate(config, data_loader_val, model, mixup_fn, mode="test")

            logger.info(
                f"Training Accuracy of the network on the {len(dataset_train)} train images: {acc1_train:.5f}%")
            logger.info(f"Validation Accuracy of the network on the {len(dataset_val)} test images: {acc1:.5f}%")

            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            logs_dict["loss"].append(loss_train)
            logs_dict["epoch_time"].append(epoch_time)
            logs_dict["loss_avg"].append(loss_meter_avg)
            logs_dict["acc1_train"].append(acc1_train)
            logs_dict["acc5_train"].append(acc5_train)

            # logs_dict["acc1_val"].append(acc1)
            # logs_dict["acc5_val"].append(acc5)
            # logs_dict["loss_val"].append(loss_val)
            #
            # acc1, acc5, loss_test = validate(config, data_loader_test, model, mixup_fn, mode="test")
            # logger.info(f"Testing Accuracy of the network on the {len(dataset_test)} test images: {acc1:.5f}%")
            logs_dict["acc1_test"].append(acc1)
            logs_dict["acc5_test"].append(acc5)
            logs_dict["loss_test"].append(loss_val)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # early_stopping(loss_val, model)
            # if early_stopping.early_stop:
            #     logger.info(f"\t\t Early stopping at epoch {epoch}!!")
            #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    logs_dict["training_time"] = total_time

    # Save trainings logs
    logs_path = os.path.join(config.OUTPUT, "logs.pkl")

    with open(logs_path, "wb") as handle:
        pickle.dump(logs_dict, handle)
    handle.close()

    del dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
    del model, optimizer, model_without_ddp
    torch.cuda.empty_cache()

def train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader, optimizer, epoch,mixup_fn,lr_scheduler,logger,lambda_drloc):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
#    device = torch.device('cuda', dist.get_rank())

    end_time_tmp = time.time()
    logger.info(f"\t\t Number of classes: {config.MODEL.NUM_CLASSES}")

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion_sup(outputs["sup"], targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_drloc)
                loss += loss_ssup
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion_sup(outputs["sup"], targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_drloc)
                loss += loss_ssup
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            if config.TRAIN.USE_DRLOC:
                logger.info(f'weights: drloc {lambda_drloc:.4f}')
                logger.info(f' '.join(['%s: [%.4f]' % (key, value) for key, value in ssup_items.items()]))

        end_time_tmp = time.time()
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss, epoch_time, loss_meter.avg

@torch.no_grad()
def validate(config, data_loader, model, mixup_fn, mode = "test"):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output["sup"], target)

        # Datasets having less than 5 classes
        # TODO: accuracy is calculated on sup attribute of the output
        # dir outputs:  ['deltaxy', 'drloc', 'plz', 'sup']
        if config.MODEL.NUM_CLASSES > 4:
            acc1, acc5 = accuracy(output["sup"], target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output["sup"], target, topk=(1, 2))

        acc1 = reduce_tensor(config, acc1)
        acc5 = reduce_tensor(config, acc5)
        loss = reduce_tensor(config, loss)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f"------------------------Mode: {mode}----------------")
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    logger.info(f'Mode: {mode} : * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

# TODO: save_checkpoint need to be changed

if __name__ == "__main__":
    # _, config = parse_option()
    args = parse_option()
    if not args.debug:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank

    print("\t\t args.local_rank: ", args.local_rank)
    if args.dsets_type == "domainnet":
        # DomainNet datasets
        datasets = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    elif args.dsets_type == "decathlon":
        # Decathlon datasets
        datasets = ["aircraft", "cifar100", "daimlerpedcls", "dtd", "gtsrb", "omniglot", "svhn", "ucf101",
                    "vgg-flowers"]
    elif args.dsets_type == "cifar-10":
        datasets = ["cifar-10"]
    elif args.dsets_type == "cifar-100":
        datasets = ["cifar-100"]
    elif args.dsets_type == "flowers102":
        datasets = ["flowers102"]
    elif args.dsets_type == "svhn":
        datasets = ["svhn"]
    else:
        print("------ Invalid dataset name --------------")
        exit(0)

    root_path = os.getcwd()
    datasets_path = os.path.join(root_path, "datasets", str(args.dsets_type))
    output_folder = os.path.join(root_path, "output", args.exp_name)
    try:
        os.mkdir(output_folder)
    except:
        print("Output folder for {} exists already !".format(args.exp_name))

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # Use args arguments
    if not args.debug:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

    for dset in datasets[1:]:  #[:1]datasets[0:1]
        print("****************** Dataset: {} ******************".format(dset))
        # Dataset path
        if args.dsets_type in ["decathlon", "domainnet"]:
            args.data_path = os.path.join(datasets_path, dset)
        else:
            args.data_path = datasets_path
        args.output = os.path.join(output_folder, dset)
        args.dataset_name = dset
        try:
            os.mkdir(args.output)
        except:
            print("Output folder for {} exists already !".format(args.output))

        config = get_config(args)

        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

        if not config.DEBUG:
            seed = config.SEED + dist.get_rank()
        else:
            seed = config.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        if not config.DEBUG:
            linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
            linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
            linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        else:
            linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
            linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
            linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

        os.makedirs(config.OUTPUT, exist_ok=True)
        if not config.DEBUG:
            logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
            if dist.get_rank() == 0:
                path = os.path.join(config.OUTPUT, "config.json")
                with open(path, "w") as f:
                    f.write(config.dump())
                logger.info(f"Full config saved to {path}")
        else:
            logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        main(config)
        print("------------------------ Done Dataset: {} -------------------".format(dset))
