# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os, math, time, datetime
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.utils.prune as prune

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
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor
)

from drloc import cal_selfsupervised_loss

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
    parser.add_argument('--finetune', type=int, default=0, help='Finetunning mode')
    parser.add_argument('--pretrained_model', type=str, help='Experiment name with in output folder')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False,
                        help='local rank for DistributedDataParallel')
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

    # Pruning
    parser.add_argument("--prune_layer", type=str, default="parallel_mlp")

    # Hyperparameters
    parser.add_argument("--lmbda", default=0, type=float, help="Sensitivity lambda. Default = 0.0001.")
    parser.add_argument("--twt", default=0, type=float, help="Threshold worsening tolerance. Default = 0.")
    parser.add_argument("--pwe", default=0, type=int, help="Plateau waiting epochs. Default = 0.")
    parser.add_argument("--mom", type=float, default=0, help="Momentum. Default = 0")
    parser.add_argument("--nesterov", default=False, action="store_true",
                        help="Use Nesterov momentum. Default = False.")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay. Default = 0.")
    parser.add_argument("--adam", default=False, action="store_true", help="Use ADAM optimizer. Default = False.")

    # Sensitivity optimizer
    parser.add_argument("--sensitivity", type=str, choices=["neuron-lobster"], default="neuron-lobster",
                        help="Sensitivty optimizer.")
    parser.add_argument("--rescale", default=False, action="store_true", help="Rescale the sensitivity value.")
    parser.add_argument("--no_prune", default=False, action="store_true", help="Disable the pruning procedure.")
    parser.add_argument("--cluster", default=False, action="store_true", help="Enable clustering.")

    # Parameters decay
    parser.add_argument("--decay_half", default=50, type=int, help="Exponential decay half-life. Default = 50.")
    parser.add_argument("--decay_step", type=int, default=10, help="Decay step size. Default = 10.")
    parser.add_argument("--decay_stop", type=float, default=1e-3,
                        help="Stop condition, interrupts the training procedure when the decay value falls below it.")
    parser.add_argument("--decay_lr", default=False, action="store_true", help="Decay lr. Default = False.")
    parser.add_argument("--decay_wd", default=False, action="store_true", help="Decay wd. Default = False.")
    parser.add_argument("--decay_lmbda", default=False, action="store_true", help="Decay lambda. Default = False.")
    parser.add_argument("--load_best", default=False, action="store_true", help="Load best model before pruning.")
    parser.add_argument("--rollback", default=False, action="store_true", help="Load best model before pruning.")

    # Masks
    parser.add_argument("--mask_params", default=False, action="store_true",
                        help="Pruned parameters mask. Default = False.")
    parser.add_argument("--mask_neurons", default=True, action="store_true",
                        help="Pruned neurons mask. Default = False.")
    parser.add_argument("--bn_prune", default=False, action="store_true",
                        help="Prune batchnorm and ignore previous conv.")

    # PRUNING AMOUNT
    parser.add_argument("--prune_type", type=str, default="magnitude")
    parser.add_argument("--prune_struct", type=str, default="structured")
    parser.add_argument("--prune_amount", type=float, default=0.2)
    parser.add_argument("--delta_loss", type=float, default=0.1)
    parser.add_argument("--weighted", type=int, default=1, help="StructLAMP++")
    parser.add_argument("--scaling", type=int, default=1, help="StructLAMP++")
    parser.add_argument("--range", type=int, default=2)

    # Debugging mode
    parser.add_argument("--debug", default=False, help="Debugging Mode. Default = False.")

    # seed
    parser.add_argument("--seed", type=int, default=0)
    args, unparsed = parser.parse_known_args()

    return args


def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight


def identity_pruning(model, layer_name="parallel_mlp"):
    for (n, m) in model.named_modules():
        if isinstance(m, nn.Linear) and (layer_name in n):
            prune.identity(m, name="weight")

def apply_mask_pruning(model, layer_name="parallel_mlp"):
    for (n, m) in model.named_modules():
        if isinstance(m, nn.Linear) and (layer_name in n):
            #m.weight = torch.nn.Parameter(torch.tensor(m.weight_orig * m.weight_mask, device="cuda"))
            m.weight = m.weight * m.weight_mask

def _count_unmasked_weights(mlist, layer_name="parallel_mlp", type="structured"):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    unmaskeds, total = [], []

    for i in range(0, len(mlist), 2):
        # We loop through only down-sampling layers: fc1
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[i], mlist[i+1]

        concat_mask = torch.cat((fc1_m.weight_mask, fc2_m.weight_mask.T), 1)
        assert concat_mask.shape[0] == fc1_m.weight_mask.shape[0]
        assert concat_mask.shape[1] == 2 * fc1_m.weight_mask.shape[1]
        # Sum over rows:
        rows = torch.sum(torch.abs(concat_mask), 1) != 0
        added = torch.sum(rows)
        unmaskeds.append(added)
        total.append(fc1_m.weight_mask.shape[0])

    unmaskeds = torch.FloatTensor(unmaskeds)
    total = torch.FloatTensor(total)
    return unmaskeds, total

def _count_neurons(mlist):
    total = []

    for i in range(0, len(mlist), 2):
        # We loop through only down-sampling layers: fc1
        (fc1_n, fc1_m), _ = mlist[i], mlist[i+1]
        added = fc1_m.weight.shape[0]
        total.append(added)

    unmaskeds = torch.FloatTensor(total)
    return unmaskeds

def get_weights(model, layer_name="parallel_mlp"):
    weights, names = [], []
    for (n, m) in model.named_modules():
        if isinstance(m, nn.Linear) and (layer_name in n):
            weight = m.weight_orig * m.weight_mask
            weights.append(weight)
            names.append(n)

    return weights, names

def structured_magnitude_pruning_layer(model, amount, layer_name="parallel_mlp", weighted=0,
                                       type="structured", normalized=1):
    # Make sure that we have only adapters linear layers
    apply_mask_pruning(model, layer_name="parallel_mlp")
    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]
    unmaskeds, total = _count_unmasked_weights(mlist, layer_name, type)
    num_surv = int(np.floor(_count_neurons(mlist).sum() * (1.0 - amount)))  #floor
    stop = False
    # if torch.sum(unmaskeds) == 0:
    #     stop = True
    #     return stop

    weights, names = get_weights(model, layer_name)
    flattened_scores = []

    for idx in range(0, len(weights), 2):
        fc1_m, fc2_m = weights[idx], weights[idx+1]
        concat = torch.cat((fc1_m, fc2_m.T), 1)
        w = torch.sum(torch.abs(concat), 1)
        if normalized:
            w /= concat.shape[1]
        flattened_scores.append(w.view(-1))

    flattened_scores_temp = []
    # flattened_scores = [score / max(score) for score in flattened_scores] # zero condi
    # Normalize it
    if weighted == 1:
        for score in flattened_scores:
            maxi = max(score)
            tmp = []
            # Handle the case where, no neurons are left: unmasks = 0
            for item in score:
                if item != 0:
                    item = item / maxi
                tmp.append(item)
            flattened_scores_temp.append(torch.Tensor(tmp))

        flattened_scores = flattened_scores_temp

    sorted = torch.cat(flattened_scores, dim=0)
    if num_surv == 0:
        threshold = torch.Tensor([np.inf]).cuda() #float('inf')
    else:
        topks, _ = torch.topk(sorted, num_surv)
        threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum()
                   for score in flattened_scores]
    assert len(final_survs) == len(unmaskeds)

    amounts = []
    for i, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / total[i]))

    counter = 0
    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]
    for idx in range(0, len(mlist), 2):
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[idx], mlist[idx + 1]
        concat = torch.cat((fc1_m.weight, fc2_m.weight.T), 1)
        concat_module = nn.Linear(concat.shape[0], concat.shape[1], bias=False)
        concat_module.weight = torch.nn.Parameter(concat).cuda()
        prune.ln_structured(concat_module, name="weight", amount=float(amounts[counter]), n=1, dim=0)

        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone( concat_module.weight_mask[:, :max_cols])

        mlist[idx + 1][1].weight, mlist[idx + 1][1].weight_mask = torch.clone(concat_module.weight[:, max_cols:].T), \
                                                                  torch.clone(concat_module.weight_mask[:, max_cols:].T)

        counter += 1

def structured_down_magnitude_pruning(model, amount, layer_name="parallel_mlp"):
    # Apply pruning masks to get the right weights
    apply_mask_pruning(model, layer_name="parallel_mlp")

    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]
    # unmaskeds, total = _count_unmasked_weights(mlist, layer_name, type="structured")
    # stop = False
    # if torch.sum(unmaskeds) == 0:  #torch.sum(unmaskeds)
    #     stop = True
    #     return stop

    for idx in range(0, len(mlist), 2):
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[idx], mlist[idx + 1]

        concat = fc1_m.weight
        concat_module = nn.Linear(concat.shape[0], concat.shape[1], bias=False)
        concat_module.weight = torch.nn.Parameter(concat).cuda()

        prune.ln_structured(concat_module, name="weight", amount=float(amount), n=1, dim=0)  # rows
        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone(concat_module.weight_mask[:, :max_cols])

def structured_magnitude_pruning(model, amount, layer_name="parallel_mlp"):
    # Apply pruning masks to get the right weights
    apply_mask_pruning(model, layer_name="parallel_mlp")

    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]
    # unmaskeds, total = _count_unmasked_weights(mlist, layer_name)
    # stop = False
    # if torch.sum(unmaskeds) == 0:  #torch.sum(unmaskeds)
    #     stop = True
    #     return stop

    for idx in range(0, len(mlist), 2):
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[idx], mlist[idx + 1]

        concat = torch.cat((fc1_m.weight, fc2_m.weight.T), 1)

        concat_module = nn.Linear(concat.shape[0], concat.shape[1], bias=False)
        concat_module.weight = torch.nn.Parameter(concat).cuda()

        prune.ln_structured(concat_module, name="weight", amount=float(amount), n=1, dim=0)  # rows
        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone(concat_module.weight_mask[:, :max_cols])
        mlist[idx+1][1].weight, mlist[idx + 1][1].weight_mask = torch.clone(concat_module.weight[:, max_cols:].T), \
                                                                  torch.clone(concat_module.weight_mask[:, max_cols:].T)


def structured_random_pruning(model, amount, layer_name="parallel_mlp"):
    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]

    for idx in range(0, len(mlist), 2):
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[idx], mlist[idx + 1]

        concat = torch.cat((fc1_m.weight, fc2_m.weight.T), 1)
        concat_module = nn.Linear(concat.shape[0], concat.shape[1], bias=False)
        concat_module.weight = torch.nn.Parameter(concat).cuda()

        prune.random_structured(concat_module, name="weight", amount=float(amount), n=1)  # rows
        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone(concat_module.weight_mask[:, :max_cols])
        mlist[idx + 1][1].weight, mlist[idx + 1][1].weight_mask = torch.clone(concat_module.weight[:, max_cols:].T), \
                                                                  torch.clone(concat_module.weight_mask[:, max_cols:].T)


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"\t\t Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger.info(f"\t\t FINETUNNING Argument: {config.MODEL.FINETUNE}")
    logger.info(f"\t\t DRLOC Argument: {config.TRAIN.USE_DRLOC}")
    logger.info(f"\t\t PRUNING TYPE: {config.PRUNE.TYPE}")
    logger.info(f"\t\t PRUNING layer name: {config.PRUNE.layer_name}")

    current = os.getcwd()
    if config.MODEL.FINETUNE == 3:
        if config.MODEL.TYPE == "swin_adapters":
            logger.info("\t\t Finetunning Swin Transformer Using Houlsbi Adapters")
            logger.info(f"\t\t TYPE OF ADAPTERS: {config.TRAIN.TYPE_ADAPTERS}")
            logger.info(f"\t\t SIZE OF ADAPTERS: {config.TRAIN.SIZE_ADAPTERS}")

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

        if config.MODEL.TYPE == "cvt_adapters":
            logger.info("\t\t Finetunning CvT Transformer Using Houlsbi Adapters")
            logger.info(f"\t\t TYPE OF ADAPTERS: {config.TRAIN.TYPE_ADAPTERS}")
            logger.info(f"\t\t SIZE OF ADAPTERS: {config.TRAIN.SIZE_ADAPTERS}")
            logger.info(f"\t\t STAGEs SIZE OF ADAPTERS: {config.MODEL.SWIN.PARAM_RATIOS}")

            pre_model_path = os.path.join(current, "pretrained", "CvT-13-224x224-IN-1k" + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.dim_embed, 1000)
            model.head = classifier
            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint, strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.dim_embed, config.MODEL.NUM_CLASSES)
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

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\t\t After Number of params: {n_parameters}")
    model.cuda()

    # logger.info(str(model))
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if not config.DEBUG:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

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
    logs_dict = {"loss": [], "loss_val": [], "loss_test": [], "epoch_time": [], "loss_avg": [], "acc1": [], "acc5": [],
                 "acc1_test": [], "acc5_test": [], "acc1_val": [], "acc5_val": [],
                 "acc1_train": [], "acc5_train": [], "params": n_parameters, "pruning": [], "pruning_epochs": []}

    init_lambda_drloc = 0.0
    # We have to add mask with identity pruning to apply pruning methods
    identity_pruning(model, layer_name="parallel_mlp")

    # Early stopping patience; how long to wait after last time validation loss improved.
    patience = 5
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for phase in range(config.TRAIN.RANGE):
        if phase == 1:
            with torch.no_grad():
                if config.PRUNE.TYPE == "random":
                    structured_random_pruning(model, amount=config.PRUNE.AMOUNT, layer_name=config.PRUNE.layer_name)
                elif config.PRUNE.TYPE == "layerwise":
                    logger.info("\t\t Applying Global layer-wise magnitude pruning")
                    stop_prune = structured_magnitude_pruning_layer(model, amount=config.PRUNE.AMOUNT,
                                     layer_name=config.PRUNE.layer_name, weighted=config.PRUNE.WEIGHTED,
                                     normalized=config.PRUNE.scaling, type=config.PRUNE.STRUCT)
                    # if stop_prune:
                    #     break
                elif config.PRUNE.TYPE == "down_magnitude":
                    logger.info("\t\t Applying magnitude pruning on downsampling")
                    stop_prune = structured_down_magnitude_pruning(model, amount=config.PRUNE.AMOUNT,
                                                                   layer_name=config.PRUNE.layer_name)
                elif config.PRUNE.TYPE == "magnitude":
                    logger.info("\t\t Applying local magnitude pruning")
                    stop_prune = structured_magnitude_pruning(model, amount=config.PRUNE.AMOUNT,
                                                              layer_name=config.PRUNE.layer_name)
                else:
                    raise NotImplementedError("Please, enter a correct pruning method!")

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            if not config.DEBUG:
                data_loader_train.sampler.set_epoch(epoch)

            loss, epoch_time, loss_meter_avg = train_one_epoch(config, model, criterion_sup, criterion_ssup,
                                                               data_loader_train, optimizer, epoch,
                                                               mixup_fn, lr_scheduler, logger, init_lambda_drloc
                                                               )
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            acc1_train, acc5_train, loss_train = validate(config, data_loader_train, model, mixup_fn, mode="train")
            acc1, acc5, loss_val = validate(config, data_loader_val, model, mixup_fn, mode="test")

            logger.info(f"Training Accuracy of the network on the {len(dataset_train)} train images: {acc1_train:.5f}%")
            logger.info(f"Validation Accuracy of the network on the {len(dataset_val)} test images: {acc1:.5f}%")

            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            logs_dict["loss"].append(loss_train)
            logs_dict["epoch_time"].append(epoch_time)
            logs_dict["loss_avg"].append(loss_meter_avg)
            logs_dict["acc1_train"].append(acc1_train)
            logs_dict["acc5_train"].append(acc5_train)

            logs_dict["acc1_test"].append(acc1)
            logs_dict["acc5_test"].append(acc5)
            logs_dict["loss_test"].append(loss_val)

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

def train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                    logger, lambda_drloc):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    if not config.DEBUG:
        device = torch.device('cuda', dist.get_rank())

    end_time_tmp = time.time()
    logger.info(f"\t\t Number of classes: {config.MODEL.NUM_CLASSES}")

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        copy_targets = torch.clone(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # targets = torch.nn.functional.one_hot(targets, num_classes=config.MODEL.NUM_CLASSES)

        # with torch.cuda.amp.autocast():
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
def validate(config, data_loader, model, mixup_fn, mode="test"):
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

    for dset in datasets:  # [:1]
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
