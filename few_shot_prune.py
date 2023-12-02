# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import copy
import os, math, time, datetime
import argparse
import numpy as np
import pickle
import pdb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.utils.prune as prune
import torch.autograd as autograd

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
    parser.add_argument('--finetune', type=int, default = 0, help='Finetunning mode')
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

    # PRUNING AMOUNT
    parser.add_argument("--prune_type", type=str, default="layerwise")
    parser.add_argument("--prune_struct", type=str, default="structured")
    parser.add_argument("--prune_amount", type=float, default=0.2)
    parser.add_argument("--delta_loss", type=float, default=0.1)
    parser.add_argument("--weighted", type=int, default=0, help="StructLAMP++")
    parser.add_argument("--scaling", type=int, default=1, help="StructLAMP++")
    parser.add_argument("--range", type=int, default=2)

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

def finalize_pruning(model):
    for n, m in model.named_modules():
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")

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

def structured_magnitude_pruning_layer(model, amount, layer_name="parallel_mlp", weighted=1,
                                       type="structured", normalized=1):
    # Make sure that we have only adapters linear layers
    apply_mask_pruning(model, layer_name="parallel_mlp")
    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]
    unmaskeds, total = _count_unmasked_weights(mlist, layer_name, type)
    num_surv = int(np.floor(_count_neurons(mlist).sum() * (1.0 - amount)))  #floor
    stop = False
    if torch.sum(unmaskeds) == 0:
        stop = True
        return stop

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
    unmaskeds, total = _count_unmasked_weights(mlist, layer_name, type="structured")
    stop = False
    if torch.sum(unmaskeds) == 0:  #torch.sum(unmaskeds)
        stop = True
        return stop

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
    unmaskeds, total = _count_unmasked_weights(mlist, layer_name, type="structured")
    stop = False
    if torch.sum(unmaskeds) == 0:  #torch.sum(unmaskeds)
        stop = True
        return stop

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

        prune.random_structured(concat_module, name="weight", amount=float(amount), dim=0)  # rows
        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone(concat_module.weight_mask[:, :max_cols])
        mlist[idx + 1][1].weight, mlist[idx + 1][1].weight_mask = torch.clone(concat_module.weight[:, max_cols:].T), \
                                                          torch.clone(concat_module.weight_mask[:, max_cols:].T)

def structured_gradient_pruning(model, gradient_scores, amount=0.5, layer_name="parallel_mlp"):
    mlist = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]

    for idx in range(0, len(mlist), 2):
        (fc1_n, fc1_m), (fc2_n, fc2_m) = mlist[idx], mlist[idx + 1]

        concat = torch.cat((fc1_m.weight, fc2_m.weight.T), 1)
        concat_module = nn.Linear(concat.shape[0], concat.shape[1], bias=False)
        concat_module.weight = torch.nn.Parameter(concat).cuda()

        param_score1 = gradient_scores[fc1_n]
        param_score2 = gradient_scores[fc2_n]
        param_score = torch.cat((param_score1, param_score2.T), 1)

        prune.ln_structured(concat_module, name="weight", n=1, dim=0, importance_scores=param_score, amount=amount)  # rows
        max_cols = fc1_m.weight.shape[1]
        mlist[idx][1].weight, mlist[idx][1].weight_mask = torch.clone(concat_module.weight[:, :max_cols]), \
                                                          torch.clone(concat_module.weight_mask[:, :max_cols])
        mlist[idx + 1][1].weight, mlist[idx + 1][1].weight_mask = torch.clone(concat_module.weight[:, max_cols:].T), \
                                                          torch.clone(concat_module.weight_mask[:, max_cols:].T)


def GraSP_fetch_data(dataloader, num_classes, samples_per_class, mixup_fn):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            print("x: ", x.shape)
            print("y: ", y.shape)
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

def count_total_parameters(net, layer_name="parallel_mlp"):
    total = 0
    for n, m in net.modules():
        if isinstance(m, nn.Linear) and layer_name in n:
            total += m.weight.numel()
    return total


def count_fc_parameters(net, layer_name="parallel_mlp"):
    total = 0
    for m in net.modules():
        if isinstance(m, nn.Linear) and layer_name in n:
            total += m.weight.numel()
    return total

# Reference: https://github.com/alecwangcq/GraSP
def GradSP_pruning(model, train_dataloader, criteria, mixup_fn, amount=0.5, layer_name="parallel_mlp",
                   num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = 1-amount
    old_net = model

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    # net = copy.deepcopy(model.module)

    weights = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and (layer_name in n) and reinit:
            nn.init.xavier_normal_(m.weight_orig)
            weights.append(m.weight_orig)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_ = True

    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        #inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class, mixup_fn)
        inputs, targets = next(iter(train_dataloader))

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N // 2])
        targets_one.append(dtarget[:N // 2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])

        outputs = model(inputs[:N // 2])
        loss = criteria(outputs["sup"] / T, targets[:N // 2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = model(inputs[N // 2:])
        loss = criteria(outputs["sup"] / T, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).cuda()
        targets = targets_one.pop(0).cuda()
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = model(inputs)
        loss = criteria(outputs["sup"] / T, targets)

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for n, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and layer_name in n:
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1

        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, (n, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Linear) and layer_name in n:
            grads[old_modules[idx]] = -layer.weight_orig.data * layer.weight_orig.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    prunable_layers = [m for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.weight.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))

# Reference: https://github.com/mil-ad/snip/blob/master/snip.py
def snip_pruning(model, train_dataloader, criteria, mixup_fn, amount=0.5, layer_name="parallel_mlp"):
    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))

    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    if mixup_fn is not None:
        inputs, targets = mixup_fn(inputs, targets)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    #net = copy.deepcopy(model.module)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and layer_name in n:
            m.weight_mask = nn.Parameter(torch.ones_like(m.weight))
            nn.init.xavier_normal_(m.weight)
            m.weight.requires_grad_ = False

    # Compute gradients (but don't apply them)
    model.train()
    model.zero_grad()
    outputs = model(inputs)
    loss = criteria(outputs["sup"], targets)
    loss.backward(retain_graph=True)

    grads_abs = [torch.abs(m.weight_mask.grad) for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * amount)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    prunable_layers = [m for n, m in model.named_modules() if isinstance(m, nn.Linear) and layer_name in n]

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))

def main(config):
    # dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = \
    #     build_loader_split(config)

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    logger.info(f"\t\t FINETUNNING Argument: {config.MODEL.FINETUNE}")
    logger.info(f"\t\t DRLOC Argument: {config.TRAIN.USE_DRLOC}")
    logger.info(f"\t\t PRUNING TYPE: {config.PRUNE.TYPE}")
    logger.info(f"\t\t PRUNING layer name: {config.PRUNE.layer_name}")

    current = os.getcwd()
    if config.MODEL.FINETUNE == 6:
        if config.MODEL.TYPE == "swin_adapters" or config.MODEL.TYPE == "swin_adapters_layer":
            logger.info("\t\t Prunning Swin Transformer Using Houlsby Adapters")
            logger.info(f"\t\t TYPE OF ADAPTERS: {config.TRAIN.TYPE_ADAPTERS}")
            logger.info(f"\t\t SIZE OF ADAPTERS: {config.TRAIN.SIZE_ADAPTERS}")
            logger.info(f"\t\t TYPE OF PRUNE: {config.PRUNE.TYPE}")

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
                if "head" in name:
                    param.requires_grad = True

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
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
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

    # We have to add mask with identity pruning to apply pruning methods
    identity_pruning(model, layer_name="parallel_mlp")

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
        print("\t\t config.MODEL.RESUME:", config.MODEL.RESUME)
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, mixup_fn, mode="test")
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
                 "acc1_train": [], "acc5_train": [], "params": n_parameters, "pruning": [], "pruning_epochs": []
                 , "grads" : []}

    init_lambda_drloc = 0.0

    # Early stopping patience; how long to wait after last time validation loss improved.
    patience = 5
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    pruning_amount = config.PRUNE.AMOUNT

    prune_amounts = []
    for phase in range(config.TRAIN.RANGE):
        if config.PRUNE.TYPE == "gradient":
            old_gradients = init_gradients(model, config.PRUNE.layer_name)

        if phase == 0:
            pruning_amount = config.PRUNE.AMOUNT
        else:
            pruning_amount = pruning_amount + (1 - pruning_amount)*config.PRUNE.AMOUNT


        if config.PRUNE.TYPE == "snip":
            snip_pruning(model, data_loader_train, criterion_sup, mixup_fn, amount=pruning_amount,
                                        layer_name=config.PRUNE.layer_name)
        elif config.PRUNE.TYPE == "GradSP":
            GradSP_pruning(model, data_loader_train, criterion_sup, mixup_fn, amount=pruning_amount,
                         layer_name=config.PRUNE.layer_name)

        # We apply same training process (annealing) for 100 epochs
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            if not config.DEBUG:
                data_loader_train.sampler.set_epoch(epoch)
            epc = epoch + (100 * phase)
            # Apply fine-tuning after pruning at initialisation of a trained model of Adapters
            loss, epoch_time, loss_meter_avg, grads = train_one_epoch(config,
                                                               model, criterion_sup, criterion_ssup, data_loader_train,
                                                               optimizer, epoch,
                                                               mixup_fn, lr_scheduler, logger, init_lambda_drloc
                                                               )
            #TODO: store the gradient in a dict to pass them to the pruning function
            if config.PRUNE.TYPE == "gradient":
                old_gradients = accumulate_gradients(old_gradients, grads)

            if not config.DEBUG:
                if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                    save_checkpoint(config, epc, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

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

        if config.PRUNE.TYPE == "gradient":
            logs_dict["grads"].append(old_gradients)

        logger.info(f"\t\t pruning_amount: {pruning_amount}")
        prune_amounts.append(pruning_amount)
        #with torch.no_grad():
        if config.PRUNE.TYPE == "random":
            structured_random_pruning(model, amount=pruning_amount, layer_name=config.PRUNE.layer_name)
        elif config.PRUNE.TYPE == "layerwise":
            logger.info("\t\t Applying Global layer-wise magnitude pruning")
            stop_prune = structured_magnitude_pruning_layer(model, amount=pruning_amount,
                                    layer_name=config.PRUNE.layer_name, weighted=config.PRUNE.WEIGHTED,
                                    normalized=config.PRUNE.scaling, type=config.PRUNE.STRUCT)
            if stop_prune:
                break
        elif config.PRUNE.TYPE == "down_magnitude":
            logger.info("\t\t Applying magnitude pruning on downsampling")
            stop_prune = structured_down_magnitude_pruning(model, amount=pruning_amount,
                                                     layer_name=config.PRUNE.layer_name)
        elif config.PRUNE.TYPE == "magnitude":
            logger.info("\t\t Applying local magnitude pruning")
            stop_prune = structured_magnitude_pruning(model, amount=pruning_amount,
                                                      layer_name=config.PRUNE.layer_name)
        elif config.PRUNE.TYPE == "gradient":
            structured_gradient_pruning(model, amount=pruning_amount, gradient_scores=old_gradients,
                                        layer_name=config.PRUNE.layer_name)
        else:
            pass
        epoch = 0
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            save_checkpoint(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, "pruned")

        logs_dict["pruning"].append(epoch)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    logs_dict["training_time"] = total_time

    # Save trainings logs
    logs_path = os.path.join(config.OUTPUT, "logs.pkl")

    with open(logs_path, "wb") as handle:
        pickle.dump(logs_dict, handle)
    handle.close()

    torch.cuda.synchronize()
    # # Fuse the model
    #finalize_pruning(model)
    if dist.get_rank() == 0:
        save_checkpoint(config, 0, model.module, max_accuracy, optimizer, lr_scheduler, logger, "last_fused")

    del dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
    del model, optimizer, model_without_ddp
    torch.cuda.empty_cache()

def init_gradients(model, param_name="parallel_mlp"):
    gradients = {}
    for name, param in model.named_parameters():
        if param_name in name:
            name = ".".join([str(item) for item in name.split(".")[:-1]])
            gradients[name] = torch.zeros_like(param)

    return gradients

def get_gradients(model, param_name="parallel_mlp"):
    gradients = {}
    for name, param in model.named_parameters():
        if param_name in name:
            name = ".".join([str(item) for item in name.split(".")[:-1]])
            gradients[name] = param.grad
    
    return gradients

def accumulate_gradients(old_grads, new_grads):
    assert len(old_grads.keys()) == len(new_grads.keys())
    for param_name in old_grads.keys():
        new_grads[param_name] += old_grads[param_name]

    return new_grads

def train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader, optimizer, epoch, mixup_fn,lr_scheduler,logger,lambda_drloc):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    if not args.debug:
        device = torch.device('cuda', dist.get_rank())

    end_time_tmp = time.time()
    logger.info(f"\t\t Number of classes: {config.MODEL.NUM_CLASSES}")

    gradients = None
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
                # Get gradients for gradient-based pruning
                if config.PRUNE.TYPE == "gradient":
                    gradients = get_gradients(model, param_name=config.PRUNE.layer_name)

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

            # Get gradients for gradient-based pruning
            if config.PRUNE.TYPE == "gradient":
                gradients = get_gradients(model, param_name=config.PRUNE.layer_name)
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
    return loss, epoch_time, loss_meter.avg, gradients

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

    for dset in datasets:
        print("****************** Dataset: {} ******************".format(dset))
        # Dataset path
        if args.dsets_type in ["decathlon", "domainnet"]:
            args.data_path = os.path.join(datasets_path, dset)
        # elif args.dsets_type in ["fvgc", "vtab"]:
        #     args.data_path = os.path.join("/data", args.dsets_type, dset)
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
