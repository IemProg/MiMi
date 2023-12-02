# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import config
from prune import validate
import csv
from copy import deepcopy
from EIDOSearch.utils import save_and_zip_model

from EIDOSearch.evaluation import test_model, architecture_stat

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_checkpoint_ft(pretrain_path, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Loading pretrained model form {pretrain_path}....................")
    state_dict = torch.load(pretrain_path, map_location='cpu')['model']
    own_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state_dict and "head" not in name:
            own_state_dict[name].copy_(param)
    logger.info(f"=> loaded successfully")
    max_accuracy = 0.0
    del state_dict
    torch.cuda.empty_cache()
    return max_accuracy

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, prefix= None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    if prefix is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'{prefix}_ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_checkpoint_best(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    
    save_path = os.path.join(config.OUTPUT, 'best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(config, tensor):
    rt = tensor.clone()
    if not config.DEBUG:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
    return rt

def print_good_bad_epochs(config, epoch, delta, bound, epoch_ok):
    csv_file_name = "good_bad_epochs.csv"
    cvs_path = os.path.join(config.OUTPUT, csv_file_name)

    vals = [epoch, delta, bound, config.PRUNE.TWT, epoch_ok]
    if not os.path.exists(cvs_path):
        titles = ["iteration", "delta", "bound", "twt", "ok"]
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(titles)
            writer.writerow(vals)
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(vals)

def print_best(config, best_epoch, new_epoch, best_loss, new_loss, newly_pruned):
    csv_file_name = "best.csv"
    cvs_path = os.path.join(config.OUTPUT, csv_file_name)
    vals = [best_epoch, new_epoch, best_loss, new_loss, newly_pruned]

    if not os.path.exists(cvs_path):
        titles = ["from", "to", "old_best", "new_best", "from_pruning"]
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(titles)
            writer.writerow(vals)
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(vals)

def print_hyp(config, epoch, lr, wd, mom, lmbda, twt):
    csv_file_name = "hyp.csv"
    cvs_path = os.path.join(config.OUTPUT, csv_file_name)

    vals = [epoch, lr, wd, mom, lmbda, twt]

    if not os.path.exists(cvs_path):
        titles = ["epoch", "lr", "wd", "mom", "lmbda", "twt"]
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(titles)
            writer.writerow(vals)
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(vals)

def print_prune(config, epoch, perf_pre, perf_post, best_epoch, prune_twt, csv_file_name):
    cvs_path = os.path.join(config.OUTPUT, csv_file_name)

    vals = [epoch, perf_pre, perf_post,
            (100 - perf_pre) * (1 + prune_twt), prune_twt, best_epoch]

    if not os.path.exists(cvs_path):
        titles = ["iteration", "acc1_pre", "acc5_pre", "loss_pre", "acc1_post", "acc5_post", "loss_post", "bound",
                  "twt", "from"]
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(titles)
            writer.writerow(vals)
    else:
        with open(cvs_path, mode='a') as runs_file:
            writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
            writer.writerow(vals)

@torch.no_grad()
def log_statistics(config, epoch, model, pruning_stat, train_performance, valid_performance, test_performance, init_macs,
                   macs, lr, wd, mom, lmbda, twt, decay, top_acc, cr_data, tb_writer, wdb_writer, newly_pruned, task):
    # print_epoch_stat(args, epoch, pruning_stat, train_performance, valid_performance, test_performance, init_macs, macs,
    #                  lr, wd, mom, lmbda, twt, decay, tb_writer, wdb_writer, task)

    if newly_pruned:
        with open(os.path.join(config.OUTPUT, "progression_log.txt"), "a") as prog_file:
            prog_file.write("\n")

        top_acc = 0

        # Print data of previous CR
        print_data(config, cr_data)

    if valid_performance > top_acc:
        top_acc = valid_performance

        with open(os.path.join(config.OUTPUT, "progression_log.txt"), "a") as prog_file:
            prog_file.write("CR: {:<20} Top-1 Train: {:<20} Top-1 Valid: {:<20} Top-1 Test: {:<20} epoch: {:<20}\n"
                            .format(pruning_stat["network_param_ratio"], train_performance, valid_performance,
                                    test_performance, epoch))
        cr_data = {
            "epoch": epoch,
            "train_performance": train_performance,
            "valid_performance": valid_performance,
            "test_performance": test_performance,
            "macs": macs,
            "pruning_stat": pruning_stat,
            "lr": lr,
            "model": deepcopy(model)
        }

    return top_acc, cr_data


def print_data(config, cr_data):
    with open(os.path.join(config.OUTPUT, "log.txt"), "a") as cr_file:
        try:
            cr_file.write("Epoch: {}\n".format(cr_data["epoch"]))
            cr_file.write("Train @1 (%): {:.2f}\n".format(cr_data["train_performance"]))
            cr_file.write("Validation @1 (%): {:.2f}\n".format(cr_data["valid_performance"]))
            cr_file.write("Test @1 (%): {:.2f}\n".format(cr_data["test_performance"]))
            cr_file.write("Test @5 (%): {:.2f}\n".format(cr_data["test_performance"]))
            cr_file.write("MACs: {:.2f}\n".format(cr_data["macs"]))
            cr_file.write("Neurons CR: {:.2f}\n".format(cr_data["pruning_stat"]["network_neuron_ratio"]))
            cr_file.write(
                "Remaining neurons (%): {:.2f}\n".format(cr_data["pruning_stat"]["network_neuron_non_zero_perc"]))
            cr_file.write("Parameters CR: {:.2f}\n".format(cr_data["pruning_stat"]["network_param_ratio"]))
            cr_file.write(
                "Remaining parameters (%): {:.2f}\n".format(cr_data["pruning_stat"]["network_param_non_zero_perc"]))
            cr_file.write("Learning Rate: {}\n".format(cr_data["lr"]))
            cr_file.write("=" * 20 + "\n\n")

            save_and_zip_model(cr_data["model"],
                               os.path.join(config.OUTPUT,
                                            "{}.pt".format(cr_data["epoch"])))
        except:
            pass


def get_and_save_statistics(config, epoch, model, loss_function,
                            train_loader, valid_performance, test_loader, lr, wd, mom, lmbda, twt, decay,
                            top_acc, cr_data, device, tb_writer, wdb_writer, newly_pruned, task, dummy_input,
                            init_macs):
    pruning_stat = architecture_stat(model)
    _, _, train_performance = validate(config, test_loader, model, verbose=False)
    _, _, test_performance = validate(config, test_loader, model, verbose = False)
    print("\t\t Testing Performance: ", test_performance)
    print("\t\t Testing Performance: ", test_performance)

    #macs, _ = profile(model, dummy_input, verbose=False)
    macs = 0
    macs = 0 #sum(macs.values())
    # top_acc, cr_data = log_statistics(config, epoch, model, pruning_stat, train_performance,
    #                                   valid_performance,
    #                                   test_performance, init_macs, macs , lr, wd, mom, lmbda, twt, decay,
    #                                   top_acc, cr_data, tb_writer, wdb_writer, newly_pruned, task)

    return top_acc, cr_data


def save_last(config, model, epoch, twt_decay_iteration, decay, model_update, bad_epochs, best_epoch, best_error):
    data = {"model": model.state_dict(),
            "epoch": epoch,
            "twt_decay_iteration": twt_decay_iteration,
            "decay": decay,
            "model_update": model_update,
            "bad_epochs": bad_epochs,
            "best_epoch": best_epoch,
            "best_error": best_error}

    torch.save(data, os.path.join(config.OUTPUT, "last.pt"))

class CosineDecay:
    def __init__(self, min, max, steps):
        self.min = min
        self.max = max
        self.steps = steps

    def __call__(self, step):
        return self.min + ((self.max - self.min) * (1 + math.cos(step / self.steps * math.pi))) / 2


class ExponentialDecay:
    def __init__(self, half_life):
        self.steps = int(half_life / math.log(2))

    def __call__(self, step):
        return math.pow(math.e, -step / self.steps)


class StepDecay:
    def __init__(self, min, max, steps):
        self.min = min
        self.max = max
        self.steps = steps

    def __call__(self, step):
        return self.max if (math.floor(step / self.steps) % 2 == 0) else self.min


def get_tb_writer(config):
    tb_name = "lr_{}".format(config.TRAIN.BASE_LR)
    tb_name += "_lam_{}".format(config.PRUNE.LMBDA) if config.PRUNE.sensitivity == "neuron-lobster" else "_lam_0"
    if not config.PRUNE.decay_lmbda:
        tb_name += "_const"
    tb_name += "_wd_{}".format(config.PRUNE.WD)
    if not config.PRUNE.decay_wd:
        tb_name += "_const"
    tb_name += "_step_{}".format(config.PRUNE.decay_step)

    path = os.path.join(config.OUTPUT)
    tb_writer = SummaryWriter(path)
    wdb_writer = None

    return tb_writer, wdb_writer