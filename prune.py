import os, sys
import time
import argparse
import datetime
import numpy as np
import pickle
from copy import deepcopy

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import *

from drloc import cal_selfsupervised_loss

# import EarlyStopping
from pytorchtools import EarlyStopping

from EIDOSearch.pruning.sensitivity import NeuronLOBSTER
from EIDOSearch.pruning import get_model_mask_parameters, get_model_mask_neurons
from EIDOSearch.pruning.clustering import weights_clustering_local as cluster_scheduler
from EIDOSearch.pruning.thresholding import threshold_scheduler_sensitivity as threshold_scheduler
from EIDOSearch.utils import save_and_zip_model
from torch.cuda.amp import GradScaler, autocast

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

# Layers considered during regularization and pruning
LOGS_ROOT = "logs"
def get_layers():
    regu_layers, prune_layers = (nn.Linear),  (nn.Linear)

    return regu_layers, prune_layers

def get_masks(config, model):
    mask_params = get_model_mask_parameters(model, get_layers()[1], config.PRUNE.layer_name) if config.PRUNE.mask_params else None
    mask_neurons = get_model_mask_neurons(model, get_layers()[1], config.PRUNE.layer_name) if config.PRUNE.mask_neurons else None

    return mask_params, mask_neurons

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
    parser.add_argument("--total_epochs", type=int, default=300)

    parser.add_argument("--type_adapters", type=str, default="parallel")
    parser.add_argument("--size_adapters", type=int, default=32)

    # Pruning
    parser.add_argument("--prune_layer", type=str, default="parallel_mlp")
    # Hyperparameters
    parser.add_argument("--lmbda", default=0, type=float, help="Sensitivity lambda. Default = 0.0001.")
    parser.add_argument("--twt", default=0, type=float, help="Threshold worsening tolerance. Default = 0.")
    parser.add_argument("--pwe", default=0, type=int, help="Plateau waiting epochs. Default = 0.")
    parser.add_argument("--mom", type=float, default=0, help="Momentum. Default = 0")
    parser.add_argument("--nesterov", default=False, action="store_true", help="Use Nesterov momentum. Default = False.")
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
    parser.add_argument("--prune_type", type=str, default="random")
    parser.add_argument("--prune_amount", type=float, default=0.2)

    # Debugging mode
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Pruned parameters mask. Default = False.")
    args, unparsed = parser.parse_known_args()

    # config = get_config(args)
    # TODO: fix the use_drlov argument : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    return args  # , config

def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight

def clustering_step(CS, performance):
    print("Clustering")
    CS.step()

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

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
            classifier = nn.Linear(model.dim, 1000)
            model.mlp_head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.dim, config.MODEL.NUM_CLASSES)
            model.mlp_head = classifier
            model.requires_grad = True
            # TODO: Make it dynamic
        elif config.MODEL.TYPE == "t2t":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint for T2T-14-224 finetunning:{msg}")
            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
        elif config.MODEL.TYPE == "residualResnet26":
            model = build_model(config)
            logger.info("\t\t Finetunning Residual resnet 26")
            pass
        else:
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')

            # TODO: make it dynamic
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            #msg = model.load_state_dict(checkpoint, strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            for param in model.parameters():
                param.requires_grad = True
            logger.info("\t\t Finetunning {} Transformer".format(config.MODEL.TYPE))
    elif config.MODEL.FINETUNE == 3:
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
            msg = model.load_state_dict(checkpoint['model'], strict=True)

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
                if "adapt_scale" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
    elif config.MODEL.FINETUNE == -1:
        logger.info("\t\t Training from Scratch")
        model = build_model(config)
    else:
        raise NotImplementedError("\t\t Choose correct fine-tuning method!")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\t\t After Number of params: {n_parameters}")
    model.cuda()

    # logger.info(str(model))
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if config.DEBUG:
        model_without_ddp = model
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module

    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

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
    # logs for training
    logs_dict = {"loss": [], "loss_test": [], "epoch_time": [], "loss_avg": [], "acc1": [], "acc5": [],
                 "acc1_train": [], "acc5_train": [], "params": n_parameters}

    init_lambda_drloc = 0.0
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize the training procedure and get initial values
    resume = None

    if config.DEBUG:
        device = torch.device('cuda0')
    else:
        device = torch.device('cuda', dist.get_rank())

    sensitivity_optimizer = get_sensitivity_optimizer(config, model, device)
    # SummaryWriter
    tb_writer, wdb_writer = get_tb_writer(config)
    dummy_size = (1, 1, 224, 224)
    dummy_input = torch.rand(dummy_size)

    macs, _ = 0, 0 #profile(model, dummy_input)
    init_macs = 0  #sum(macs.values())
    loss_function, cross_valid, top_cr, top_acc, cr_data, newly_pruned, TS, CS, twt_decay_function, \
    best_error, best_epoch, epoch, temperature, bad_epochs, task, resume = \
        init_train(config, model, criterion_sup, optimizer, sensitivity_optimizer,
                   data_loader_train, data_loader_train, data_loader_val, tb_writer, wdb_writer, dummy_input, init_macs, resume, device)

    decayed_temp = False
    reset_bad = False

    best_sd = deepcopy(model.state_dict())
    previous_sd = deepcopy(model.state_dict())

    twt_decay_iteration = resume["twt_decay_iteration"] + 1 if resume is not None else 0
    model_update = resume["model_update"] if resume and "model_update" in resume is not None else False

    init_twt = config.PRUNE.TWT

    scaler = GradScaler() if config.AMP_OPT_LEVEL else None
    if sensitivity_optimizer is not None:
        sensitivity_optimizer.set_scaler(scaler)

    decay_steps_count = 0

    # Epochs
    logger.info("\t\t Start Training ...")
    start_time = time.time()
    while epoch < config.TRAIN.EPOCHS:
        data_loader_train.sampler.set_epoch(epoch)
        # Get parameters/neurons masks
        mask_params, mask_neurons = get_masks(config, model)

        # Decay parameters
        decay_parameters(config, temperature, sensitivity_optimizer)
        model.train()

        # Reduce `twt`
        config.defrost()
        config.PRUNE.TWT = init_twt * twt_decay_function(twt_decay_iteration)
        config.freeze()
        lr, wd, mom, lmbda, twt = get_current_hyperparameters(config, optimizer, sensitivity_optimizer)

        # Get and save epoch statistics
        print_hyp(config, epoch, lr, wd, mom, lmbda, twt)
        loss, epoch_time, loss_meter_avg = train_one_epoch(config, model, criterion_sup, criterion_ssup,
                                                           data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                                                           logger, init_lambda_drloc, sensitivity_optimizer,
                                                           mask_params, mask_neurons)

        # Test this epoch model
        acc1_train, acc5_train, loss_train = validate(config, data_loader_train, model, mode="Training", verbose=False)
        logger.info(f"\t\t Training Loss: {loss_train}")
        logger.info(f"\t\t Training Accuracy: {acc1_train}")

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        # Validation on training dataset
        acc1, acc5, loss_test = validate(config, data_loader_val, model, mode="test")

        logger.info(f"Training Accuracy of the network on the {len(dataset_train)} train images: {acc1_train:.5f}%")
        logger.info(f"Validation Accuracy of the network on the {len(dataset_val)} test images: {acc1:.5f}%")

        # if dist.get_rank() == 0 and acc1 > max_accuracy:
        #    save_checkpoint_best(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        logs_dict["loss"].append(loss_train)
        logs_dict["epoch_time"].append(epoch_time)
        logs_dict["loss_avg"].append(loss_meter_avg)
        logs_dict["acc1_train"].append(acc1_train)
        logs_dict["acc5_train"].append(acc5_train)

        logs_dict["acc1"].append(acc1)
        logs_dict["acc5"].append(acc5)
        logs_dict["loss_test"].append(loss_test)

        bound = best_error * config.PRUNE.TWT
        delta = loss_train - best_error
        epoch_ok = delta < bound and delta != 0
        print_good_bad_epochs(config, epoch, delta, bound, epoch_ok)

        # The loss worsened by more than twt with respect to the lowest
        if not epoch_ok:
            bad_epochs += 1
            if config.PRUNE.rollback:
                model.load_state_dict(previous_sd)
                if not bad_epochs > config.PRUNE.PWE:
                    continue

            # We had more consecutive bad epochs that the patience, i.e. we found a plateau
            if bad_epochs > config.PRUNE.PWE:
                print("Patience over, bad epoch: {} accepted model: {}".format(bad_epochs, model_update))
                with open(os.path.join(config.OUTPUT, "patience.txt"), "a") as patience_txt:
                    patience_txt.write(
                        "{} - patience over - bad epoch {} - accepted model {}".format(epoch, bad_epochs, model_update))

                # Before the plateau we found at least one good model
                if config.PRUNE.NO_PRUNE:
                    temperature /= config.PRUNE.decay_step
                    config.PRUNE.TWT = init_twt
                    twt_decay_iteration = 0
                    decayed_temp = True
                else:
                    if model_update:
                        # Load best model (lowest error)
                        if config.PRUNE.load_best:
                            model.load_state_dict(best_sd)
                            valid_performance, _, loss_train = validate(config, data_loader_train, model)
                            logger.info(f"\t\t Validation performance: {loss_train}")
                            logger.info(f"\t\t Accuracy score: {valid_performance}")

                        if dist.get_rank() == 0:
                            save_and_zip_model(model, os.path.join(config.OUTPUT, "pre_prune_{}.pt".format(epoch)))

                        # Thresholding
                        performance_pre_prune = loss_train
                        thresholding_step(TS)
                        valid_performance, _, loss_train = validate(config, data_loader_train, model)
                        logger.info(f"\t\t Training loss: {loss}")
                        logger.info(f"\t\t Accuracy score: {valid_performance}")

                        print_prune(config, epoch, performance_pre_prune, loss_train,
                                    best_epoch if config.PRUNE.load_best else epoch, 0, "thresholding.csv")
                        save_and_zip_model(model, os.path.join(config.OUTPUT, "after_thresholding_{}.pt".format(epoch)))

                        # Clustering
                        if config.PRUNE.CLUSTER:
                            performance_pre_prune = loss_train
                            clustering_step(CS, performance_pre_prune)
                            valid_performance, _, loss_train = validate(config, data_loader_train, model)
                            logger.info(f"\t\t Training loss: {loss}")
                            logger.info(f"\t\t Accuracy score: {valid_performance}")

                            print_prune(config, epoch, performance_pre_prune, loss_train, epoch, 0,
                                        "clustering.csv")
                            save_and_zip_model(model,
                                               os.path.join(config.OUTPUT, "after_clustering",
                                                            "{}.pt".format(epoch)))

                        newly_pruned = True

                    # We never had a good model before the plateau
                    else:
                        temperature /= config.PRUNE.decay_step
                        config.defrost()
                        config.MODEL.TWT = init_twt
                        config.freeze()
                        twt_decay_iteration = 0
                        decayed_temp = True
                        decay_steps_count += 1

                # Patience block is over, reset counters
                model_update = False
                reset_bad = True
        else:
            previous_sd = deepcopy(model.state_dict())
            model_update = True
            reset_bad = True

        tb_writer.add_scalar("Plateau Epochs", bad_epochs, epoch)

        if reset_bad:
            reset_bad = False
            bad_epochs = 0

        if loss_train < best_error or decayed_temp:
            logger.info(f"Found new best at epoch {epoch} with error {loss}")
            print_best(config, best_epoch, epoch, best_error, loss, decayed_temp)
            best_error = loss
            best_epoch = epoch
            best_sd = deepcopy(model.state_dict())
            decayed_temp = False

        tb_writer.add_scalar("Lowest Loss", best_error, epoch)
        top_acc, cr_data = get_and_save_statistics(config, epoch, model, criterion_sup, data_loader_train, loss_train,
                                                   data_loader_val, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY,
                                                   config.TRAIN.OPTIMIZER.MOMENTUM,
                                                   config.PRUNE.LMBDA, config.PRUNE.TWT,
                                                   temperature, top_acc, cr_data,
                                                   device, tb_writer, wdb_writer, newly_pruned, task, dummy_input,
                                                   init_macs)

        epoch += 1
        twt_decay_iteration += 1
        newly_pruned = False

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(loss_test, model)
        #
        # if early_stopping.early_stop:
        #     logger.info("\t\t Early stopping !! ")
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

    # Flush all the remaining 'to print' elements
    print_data(config, cr_data)

    del dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
    del model, optimizer, model_without_ddp
    torch.cuda.empty_cache()

def get_sensitivity_optimizer(config, model, device):
    print("\t\t Building sensitivity optimizer!")
    sensitivity_optimizer = None
    layers = get_layers()[0]

    if config.PRUNE.sensitivity == "neuron-lobster":
        sensitivity_optimizer = NeuronLOBSTER(model, config.PRUNE.LMBDA, layers, bn_prune=config.PRUNE.bn_prune,
                                              device=device, name_layer=config.PRUNE.layer_name)

    return sensitivity_optimizer

def train_one_epoch(config, model, criterion_sup, criterion_ssup, data_loader, optimizer, epoch,
                    mixup_fn, lr_scheduler, logger, lambda_drloc, sensitivity_optimizer, mask_params, mask_neurons):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

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
                if sensitivity_optimizer is not None:
                    sensitivity_optimizer.step(mask_params, mask_neurons, config.PRUNE.RESCALE)
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
            if sensitivity_optimizer is not None:
                sensitivity_optimizer.step(mask_params, mask_neurons, config.PRUNE.RESCALE)
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

def reduce_tensor(tensor, config):
    rt = tensor.clone()
    if config.DEBUG:
        return rt
    else:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt

@torch.no_grad()
def validate(config, data_loader, model, mode="test", verbose=True):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            #if mode == "train":
            #    if mixup_fn is not None:
            #        images, targets = mixup_fn(images, target)

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

            acc1 = reduce_tensor(acc1, config)
            acc5 = reduce_tensor(acc5, config)
            loss = reduce_tensor(loss, config)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose:
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
    if verbose:
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

def decay_parameters(config, decay, sensitivity_optimizer):
    if config.PRUNE.decay_lmbda:
        if sensitivity_optimizer is not None:
            sensitivity_optimizer.set_lambda(config.PRUNE.lmbda * decay)

def get_current_hyperparameters(config, pytorch_optimizer, sensitivity_optimizer):
    lr = [p['lr'] for p in pytorch_optimizer.param_groups]
    wd = [p['weight_decay'] for p in pytorch_optimizer.param_groups]
    mom = [p['momentum'] for p in pytorch_optimizer.param_groups if "momentum" in p]

    lmbda = sensitivity_optimizer.lmbda if sensitivity_optimizer is not None else 0
    twt = config.PRUNE.TWT

    return lr, wd, mom, lmbda, twt

def thresholding_step(TS):
    print("\t\t Thresholding")
    TS.set_twt(0)

def init_train(config, model, loss_function, pytorch_optimizer, sensitivity_optimizer, train_loader, valid_loader, test_loader,
               tb_writer, wdb_writer, dummy_input, init_macs, resume, device):
    logger.info("\t\t Initializing training procedure")
    pytorch_optimizer.zero_grad()

    task = "classification"
    cross_valid = False
    top_cr = 1
    top_acc = 0
    cr_data = {}
    newly_pruned = False

    # Get threshold scheduler
    TS = threshold_scheduler(model, sensitivity_optimizer, get_layers()[1], valid_loader, loss_function, 0, device,
                             config.AMP_OPT_LEVEL, config.PRUNE.RESCALE, config.PRUNE.bn_prune,
                             os.path.join(config.OUTPUT, "thresholding.txt"), task)

    CS = cluster_scheduler(model, sensitivity_optimizer, get_layers()[1], valid_loader, loss_function, 1e-4, device,
                           config.AMP_OPT_LEVEL, config.PRUNE.bn_prune, os.path.join(config.OUTPUT, "clustering.txt"),
                           task)

    twt_decay_function = ExponentialDecay(config.PRUNE.decay_half)
    if resume is None:
        ep = "INIT"
        epoch = 0
        temperature = 1
        bad_epochs = 0
        twt_decay_iteration = 0
    else:
        raise NotImplementedError("Resume is not implemented !")

    acc1_v, _, valid_performance = validate(config, valid_loader, model)
    print("\t\t Training loss: ", valid_performance)
    print("\t\t Accuracy score: ", acc1_v)

    lr, wd, mom, lmbda, twt = get_current_hyperparameters(config, pytorch_optimizer, sensitivity_optimizer)
    get_and_save_statistics(config, ep, model, loss_function,
                            train_loader, valid_performance, test_loader,
                            lr, wd, mom, lmbda, twt * twt_decay_function(twt_decay_iteration), temperature,
                            top_acc, cr_data, device, tb_writer, wdb_writer, newly_pruned, task, dummy_input, init_macs)

    # Get and save epoch statistics
    print_hyp(config, epoch, lr, wd, mom, lmbda, twt)

    if resume is None:
        best_error = valid_performance
        best_epoch = 0
    else:
        raise NotImplementedError("Resume is not implemented !")

    return loss_function, cross_valid, top_cr, top_acc, cr_data, newly_pruned, TS, CS, twt_decay_function, \
                best_error, best_epoch, epoch, temperature, bad_epochs, task, resume

# TODO: save_checkpoint need to be changed

if __name__ == "__main__":
    # _, config = parse_option()
    args = parse_option()
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    print("\t\t args.local_rank: ", args.local_rank)
    if args.dsets_type == "domainnet":
        # DomainNet datasets
        datasets = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    elif args.dsets_type == "decathlon":
        # Decathlon datasets
        datasets = ["aircraft", "cifar100", "daimlerpedcls", "dtd", "gtsrb", "omniglot", "svhn", "ucf101",
                    "vgg-flowers"]
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
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    for dset in datasets[3:4]:  #[:1]
        print("****************** Dataset: {} ******************".format(dset))
        # Dataset path
        args.data_path = os.path.join(datasets_path, dset)
        args.output = os.path.join(output_folder, dset)
        args.dataset_name = dset
        try:
            os.mkdir(args.output)
        except:
            print("Output folder for {} exists already !".format(args.output))

        config = get_config(args)

        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
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
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        main(config)
        print("------------------------ Done Dataset: {} -------------------".format(dset))
