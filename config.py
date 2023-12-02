# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'aircraft'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Path to pretrained model
_C.MODEL.PRETRAINED_MODEL = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.FINETUNE = 0
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False # absolute position embedding
_C.MODEL.SWIN.RPE = True  # relative position embedding
_C.MODEL.SWIN.PATCH_NORM = True

_C.MODEL.SWIN.PARAM_RATIOS = [0, 0, 0, 4]

# CvT Transformer parameters
_C.MODEL.CVT = CN()
_C.MODEL.CVT.INIT= 'trunc_norm'
_C.MODEL.CVT.NUM_STAGES= 3
_C.MODEL.CVT.PATCH_SIZE= [7, 3, 3]
_C.MODEL.CVT.PATCH_STRIDE= [4, 2, 2]
_C.MODEL.CVT.PATCH_PADDING= [2, 1, 1]
_C.MODEL.CVT.DIM_EMBED= [64, 192, 384]
_C.MODEL.CVT.NUM_HEADS= [1, 3, 6]
_C.MODEL.CVT.DEPTH= [1, 2, 10]
_C.MODEL.CVT.MLP_RATIO= [4.0, 4.0, 4.0]
_C.MODEL.CVT.ATTN_DROP_RATE= [0.0, 0.0, 0.0]
_C.MODEL.CVT.DROP_RATE= [0.0, 0.0, 0.0]
_C.MODEL.CVT.DROP_PATH_RATE= [0.0, 0.0, 0.1]
_C.MODEL.CVT.QKV_BIAS= [True, True, True]
_C.MODEL.CVT.CLS_TOKEN= [False, False, True]
_C.MODEL.CVT.POS_EMBED= [False, False, False]
_C.MODEL.CVT.QKV_PROJ_METHOD= ['dw_bn', 'dw_bn', 'dw_bn']
_C.MODEL.CVT.KERNEL_QKV= [3, 3, 3]
_C.MODEL.CVT.PADDING_KV= [1, 1, 1]
_C.MODEL.CVT.STRIDE_KV= [2, 2, 2]
_C.MODEL.CVT.PADDING_Q= [1, 1, 1]
_C.MODEL.CVT.STRIDE_Q= [1, 1, 1]


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.RANGE = 2
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 20  # k = 20
_C.TRAIN.WEIGHT_DECAY = 0.05 # 0.05
_C.TRAIN.BASE_LR = 5e-4 # 1e-6  # 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1  # it was 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30  # 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1  # default = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw" # 'adamw'  # 'adamax'

# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Pretraining
_C.TRAIN.USE_DRLOC = False
_C.TRAIN.LAMBDA_DRLOC = 0.5
_C.TRAIN.SAMPLE_SIZE = 32
_C.TRAIN.USE_NORMAL = False
_C.TRAIN.DRLOC_MODE = 'l1'
_C.TRAIN.USE_ABS = False
_C.TRAIN.SSL_WARMUP_EPOCHS = _C.TRAIN.WARMUP_EPOCHS
_C.TRAIN.USE_MULTISCALE = False

# Adapters
_C.TRAIN.USE_ADAPTERS = True
_C.TRAIN.SIZE_ADAPTERS = 256

_C.TRAIN.TYPE_ADAPTERS = "series"

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, ²if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 10
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# Debug
_C.DEBUG = False
# -----------------------------------------------------------------------------
# Prune
# -----------------------------------------------------------------------------
_C.PRUNE = CN()
_C.PRUNE.layer_name = "parallel_mlp"
_C.PRUNE.TYPE = "magnitude"
_C.PRUNE.STRUCT = "structured"
_C.PRUNE.AMOUNT = 0.5
_C.PRUNE.WEIGHTED = 1

_C.PRUNE.DELTA_LOSS = 0.02
_C.PRUNE.load_best = False

# Hyperparameters
_C.PRUNE.RESCALE = False
_C.PRUNE.CLUSTER = False
_C.PRUNE.LMBDA = 1e-4
_C.PRUNE.TWT = 0
_C.PRUNE.PWE = 0
_C.PRUNE.MOM = 0
_C.PRUNE.NESTEROV = False
_C.PRUNE.WD = 0
_C.PRUNE.ADAM = False

_C.PRUNE.sensitivity = "neuron-lobster"
_C.PRUNE.decay_half = 50
_C.PRUNE.decay_step = 10
_C.PRUNE.decay_stop = 1e-3
_C.PRUNE.decay_lr = False
_C.PRUNE.decay_wd = False
_C.PRUNE.decay_lmbda = False
_C.PRUNE.load_best = False
_C.PRUNE.rollback = False
_C.PRUNE.mask_params = False
_C.PRUNE.mask_neurons = False
_C.PRUNE.bn_prune = False
_C.PRUNE.NO_PRUNE = False


# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT =  CN()
_C.MODEL.PROMPT.NUM_TOKENS = 100
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin:
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.PROMPT.DEEP = True # "whether do deep prompt or not, only for prepend location"


_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_C.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_C.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False

_C.MODEL.SPEC = ''

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    config.LOCAL_RANK = args.local_rank
    config.MODEL.TYPE = args.model_type
    #config.MODEL.NUM_CLASSES = args.num_classes

    # merge from specific arguments
    if args.dataset_name:
        config.DATA.DATASET = args.dataset_name
    if args.dsets_type:
        config.DATA.TYPE = args.dsets_type

    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.exp_name:
        config.EXP_NAME = args.exp_name

    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    if args.use_drloc and args.lambda_drloc > 0:
        args.use_drloc = False
        config.TRAIN.USE_DRLOC = args.use_drloc

    if args.use_drloc:
        config.TRAIN.DRLOC_MODE = args.drloc_mode

    config.MODEL.FULL_NAME = config.MODEL.TYPE + "_" + config.MODEL.NAME \
                             + "_" + str(config.MODEL.DROP_PATH_RATE)

    config.TRAIN.LAMBDA_DRLOC = args.lambda_drloc
    config.TRAIN.SAMPLE_SIZE = args.sample_size
    config.TRAIN.USE_MULTISCALE = args.use_multiscale
    config.TRAIN.USE_NORMAL = args.use_normal
    config.MODEL.SWIN.APE = args.ape
    config.MODEL.SWIN.RPE = args.rpe
    config.TRAIN.USE_ABS = args.use_abs

    config.TRAIN.EPOCHS = args.total_epochs
    config.TRAIN.SSL_WARMUP_EPOCHS = args.ssl_warmup_epochs

    config.TRAIN.SIZE_ADAPTERS = args.size_adapters
    config.TRAIN.TYPE_ADAPTERS = args.type_adapters

    config.MODEL.FINETUNE = args.finetune
    if config.MODEL.PRETRAINED_MODEL:
        config.MODEL.PRETRAINED_MODEL = args.pretrained_model

    # Set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.FULL_NAME, config.TAG)

    # Pruning parameters
    config.PRUNE.TYPE = args.prune_type
    config.PRUNE.STRUCT = args.prune_struct
    config.PRUNE.AMOUNT = args.prune_amount
    config.PRUNE.DELTA_LOSS = args.delta_loss
    config.PRUNE.WEIGHTED = args.weighted
    config.PRUNE.scaling = args.scaling
    config.TRAIN.RANGE = args.range


    # Seed
    config.SEED = args.seed
    # DEBUG
    config.DEBUG = args.debug

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
