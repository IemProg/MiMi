# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.datasets import SVHN, CIFAR10, CIFAR100

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
from .dataloader import *
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .dataloader import *
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset


from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)

def build_loader(config):
    config.defrost()
    if config.DATA.TYPE not in ["FVGC", "VTAB"]:
        dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
        dataset_test, _ = build_dataset(is_train=False, config=config)
    else:
        dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
        dataset_test, _ = build_dataset(is_train=False, config=config)
    config.freeze()


    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES
        )

    if not config.DEBUG:
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        indices = np.arange(dist.get_rank(), len(dataset_test), dist.get_world_size())
        sampler_test = SubsetRandomSampler(indices)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
    else:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
            num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=False)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY, drop_last=True)

    return dataset_train, dataset_test, data_loader_train, data_loader_test, mixup_fn

def build_loader_split(config, val_size=0.05, seed=42):
    config.defrost()
    dataset_train_all, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    dataset_test, _ = build_dataset(is_train=False, config=config)

    # train_length = len(dataset_train) - int(val_size * len(dataset_train))
    # dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_length, len(dataset_train) - train_length],
    #                                                            generator=torch.Generator().manual_seed(seed))
    if not config.DEBUG:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        num_tasks = 0
        global_rank = 0

    from collections import Counter
    print("Counter: ", Counter(dataset_train_all.targets))

    # Split dataset into train and validation
    if config.DATA.TYPE == "decathlon":
        train_indices, val_indices = train_test_split(list(range(len(dataset_train_all.labels))), test_size=val_size,
                                                      stratify=dataset_train_all.labels, random_state=seed)
    else:
        if config.DATA.DATASET == "painting":
            # to handle the case of having one instance for class 313
            nbr_items = len(dataset_train_all.targets) + 1
            print(dir(dataset_train_all))
            targets = dataset_train_all.targets.append(314)
            print(type(targets))
            train_indices, val_indices = train_test_split(list(range(nbr_items)),
                                                          test_size=val_size,
                                                          stratify=targets, random_state=seed)
        else:
            train_indices, val_indices = train_test_split(list(range(len(dataset_train_all.targets))),
                                                          test_size=val_size,
                                                          stratify=dataset_train_all.targets, random_state=seed)
    print(f"len train_indices {val_indices}")
    print(f"len val_indices {val_indices}")
    dataset_train = torch.utils.data.Subset(dataset_train_all, train_indices)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dataset_val = torch.utils.data.Subset(dataset_train_all, val_indices)
    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    indices = np.arange(dist.get_rank(), len(dataset_test), dist.get_world_size())
    sampler_test = SubsetRandomSampler(indices)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    decathlon_names = ["aircraft", "cifar100", "daimlerpedcls", "dtd", "gtsrb", "omniglot", "svhn", "ucf101"]

    if config.DATA.DATASET  == 'imagenet':
        if config.DATA.ZIP_MODE:
            prefix = 'train' if is_train else 'val'
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            prefix = 'train' if is_train else 'val'
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet-100':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'cifar-10':
        dataset = CIFAR10(config.DATA.DATA_PATH, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif config.DATA.DATASET == 'cifar-100':
        dataset = CIFAR100(config.DATA.DATA_PATH, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif config.DATA.DATASET == 'flowers102':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    elif config.DATA.DATASET == 'svhn':
        split = "train" if is_train else "test"
        dataset = SVHN(config.DATA.DATA_PATH, split=split, transform=transform, download=True)
        nb_classes = 10

    # Domainnet datasets
    elif config.DATA.DATASET in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345

    # VTAB datasets
    elif config.DATA.DATASET in ['caltech101', 'cifar(num_classes=100)',
    'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'sun397', 'svhn', 'resisc45','eurosat', 'dmlab',
    'kitti(task="closest_vehicle_distance")', 'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")', 'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")', 'clevr(task="count_all")', 'diabetic_retinopathy(config="btgraham-300")']:
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        split = "train" if is_train else "test"
        dataset = TFDataset(config, split)
        nb_classes = dataset.num_classes

    # FGVC datasets
    elif config.DATA.DATASET in ["CUB", "OxfordFlowers", "StanfordCars", "StanfordDogs", "nabirds"]:
        assert (
            config.DATA.DATASET in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(config.DATA.DATASET)
        dataset = _DATASET_CATALOG[config.DATA.DATASET](config.DATA, split)
        nb_classes = dataset.num_classes

    # Decathlon datasets
    elif config.DATA.DATASET in decathlon_names:
        nb_classes_dsets = [100, 100, 2, 47, 43, 1623, 10, 101, 102]
        index_dset = decathlon_names.index(config.DATA.DATASET)

        prefix = 'train' if is_train else 'test'
        path_splitted = config.DATA.DATA_PATH.split("/")[:-1]
        root = os.path.join(*path_splitted)
        root = os.path.join("/", root)

        dataset = prepare_data_loaders(root, prefix, config.DATA.DATASET, transform=transform)
        nb_classes = nb_classes_dsets[index_dset]

    else:
        raise NotImplementedError("We only support ImageNet/VTAB/FVGC/DomainNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
