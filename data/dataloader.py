import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

import pickle
from PIL import Image
from pycocotools.coco import COCO
import os, sys
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, index=None,
                 labels=None, imgs=None, loader=pil_loader, skip_label_indexing=0):

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        if index is not None:
            imgs = [imgs[i] for i in index]
        self.imgs = imgs
        if index is not None:
            if skip_label_indexing == 0:
                labels = [labels[i] for i in index]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index][0]
        target = self.labels[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def prepare_data_loaders(data_dir, prefix, dataset_name, transform, index=None):
    train = [0]
    val = [1]

    # Paths to datasets
    imdb_dir = os.path.join(data_dir, "decathlon-1.0-annots", "annotations")
    data_dir = os.path.join(data_dir, dataset_name)

    imdb_names_train = os.path.join(imdb_dir, dataset_name + '_train.json')
    imdb_names_val = os.path.join(imdb_dir, dataset_name + '_val.json')
    imdb_names = [imdb_names_train, imdb_names_val]

    imgnames_train = []
    imgnames_val = []
    labels_train = []
    labels_val = []

    for itera1 in train + val:
        annFile = imdb_names[itera1]
        coco = COCO(annFile)
        imgIds = coco.getImgIds()
        annIds = coco.getAnnIds(imgIds=imgIds)
        anno = coco.loadAnns(annIds)
        images = coco.loadImgs(imgIds)
        timgnames = [img['file_name'] for img in images]
        timgnames_id = [img['id'] for img in images]
        labels = [int(ann['category_id']) - 1 for ann in anno]
        min_lab = min(labels)
        labels = [lab - min_lab for lab in labels]
        max_lab = max(labels)

        imgnames = []
        for j in range(len(timgnames)):
            tpath = timgnames[j].split("/")[2:]
            tpath = os.path.join("/", *tpath)
            imgnames.append((data_dir + tpath, timgnames_id[j]))

        if itera1 in train:
            imgnames_train += imgnames
            labels_train += labels

        if itera1 in val:
            imgnames_val += imgnames
            labels_val += labels

    if prefix == "train":
        path = os.path.join(data_dir, "train")
        train_loader = ImageFolder(path, transform, None, index, labels_train, imgnames_train)
        return train_loader
    else:
        path = os.path.join(data_dir, "val")
        val_loader = ImageFolder(path, transform, None, None, labels_val, imgnames_val)
        return val_loader

