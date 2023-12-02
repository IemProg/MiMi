from torchvision import models

from munch import Munch
from drloc import DenseRelativeLoc


import torch
import torch.nn as nn
import torch.nn.functional as F

import math, re
import numpy as np

class config_task():
    # File allowing to change which task is currently used for training/testing
    task = 0
    mode = 'normal'
    proj = '11'
    factor = 1.
    dropouts_args = '00'

    wd3x3, wd1x1, wd = [1.], [1.], 1
    decay3x3 = np.array(wd3x3) * 0.0001
    decay1x1 = np.array(wd1x1) * 0.0001
    wd = wd * 0.0001

    isdropout1 = (dropouts_args[0] == '1')
    isdropout2 = (dropouts_args[1] == '1')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):

    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        if config_task.mode == 'series_adapters':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        elif config_task.mode == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride)
        else:
            self.conv = conv1x1_fonc(planes)
    def forward(self, x):
        y = self.conv(x)
        if config_task.mode == 'series_adapters':
            y += x
        return y

class conv_task(nn.Module):

    def __init__(self, in_planes, planes, stride=1, nb_tasks=1, is_proj=1, second=0):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        if config_task.mode == 'series_adapters' and is_proj:
            self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes), nn.BatchNorm2d(planes)) for i in range(nb_tasks)])
        elif config_task.mode == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(nb_tasks)])

            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])

    def forward(self, x):
        task = config_task.task
        y = self.conv(x)
        if self.second == 0:
            if config_task.isdropout1:
                x = F.dropout2d(x, p=0.5, training = self.training)
        else:
            if config_task.isdropout2:
                x = F.dropout2d(x, p=0.5, training = self.training)
        if config_task.mode == 'parallel_adapters' and self.is_proj:
            y = y + self.parallel_conv[task](x)
        y = self.bns[task](y)
        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0, nb_tasks=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=int(config_task.proj[0]))
        self.conv2 = nn.Sequential(nn.ReLU(True), conv_task(planes, planes, 1, nb_tasks, is_proj=int(config_task.proj[1]), second=1))
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut == 1:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual*0),1)
        y += residual
        y = F.relu(y)
        return y


class CustumResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=[10]):
        super(CustumResNet, self).__init__()
        num_classes = [num_classes]
        nb_tasks = len(num_classes)
        blocks = [block, block, block]
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.pre_layers_conv = conv_task(3,int(32*factor), 1, nb_tasks)
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2, nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2, nb_tasks=nb_tasks)
        self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256*factor)),nn.ReLU(True)) for i in range(nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linears = nn.ModuleList([nn.Linear(int(256*factor), num_classes[i]) for i in range(nb_tasks)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, nb_tasks=nb_tasks))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers_conv(x)
        task = config_task.task
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end_bns[task](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linears[task](x)
        return x

def resnet26(num_classes=10, blocks=BasicBlock):
    return  CustumResNet(blocks, [4,4,4], num_classes)

class ResidualResNet(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_path,
        pretrained_bool=1,
        use_drloc=False,     # relative distance prediction
        drloc_mode="l1",
        sample_size=32,
        use_abs=False,
        residual_mode="parallel_adapters"
    ):
        super().__init__()
        self.use_drloc = use_drloc
        # don't use the pretrained model

        if pretrained_bool == 1:
            # Finetunning with Parellel Adapters

            # Load checkpoint and initialize the networks with the weights of a pretrained network
            print('\t\t Loading Resnet26 pretrained network with Residuals')
            # checkpoint = torch.load(pretrained_path, encoding='latin1')
            # checkpoint = torch.load(pretrained_path)

            # net_old = checkpoint['net']
            net_old = resnet26(num_classes)
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            msg = net_old.load_state_dict(checkpoint, strict=False)
            print("\t\t Message: ", msg)
            # net_old = torch.load(pretrained_path)
            net = resnet26(num_classes)

            store_data = []
            for name, m in net_old.named_modules():
                if isinstance(m, nn.Conv2d) and (m.kernel_size[0] == 3):
                    store_data.append(m.weight.data)

            element = 0
            for name, m in net.named_modules():
                if isinstance(m, nn.Conv2d) and (m.kernel_size[0] == 3):
                    m.weight.data = store_data[element]
                    element += 1

            store_data = []
            store_data_bias = []
            store_data_rm = []
            store_data_rv = []
            names = []

            for name, m in net_old.named_modules():
                if isinstance(m, nn.BatchNorm2d) and 'bns.' in name:
                    names.append(name)
                    store_data.append(m.weight.data)
                    store_data_bias.append(m.bias.data)
                    store_data_rm.append(m.running_mean)
                    store_data_rv.append(m.running_var)

            # Special case to copy the weight for the BN layers when the target and source networks have not the same number of BNs
            condition_bn = 'noproblem'
            if len(names) != 51 and residual_mode == 'series_adapters':
                condition_bn = 'bns.....conv'

            tasks = [num_classes]
            for id_task in range(len(tasks)):
                element = 0
                for name, m in net.named_modules():
                    if isinstance(m, nn.BatchNorm2d) and 'bns.' + str(id_task) in name and not re.search(condition_bn,
                                                                                                         name):
                        m.weight.data = store_data[element].clone()
                        m.bias.data = store_data_bias[element].clone()
                        m.running_var = store_data_rv[element].clone()
                        m.running_mean = store_data_rm[element].clone()
                        element += 1

            del net_old

            # Freeze 3*3 convolution layers
            for name, m in net.named_modules():
                if isinstance(m, nn.Conv2d) and (m.kernel_size[0] == 3):
                    m.weight.requires_grad = False

            self.model  = net
            #self.num_ftrs = net.linears[0].in_features
            #net.linears = nn.Linear(self.num_ftrs, num_classes)

            #layers = [v for v in net.children()]
            #self.model = nn.Sequential(*layers[:-2])
            #self.avgpool = layers[-2]
            #self.linears = layers[-1]

        else:
            self.model = resnet26(num_classes)

        if self.use_drloc:
            self.drloc = nn.ModuleList()
            self.drloc.append(DenseRelativeLoc(
                in_dim=self.num_ftrs,
                out_dim=2 if drloc_mode=="l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs))

    def forward(self, x):
        #TODO: Complete DR_Loc code
        outs = Munch()
        # SSUP
        B, C, H, W = x.size()
        outs.drloc = []
        outs.deltaxy = []
        outs.plz = []

        #x = self.fc(torch.flatten(self.pool(x), 1))
        x = self.model(x)
        outs.sup = x
        return outs