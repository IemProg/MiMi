import torch
import torch.nn as nn
import torchvision
from torchvision import models

from munch import Munch
from drloc import DenseRelativeLoc

from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class ParallelMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 type="series", mlp_bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or 1
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)
        self.drop = nn.Dropout(drop)
        self.type = type
    def forward(self, x):
        shortcut = torch.clone(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if self.type == "parallel":
            return x
        else:
            return x + shortcut

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckWithAdapters(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,  param_ratio=32, drop=0.):
        super(BottleneckWithAdapters, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.m = nn.Flatten()
        mlp_hidden_dim = 512
        if param_ratio != 0:
            # Parallel Adapters
            if "ffn" in self.type_adapter.lower():
                if "parallel" in self.type_adapter.lower():
                    self.norm4 = norm_layer(dim)
                    self.parallel_mlp2 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="parallel", mlp_bias=False)
                else:
                    self.parallel_mlp2 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, mlp_bias=False)
            elif "attn" in self.type_adapter.lower():
                if "parallel" in self.type_adapter.lower():
                    self.norm3 = norm_layer(dim)
                    self.parallel_mlp1 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="parallel", mlp_bias=False)
                else:
                    self.parallel_mlp1 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, mlp_bias=False)
            elif "mh" in self.type_adapter.lower():
                self.parallel_mlp1 = ParallelMlp(in_features=self.dim,
                                                 hidden_features=mlp_hidden_dim // param_ratio,
                                                 out_features=self.dim * self.num_heads,
                                                 act_layer=act_layer, drop=drop, mlp_bias=False)
            else:
                if self.type_adapter.lower() == "parallel":
                    self.norm3 = norm_layer(dim)
                    self.norm4 = norm_layer(dim)
                    self.parallel_mlp1 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="parallel", mlp_bias=False)
                    self.parallel_mlp2 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="parallel", mlp_bias=False)
                else:
                    self.parallel_mlp1 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="series", mlp_bias=False)
                    self.parallel_mlp2 = ParallelMlp(in_features=self.dim,
                                                     hidden_features=mlp_hidden_dim // param_ratio,
                                                     act_layer=act_layer, drop=drop, type="series", mlp_bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        shape = out.shape
        out = self.m(out)
        out = self.parallel_mlp2(out) #need to flattening
        out = out.reshape(shape)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        shape = out.shape
        out = self.m(out)
        out = self.parallel_mlp2(out)  # need to flattening
        out = out.reshape(shape)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        shape = out.shape
        out = self.m(out)
        out = self.parallel_mlp2(out)  # need to flattening
        out = out.reshape(shape)

        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_bool,
        use_drloc=False,     # relative distance prediction
        drloc_mode="l1",
        sample_size=32,
        use_abs=False
    ):
        super().__init__()
        self.use_drloc = use_drloc
        # don't use the pretrained model

        if pretrained_bool == 1:
            model = resnet50(pretrained=True)
        else:
            model = resnet50(pretrained=False)
        self.num_ftrs = model.fc.in_features
        model.fc = nn.Linear(self.num_ftrs, num_classes)

        layers = [v for v in model.children()]
        self.model = nn.Sequential(*layers[:-2])
        self.pool = layers[-2]
        self.fc   = layers[-1]

        if self.use_drloc:
            self.drloc = nn.ModuleList()
            self.drloc.append(DenseRelativeLoc(
                in_dim=self.num_ftrs,
                out_dim=2 if drloc_mode=="l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs))

    def forward(self,x):
        x = self.model(x) # [B, C, H, W]
        outs = Munch()

        # SSUP
        B, C, H, W = x.size()
        if self.use_drloc:
            outs.drloc = []
            outs.deltaxy = []
            outs.plz = []

            for idx, x_cur in enumerate([x]):
                drloc_feats, deltaxy = self.drloc[idx](x_cur)
                outs.drloc.append(drloc_feats)
                outs.deltaxy.append(deltaxy)
                outs.plz.append(H) # plane size

        x = self.fc(torch.flatten(self.pool(x), 1))
        outs.sup = x
        return outs

