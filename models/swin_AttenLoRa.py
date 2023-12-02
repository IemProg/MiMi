# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
from munch import Munch

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from typing import Union, Optional
from drloc import DenseRelativeLoc


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))

def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))


def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    #return torch.stack([torch.kron(ai, bi) for ai, bi in zip(a,b)], dim=0)
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                    A.size(1)*B.size(1),
                                                    A.size(2)*B.size(2))
    return res

def matvec_product(W: torch.Tensor, x: torch.Tensor, bias: Optional[torch.Tensor], phm_rule: Union[torch.Tensor],
                   kronecker_prod=False) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product
    """
    if kronecker_prod:
        H = kronecker_product(phm_rule, W).sum(0)
    else:
        H = kronecker_product_einsum_batched(phm_rule, W).sum(0)

    y = torch.matmul(input=x, other=H)
    if bias is not None:
        y += bias
    return y

Compacter_params = {"reduction_factor": 32, "non_linearity": "gelu_new", "unfreeze_lm_head": False,
"unfreeze_layer_norms": True, "hypercomplex_adapters": True, "overwrite_output_dir": True,
"hypercomplex_division": 12, "hypercomplex_nonlinearity": "glorot-uniform", "normalize_phm_weight": False,
"learn_phm": True, "factorized_phm": True,
"shared_phm_rule": True, "factorized_phm_rule": False,
"phm_c_init": "normal", "phm_init_range": 0.0001,
"kronecker_prod" : False, "shared_W_phm" : False, "phm_rank" : 1
}

PHM_params = {
"reduction_factor": 32, "unfreeze_lm_head": False, "unfreeze_layer_norms": True, "overwrite_output_dir": True,
"hypercomplex_adapters": True, "adapter_config_name": "adapter", "hypercomplex_division": 16,
"hypercomplex_nonlinearity": "glorot-uniform", "normalize_phm_weight": False,
"learn_phm": True, "phm_c_init": "normal", "phm_init_range": 0.0001,
"shared_phm_rule" : False, "factorized_phm" : False, "shared_W_phm" : False,
"factorized_phm_rule" : False, "phm_rank" : 1,"kronecker_prod" : False
}

Compacter_params = PHM_params

class PHMLinear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 phm_dim: int,
                 phm_rule: Union[None, torch.Tensor] = None,
                 bias: bool = True,
                 w_init: str = "phm",
                 c_init: str = "random",
                 learn_phm: bool = True,
                 shared_phm_rule=False,
                 factorized_phm=False,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=False) -> None:
        super(PHMLinear, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert c_init in ["normal", "uniform"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                                                  requires_grad=learn_phm)
                self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                                                   requires_grad=learn_phm)
            else:
                self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim),
                                             requires_grad=learn_phm)
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm
        self.factorized_phm = factorized_phm
        if not self.shared_W_phm:
            if self.factorized_phm:
                self.W_left = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank)),
                                           requires_grad=True)
                self.W_right = nn.Parameter(torch.Tensor(size=(phm_dim, self.phm_rank, self._out_feats_per_axis)),
                                            requires_grad=True)
            else:
                self.W = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)),
                                      requires_grad=True)
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def init_W(self):
        if self.w_init == "glorot-normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_normal(self.W_left.data[i])
                    self.W_right.data[i] = glorot_normal(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_normal(self.W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_uniform(self.W_left.data[i])
                    self.W_right.data[i] = glorot_uniform(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_uniform(self.W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i].normal_(mean=0, std=self.phm_init_range)
                    self.W_right.data[i].normal_(mean=0, std=self.phm_init_range)
            else:
                for i in range(self.phm_dim):
                    self.W.data[i].normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
            self.init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == "uniform":
                    self.phm_rule_left.data.uniform_(-0.01, 0.01)
                    self.phm_rule_right.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule_left.data.normal_(std=0.01)
                    self.phm_rule_right.data.normal_(std=0.01)
                else:
                    raise NotImplementedError
            else:
                if self.c_init == "uniform":
                    self.phm_rule.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule.data.normal_(mean=0, std=0.01)
                else:
                    raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right
        else:
            self.phm_rule = phm_rule

    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
        if self.factorized_phm:
            W = torch.bmm(self.W_left, self.W_right)
        if self.factorized_phm_rule:
            phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)

        return matvec_product(
            W=W if self.factorized_phm else self.W,
            x=x,
            bias=self.b,
            phm_rule=phm_rule if self.factorized_phm_rule else self.phm_rule,
            kronecker_prod=self.kronecker_prod)

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        #Before it was w_init: normal
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.rpe = rpe
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Kronecker Adaptation
        hidden_dim = self.dim // 2
        in_features = self.dim
        out_features = self.dim
        # Hyper-complex Adapter layer, in which the weights of up and down sampler modules are parameters are 1/n times
        # of the conventional adapter layers, where n is hyper-complex division number.

        self.q_down_sampler_phm = PHMLinear(in_features=in_features,
                                          out_features=hidden_dim,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])
        self.q_up_sampler_phm = PHMLinear(in_features=hidden_dim,
                                          out_features=out_features,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])

        self.v_down_sampler_phm = PHMLinear(in_features=in_features,
                                          out_features=hidden_dim,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])
        self.v_up_sampler_phm = PHMLinear(in_features=hidden_dim,
                                          out_features=out_features,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])

        self.k_down_sampler_phm = PHMLinear(in_features=in_features,
                                          out_features=hidden_dim,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])
        self.k_up_sampler_phm = PHMLinear(in_features=hidden_dim,
                                          out_features=out_features,
                                          bias=True,
                                          c_init=Compacter_params["phm_c_init"],
                                          phm_dim=Compacter_params["hypercomplex_division"],
                                          learn_phm=Compacter_params["learn_phm"],
                                          w_init=Compacter_params["hypercomplex_nonlinearity"],
                                          shared_phm_rule=Compacter_params["shared_phm_rule"],
                                          factorized_phm=Compacter_params["factorized_phm"],
                                          shared_W_phm=Compacter_params["shared_W_phm"],
                                          factorized_phm_rule=Compacter_params["factorized_phm_rule"],
                                          phm_rank=Compacter_params["phm_rank"],
                                          phm_init_range=Compacter_params["phm_init_range"],
                                          kronecker_prod=Compacter_params["kronecker_prod"])

    def reset_parameters(self):
        if self.w_init == "glorot-normal":
            for i in range(self.phm_dim):
                self.phm_rule1.data[i] = glorot_normal(self.W.data[i])
                self.phm_rule2.data[i] = glorot_normal(self.W.data[i])
            for i in range(self.phm_dim):
                self.W_left1.data[i] = glorot_normal(self.W_left1.data[i])
                self.W_right1.data[i] = glorot_normal(self.W_right1.data[i])
        elif self.w_init == "glorot-uniform":
            for i in range(self.phm_dim):
                self.phm_rule1.data[i] = glorot_uniform(self.W.data[i])
            for i in range(self.phm_dim):
                self.W_left1.data[i] = glorot_uniform(self.W_left1.data[i])
                self.W_right1.data[i] = glorot_uniform(self.W_right1.data[i])
        elif self.w_init == "normal":
            for i in range(self.phm_dim):
                self.phm_rule1.data[i].normal_(mean=0, std=self.phm_init_range)
                self.phm_rule2.data[i].normal_(mean=0, std=self.phm_init_range)
            for i in range(self.phm_dim):
                self.W_left1.data[i].normal_(mean=0, std=self.phm_init_range)
                self.W_right1.data[i].normal_(mean=0, std=self.phm_init_range)
        elif self.w_init == "LoRa":
            for i in range(self.phm_dim):
                self.phm_rule1.data[i].normal_(mean=0, std=self.phm_init_range)
            for i in range(self.phm_dim):
                nn.init.zeros_(self.W_right1.data[i])
                nn.init.zeros_(self.W_left1.data[i])
        elif self.w_init == "default":
            for i in range(self.phm_dim):
                self.phm_rule1.data[i] = glorot_normal(self.W.data[i])
            for i in range(self.phm_dim):
                nn.init.eye_(self.W_right1.data[i])
                nn.init.zeros_(self.W_left1.data[i])
        else:
            raise NotImplementedError

    def phm_forward(self, x, phm_rule, w, phm_bias):
        return matvec_product(W=w, x=x,  bias=phm_bias, phm_rule=phm_rule, kronecker_prod=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        lora_input = x #torch.clone(x)

        q_delta = self.q_up_sampler_phm(self.q_down_sampler_phm(lora_input))
        v_delta = self.v_up_sampler_phm(self.v_down_sampler_phm(lora_input))
        k_delta = self.k_up_sampler_phm(self.k_down_sampler_phm(lora_input))

        q_delta = q_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_delta = v_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_delta = k_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.contiguous() + q_delta
        v = v.contiguous() + v_delta
        k = k.contiguous() + k_delta

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_with_attention(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        lora_input = x #torch.clone(x)
        q_delta = self.q_up_sampler_phm(self.q_down_sampler_phm(lora_input))
        v_delta = self.v_up_sampler_phm(self.v_down_sampler_phm(lora_input))
        k_delta = self.k_up_sampler_phm(self.k_down_sampler_phm(lora_input))

        q_delta = q_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_delta = v_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_delta = k_delta.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.contiguous() + q_delta
        v = v.contiguous() + v_delta
        k = k.contiguous() + k_delta
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).contiguous() + q_delta

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpe=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_with_attention(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn.forward_with_attention(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 rpe=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 rpe=rpe)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward_with_attention(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, att = blk.forward_with_attention(x)
                attns.append(att)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, attns


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerAttenLoRa(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False, rpe=True,
                 patch_norm=True,
                 use_checkpoint=False,
                 use_drloc=False,
                 sample_size=32,
                 use_multiscale=False,
                 drloc_mode="l1",
                 use_abs=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.use_drloc = use_drloc
        self.use_multiscale = use_multiscale

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               rpe=rpe)
            self.layers.append(layer)

        if self.use_drloc:
            self.drloc = nn.ModuleList()
            if self.use_multiscale:
                for i_layer in range(self.num_layers):
                    self.drloc.append(DenseRelativeLoc(
                        in_dim=min(int(embed_dim * 2 ** (i_layer+1)), self.num_features),
                        out_dim=2 if drloc_mode=="l1" else max(img_size // (4 * 2**i_layer), img_size//(4 * 2**(self.num_layers-1))),
                        sample_size=sample_size,
                        drloc_mode=drloc_mode,
                        use_abs=use_abs))
            else:
                self.drloc.append(DenseRelativeLoc(
                    in_dim=self.num_features,
                    out_dim=2 if drloc_mode=="l1" else img_size//(4 * 2**(self.num_layers-1)),
                    sample_size=sample_size,
                    drloc_mode=drloc_mode,
                    use_abs=use_abs))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if self.use_multiscale:
            all_layers = []
            for layer in self.layers:
                x = layer(x)
                all_layers.append(x)
            return all_layers
        else:
            for layer in self.layers:
                x = layer(x)
            return [x]

    def forward(self, x):
        xs = self.forward_features(x)
        x_last = self.norm(xs[-1])  # [B, L, C]
        # SUP
        pool = self.avgpool(x_last.transpose(1, 2))  # [B, C, 1]
        sup = self.head(torch.flatten(pool, 1))
        outs = Munch(sup=sup)

        # SSUP
        if self.use_drloc:
            outs.drloc = []
            outs.deltaxy = []
            outs.plz = []

            for idx, x_cur in enumerate(xs):
                x_cur = x_cur.transpose(1, 2) # [B, C, L]
                B, C, HW = x_cur. size()
                H = W = int(math.sqrt(HW))
                feats = x_cur.view(B, C, H, W) # [B, C, H, W]

                drloc_feats, deltaxy = self.drloc[idx](feats)
                outs.drloc.append(drloc_feats)
                outs.deltaxy.append(deltaxy)
                outs.plz.append(H) # plane size
        return outs

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def _get_last_attention(self,x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        attns = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x)
            attns.append(att)
        return x, attns