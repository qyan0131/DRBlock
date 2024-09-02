# -*- coding: utf-8 -*-
"""
------ * Supplementary Material * ------ 
AAAI23 submission ID 9008: 
    DR-Block: Convolutional Dense Reparameterization 
                for CNN Free Improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def getIdentityKernel(in_ch, k_size=1):
    center = (k_size-1)//2
    kernel_value = np.zeros((in_ch, in_ch, k_size, k_size), dtype=np.float32)
    for i in range(in_ch):
        kernel_value[i, i % in_ch, center, center] = 1
    id_tensor = torch.from_numpy(kernel_value).float()
    b = torch.zeros(in_ch)
    return id_tensor, b


def getIdentityConv(in_ch, k=1):
    r = nn.Conv2d(in_ch, in_ch, k, padding=k//2, bias=True, stride=1)
    ew, eb = getIdentityKernel(in_ch, 1)
    r.weight.data = ew
    r.bias.data = eb
    return r
    

def repBNkxk(in_ch, bn, k_size=3):
    input_dim = in_ch
    kernel_value = np.zeros((in_ch, input_dim, k_size, k_size), dtype=np.float32)
    for i in range(in_ch):
        kernel_value[i, i % input_dim, (k_size-1)//2, (k_size-1)//2] = 1
    id_tensor = torch.from_numpy(kernel_value).float()
    kernel = id_tensor
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    kernel = kernel * t
    bias = beta - running_mean * gamma / std
    
    r = nn.Conv2d(in_ch, in_ch, k_size, padding=k_size//2, bias=True, stride=1)
    r.weight.data = kernel
    r.bias.data = bias
    return r
    

def _repconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def _rep_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = _repconcat(k_slices, b_slices)
    return k, b_hat + b2

def repConv(in_ch, out_ch, c, k=1):
    r = nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=True, stride=1)
    weight = c[0].weight
    bn = c[1]
    k, b = fusebn(weight, bn)
    r.weight.data = k
    r.bias.data = b
    return r

def repDenseConcat(in_ch, out_ch, c1, c2):
    r = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=True, stride=1)
    if c2 is None:
        ew, eb = getIdentityKernel(in_ch, 1)
        ew = torch.cat([c1.weight.data, ew], 0)
        if c1.bias is None:
            c1bias = torch.zeros(c1.weight.data.shape[0])
            eb = torch.cat([c1bias, eb], 0)
        else:
            eb = torch.cat([c1.bias.data, eb], 0)
    else:
        ew = torch.cat([c1.weight.data, c2.weight.data], 0)
        if c1.bias is None:
            c1b = torch.zeros(c1.weight.data.shape[0])
        else:
            c1b = c1.bias.data
        if c2.bias is None:
            c2b = torch.zeros(c2.weight.data.shape[0])
        else:
            c2b = c2.bias.data
        eb = torch.cat([c1b, c2b], 0)
    r.weight.data = ew
    r.bias.data = eb
    return r

def rep1x1tokxk(in_ch ,out_ch, c1, c2, k=1, s=1):
    r = nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=True, stride=s)
    w1 = c1.weight
    w2 = c2.weight
    b1 = c1.bias
    if c2.bias is not None:
        b2 = c2.bias
    else:
        b2 = torch.zeros(c2.weight.data.shape[0])
    k, b = _rep_1x1_kxk(w1, b1, w2, b2, 1)
    r.weight.data = k
    r.bias.data = b
    return r
