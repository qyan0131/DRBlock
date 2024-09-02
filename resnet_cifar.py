# -*- coding: utf-8 -*-
"""
------ * Supplementary Material * ------ 
AAAI23 submission ID 9008: 
    DR-Block: Convolutional Dense Reparameterization 
                for CNN Free Improvement
"""

import torch.nn as nn
import torch.nn.functional as F

from dr_block import DR_Block
from repUtils import fusebn

CONV_BN_IMPL = 'base'

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7 or in_channels==3:
        blk_type = ConvBN
        print('Using Base')
    elif CONV_BN_IMPL == 'dr_block':
        blk_type = DR_Block
        print('Using DR_Block')
    else:
        raise NotImplementedError
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=False)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7 or in_channels==3:
        blk_type = ConvBN
        print('Using Base')
    elif CONV_BN_IMPL == 'dr_block':
        blk_type = DR_Block
        print('Using DR_Block')
    else:
        raise NotImplementedError
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=False, nonlinear=nn.ReLU(True))


def switch_conv_bn_impl(block_type):
    assert block_type in ['base', 'dr_block']
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type
    print("[INFO] Switch to", CONV_BN_IMPL)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels, out_channels, 3, stride, 1)
        self.conv2 = conv_bn(out_channels, out_channels, 3, 1, 1)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels)) 

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = ConvBN(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = conv_bn_relu(bottleneck_channels, bottleneck_channels, 3, stride, 1)
        self.conv3 = ConvBN(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.shortcut = nn.Sequential()  
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        y = self.conv3(y) 
        y += self.shortcut(x)
        y = F.relu(y, inplace=True) 
        return y


class Network(nn.Module):
    def __init__(self, n_blocks, block_type='basic', model_type='base', num_classes=100):
        super().__init__()

        initial_channels = 64

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        assert model_type in ['base', 'dr_block']
        switch_conv_bn_impl(model_type)

        n_channels = [
            initial_channels,
            initial_channels * block.expansion,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
            initial_channels * 8 * block.expansion,
        ]

        self.conv = nn.Conv2d(3,
                              n_channels[0],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(initial_channels)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks[0],
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks[1],
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[2],
                                       n_channels[3],
                                       n_blocks[2],
                                       block,
                                       stride=2)
        self.stage4 = self._make_stage(n_channels[3],
                                       n_channels[4],
                                       n_blocks[3],
                                       block,
                                       stride=2)

        self.fc = nn.Linear(n_channels[4], num_classes)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
