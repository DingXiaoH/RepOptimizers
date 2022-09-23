# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
from se_block import SEBlock
from scale_layer import ScaleLayer



def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result

class RealVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_post_se=False):
        super(RealVGGBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

    def forward(self, inputs):
        out = self.post_se(self.relu(self.bn(self.conv(inputs))))
        return out


#   As baseline
class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 stride=1, use_post_se=False, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.relu = nn.ReLU()
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
            self.bn = nn.Identity()
        else:
            self.conv_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(out_channels)
            self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
            if in_channels == out_channels and stride == 1:
                self.bn_identity = nn.BatchNorm2d(out_channels)
        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

    def forward(self, inputs):
        out = self.bn_3x3(self.conv_3x3(inputs)) + self.bn_1x1(self.conv_1x1(inputs))
        if hasattr(self, 'bn_identity'):
            out += self.bn_identity(inputs)
        out = self.post_se(self.relu(out))
        return out


#   A CSLA block is a LinearAddBlock with is_csla=True
class LinearAddBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_post_se=False, is_csla=False, conv_scale_init=None):
        super(LinearAddBlock, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.scale_conv = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.scale_1x1 = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=conv_scale_init)
        if in_channels == out_channels and stride == 1:
            self.scale_identity = ScaleLayer(num_features=out_channels, use_bias=False, scale_init=1.0)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_csla:     # Make them constant
            self.scale_1x1.requires_grad_(False)
            self.scale_conv.requires_grad_(False)
        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

    def forward(self, inputs):
        out = self.scale_conv(self.conv(inputs)) + self.scale_1x1(self.conv_1x1(inputs))
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs)
        out = self.post_se(self.relu(self.bn(out)))
        return out



def get_block(mode, in_channels, out_channels, stride, use_post_se, conv_scale_init=1.0):
    if mode == 'target':
        return RealVGGBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, use_post_se=use_post_se)
    elif mode == 'repvgg':
        return RepVGGBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, use_post_se=use_post_se)
    else:
        return LinearAddBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, use_post_se=use_post_se,
                              is_csla=mode == 'csla', conv_scale_init=conv_scale_init)

class RepOptVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None,
                 strides=(2,2,2,2,2), use_post_se=False, mode='target'):
        super(RepOptVGG, self).__init__()
        assert len(width_multiplier) == 4
        assert mode in ['target', 'csla', 'hs', 'repvgg']
        self.mode = mode
        self.num_classes = num_classes
        self.in_channels = min(64, int(64 * width_multiplier[0]))
        self.stage0 = get_block(self.mode, in_channels=3, out_channels=self.in_channels, stride=strides[0], use_post_se=use_post_se)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=strides[1], use_post_se=use_post_se)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=strides[2], use_post_se=use_post_se)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=strides[3], use_post_se=use_post_se)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=strides[4], use_post_se=use_post_se)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, channels, num_blocks, stride, use_post_se, block_idx_base=0):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for i, stride in enumerate(strides):
            if i + block_idx_base == 0:
                conv_scale_init = 1.0
            else:
                conv_scale_init = (2.0 / (i + block_idx_base)) ** 0.5
            block = get_block(self.mode, in_channels=self.in_channels, out_channels=channels, stride=stride, use_post_se=use_post_se, conv_scale_init=conv_scale_init)
            blocks.append(block)
            self.in_channels = channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out