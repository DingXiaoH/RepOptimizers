# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization By Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, and Jian Dong.
https://arxiv.org/abs/2211.06088
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_layer import ScaleLayer


__all__ = [
    'repoptghostnet_0_5x',
    'repoptghostnet_repid_0_5x',
    'repoptghostnet_norep_0_5x',
    'repoptghostnet_wo_0_5x',
    'repoptghostnet_0_58x',
    'repoptghostnet_0_8x',
    'repoptghostnet_1_0x',
    'repoptghostnet_1_11x',
    'repoptghostnet_1_3x',
    'repoptghostnet_1_5x',
    'repoptghostnet_2_0x',
    'repoptghostnet',
]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class RepOptGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, mode='rep', num_stage_modules=1
    ):
        super(RepOptGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        if mode == 'target':
            print('==================== Training target model. Please make sure you are using RepOptimizer. =================')
            self.final_bn = nn.BatchNorm2d(new_channels)
            layer_following_cheap_operation = nn.Identity()
        else:
            #   fusion_conv = []    The official implementatin of RepGhost uses no "fusion_conv" (it is set as nn.Identity). So we remove it here for brevity.
            #   fusion_bn = []      The official implementatin of RepGhost set fusion_bn as nn.Sequential which has only one element. We simplify it as well.

            if mode == 'rep':
                self.fusion_bn = nn.BatchNorm2d(init_channels)
                layer_following_cheap_operation = nn.BatchNorm2d(new_channels)
                self.final_bn = nn.Identity()
            elif mode == 'hs':
                self.fusion_scale = ScaleLayer(init_channels, use_bias=False)
                scale_init = (2 / (num_stage_modules + 1)) ** 0.5
                print('module idx', num_stage_modules, 'scale init', scale_init)
                layer_following_cheap_operation = ScaleLayer(new_channels, use_bias=False, scale_init=scale_init)
                self.final_bn = nn.BatchNorm2d(new_channels)
            else:
                assert mode == 'csla'
                print('============ Training CSLA model. No need to do so. Are you doing a sanity check? Do not forget to load the constant scale values. =============== ')
                self.fusion_scale = ScaleLayer(init_channels, use_bias=False)
                layer_following_cheap_operation = ScaleLayer(new_channels, use_bias=False)
                layer_following_cheap_operation.requires_grad_(False)
                self.final_bn = nn.BatchNorm2d(new_channels)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            layer_following_cheap_operation,
        )

        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Identity()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        if hasattr(self, 'fusion_bn'):
            x2 += self.fusion_bn(x1)
        elif hasattr(self, 'fusion_scale'):
            x2 += self.fusion_scale(x1)
        x2 = self.final_bn(x2)
        return self.relu(x2)




class RepOptGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
        mode='rep',
        num_stage_blocks=1
    ):
        super(RepOptGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepOptGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            mode=mode,
            num_stage_modules=num_stage_blocks*2-1
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepOptGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            mode=mode,
            num_stage_modules=num_stage_blocks * 2
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        return x + self.shortcut(residual)


class RepOptGhostNet(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        mode='rep'
    ):
        super(RepOptGhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.num_classes = num_classes

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        num_stage_blocks = 0        #   num of blocks in a stage. use this to decide the initial value of scales in Hyper-Search (following the initialization strategy of RepOpt-VGG)
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if s == 1 and input_channel == output_channel:
                    num_stage_blocks += 1
                else:
                    num_stage_blocks = 1
                layers.append(
                    RepOptGhostBottleneck(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        mode=mode,
                        num_stage_blocks=num_stage_blocks
                    ),
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width * 2, 4)
        stages.append(
            nn.Sequential(
                ConvBnAct(input_channel, output_channel, 1),
            ),
        )
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(
            input_channel, output_channel, 1, 1, 0, bias=True,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x



def repoptghostnet(enable_se=True, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ]

    return RepOptGhostNet(cfgs, **kwargs)


def repoptghostnet_0_5x(**kwargs):
    return repoptghostnet(width=0.5, **kwargs)


def repoptghostnet_0_58x(**kwargs):
    return repoptghostnet(width=0.58, **kwargs)


def repoptghostnet_0_8x(**kwargs):
    return repoptghostnet(width=0.8, **kwargs)


def repoptghostnet_1_0x(**kwargs):
    return repoptghostnet(width=1.0, **kwargs)


def repoptghostnet_1_11x(**kwargs):
    return repoptghostnet(width=1.11, **kwargs)


def repoptghostnet_1_3x(**kwargs):
    return repoptghostnet(width=1.3, **kwargs)


def repoptghostnet_1_5x(**kwargs):
    return repoptghostnet(width=1.5, **kwargs)


def repoptghostnet_2_0x(**kwargs):
    return repoptghostnet(width=2.0, **kwargs)


