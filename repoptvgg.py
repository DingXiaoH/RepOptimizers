# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from torch.optim.sgd import SGD
import torch.nn.functional as F
from se_block import SEBlock
from scale_layer import ScaleLayer
from optimizer import set_weight_decay

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


def extract_blocks_into_list(model):
    stages = [model.stage0, model.stage1, model.stage2, model.stage3, model.stage4]
    blocks = []
    for stage in stages:
        if isinstance(stage, RealVGGBlock) or isinstance(stage, LinearAddBlock):
            blocks.append(stage)
        else:
            assert isinstance(stage, nn.Sequential)
            for block in stage.children():
                assert isinstance(block, RealVGGBlock) or isinstance(block, LinearAddBlock)
                blocks.append(block)
    return blocks

def extract_scales(model):
    blocks = extract_blocks_into_list(model)
    scales = []
    for b in blocks:
        assert isinstance(b, LinearAddBlock)
        if hasattr(b, 'scale_identity'):
            scales.append((b.scale_identity.weight.detach(), b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        else:
            scales.append((b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        print('extract scales: ', scales[-1][-2].mean(), scales[-1][-1].mean())
    return scales


class RepVGGOptimizer(SGD):

    #   scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)
    def __init__(self, model, scales, num_blocks, width_multiplier,
                 lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 reinit=True, use_identity_scales_for_reinit=True,
                 cpu_mode=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        parameters = set_weight_decay(model)
        super(SGD, self).__init__(parameters, defaults)
        self.num_blocks = num_blocks
        self.width_multiplier = width_multiplier
        self.num_layers = len(scales)

        blocks = extract_blocks_into_list(model)
        convs = [b.conv for b in blocks]

        if reinit:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    gamma_init = m.weight.mean()
                    if gamma_init == 1.0:
                        print('Checked. This is training from scratch.')
                    else:
                        raise Warning('========================== Warning! Is this really training from scratch ? =================')
            print('##################### Re-initialize #############')
            self.reinitialize(scales, convs, use_identity_scales_for_reinit)

        self.generate_gradient_masks(scales, convs, cpu_mode)

    def reinitialize(self, scales_by_idx, conv3x3_by_idx, use_identity_scales):
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            in_channels = conv3x3.in_channels
            out_channels = conv3x3.out_channels
            kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1)
            if len(scales) == 2:
                conv3x3.weight.data = conv3x3.weight * scales[1].view(-1, 1, 1, 1) \
                                      + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[0].view(-1, 1, 1, 1)
            else:
                assert len(scales) == 3
                assert in_channels == out_channels
                identity = torch.from_numpy(np.eye(out_channels, dtype=np.float32).reshape(out_channels, out_channels, 1, 1))
                conv3x3.weight.data = conv3x3.weight * scales[2].view(-1, 1, 1, 1) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[1].view(-1, 1, 1, 1)
                if use_identity_scales:     # You may initialize the imaginary CSLA block with the trained identity_scale values. Makes almost no difference.
                    identity_scale_weight = scales[0]
                    conv3x3.weight.data += F.pad(identity * identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
                else:
                    conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])


    def generate_gradient_masks(self, scales_by_idx, conv3x3_by_idx, cpu_mode=False):
        self.grad_mask_map = {}
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            para = conv3x3.weight
            if len(scales) == 2:
                mask = torch.ones_like(para) * (scales[1] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[0] ** 2).view(-1, 1, 1, 1)
            else:
                mask = torch.ones_like(para) * (scales[2] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[1] ** 2).view(-1, 1, 1, 1)
                ids = np.arange(para.shape[1])
                assert para.shape[1] == para.shape[0]
                mask[ids, ids, 1:2, 1:2] += 1.0
            if cpu_mode:
                self.grad_mask_map[para] = mask
            else:
                self.grad_mask_map[para] = mask.cuda()

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p in self.grad_mask_map:
                    d_p = p.grad.data * self.grad_mask_map[p]  # Note: multiply the mask here
                else:
                    d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss