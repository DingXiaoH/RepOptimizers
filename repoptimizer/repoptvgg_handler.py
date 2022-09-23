# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from repoptimizer.repoptimizer_utils import RepOptimizerHandler
from repoptvgg_model import *
import numpy as np


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


class RepOptVGGHandler(RepOptimizerHandler):

    #   scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)
    def __init__(self, model, scales,
                 reinit=True, use_identity_scales_for_reinit=True,
                 cpu_mode=False,
                 update_rule='sgd'):
        blocks = extract_blocks_into_list(model)
        convs = [b.conv for b in blocks]
        assert update_rule in ['sgd', 'adamw']      # Currently supports two update functions
        self.update_rule = update_rule
        self.model = model
        self.scales = scales
        self.convs = convs
        self.reinit = reinit
        self.use_identity_scales_for_reinit = use_identity_scales_for_reinit
        self.cpu_mode = cpu_mode

    def reinitialize(self):
        if self.reinit:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    gamma_init = m.weight.mean()
                    if gamma_init == 1.0:
                        print('Checked. This is training from scratch.')
                    else:
                        raise Warning('========================== Warning! Is this really training from scratch? =================')
            print('##################### Re-initialize #############')
        else:
            raise Warning('========================== Warning! Re-init disabled. Guess you are doing an ablation study? =================')

        for scale, conv3x3 in zip(self.scales, self.convs):
            in_channels = conv3x3.in_channels
            out_channels = conv3x3.out_channels
            kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1)
            if len(scale) == 2:
                conv3x3.weight.data = conv3x3.weight * scale[1].view(-1, 1, 1, 1) \
                                      + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scale[0].view(-1, 1, 1, 1)
            else:
                assert len(scale) == 3
                assert in_channels == out_channels
                identity = torch.eye(out_channels).reshape(out_channels, out_channels, 1, 1)
                conv3x3.weight.data = conv3x3.weight * scale[2].view(-1, 1, 1, 1) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scale[1].view(-1, 1, 1, 1)
                if self.use_identity_scales_for_reinit:     # You may initialize the imaginary CSLA block with the trained identity_scale values. Makes almost no difference.
                    identity_scale_weight = scale[0]
                    conv3x3.weight.data += F.pad(identity * identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
                else:
                    conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])


    def generate_grad_mults(self):
        grad_mult_map = {}
        if self.update_rule == 'sgd':
            order = 2
        else:
            order = 1
        for scales, conv3x3 in zip(self.scales, self.convs):
            para = conv3x3.weight
            if len(scales) == 2:
                mask = torch.ones_like(para) * (scales[1] ** order).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[0] ** order).view(-1, 1, 1, 1)
            else:
                mask = torch.ones_like(para) * (scales[2] ** order).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[1] ** order).view(-1, 1, 1, 1)
                ids = np.arange(para.shape[1])
                assert para.shape[1] == para.shape[0]
                mask[ids, ids, 1:2, 1:2] += 1.0
            if self.cpu_mode:
                grad_mult_map[para] = mask
            else:
                grad_mult_map[para] = mask.cuda()
        return grad_mult_map
