# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
from repoptimizer.repoptimizer_utils import RepOptimizerHandler
from repoptimizer.repoptghostnet_model import RepOptGhostModule, repoptghostnet_0_5x
from repoptimizer.repoptimizer_sgd import RepOptimizerSGD


def extract_blocks_into_list(model):
    blocks = []
    for block in model.modules():
        if isinstance(block, RepOptGhostModule):
            blocks.append(block)
    return blocks


def extract_scales(model):
    blocks = extract_blocks_into_list(model)
    scales = []
    for b in blocks:
        assert isinstance(b, RepOptGhostModule)
        layer_following_cheap_operation = b.cheap_operation[1]
        scales.append((b.fusion_scale.weight.detach(), layer_following_cheap_operation.weight.detach()))
        print('extract scales: ', scales[-1][0].mean(), scales[-1][1].mean())
    return scales


def identity_kernel_for_groupwise_kernel(channels, kernel_size, groups):
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    input_dim = channels // groups
    id_kernel = torch.zeros(channels, input_dim, kernel_size[0], kernel_size[1])
    for i in range(channels):
        id_kernel[i, i % input_dim, kernel_size[0]//2, kernel_size[1]//2] = 1
    return id_kernel


class RepOptGhostNetHandler(RepOptimizerHandler):

    #   scales is a list, scales[i] is a two-tuple (fusion_scale.weight, cheap_operation[1].weight)
    def __init__(self, model, scales,
                 reinit=True,
                 use_identity_scales_for_reinit=True,
                 cpu_mode=False,
                 update_rule='sgd'):
        blocks = extract_blocks_into_list(model)
        convs = [b.cheap_operation[0] for b in blocks]
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
            for scale, conv in zip(self.scales, self.convs):
                in_channels = conv.in_channels
                out_channels = conv.out_channels
                assert in_channels == out_channels
                assert len(scale) == 2
                groups = conv.groups
                kernel_size = conv.kernel_size
                if self.use_identity_scales_for_reinit:  # You may initialize the imaginary CSLA block with the trained identity_scale values. Makes almost no difference.
                    id_scale = scale[0]
                else:
                    id_scale = torch.ones(in_channels)
                conv.weight.data *= scale[1].view(-1, 1, 1, 1)
                conv.weight.data += identity_kernel_for_groupwise_kernel(in_channels, kernel_size, groups) * id_scale.view(-1, 1, 1, 1)
        else:
            raise Warning('========================== Warning! Re-init disabled. Guess you are doing an ablation study? =================')




    def generate_grad_mults(self):
        grad_mult_map = {}
        if self.update_rule == 'sgd':
            power = 2
        else:
            power = 1
        for scales, conv in zip(self.scales, self.convs):
            para = conv.weight
            assert len(scales) == 2
            #   this is just a degraded case of RepOpt-VGG
            mask = torch.ones_like(para) * (scales[1].view(-1, 1, 1, 1) ** power)
            in_channels = conv.in_channels
            out_channels = conv.out_channels
            assert in_channels == out_channels
            groups = conv.groups
            kernel_size = conv.kernel_size
            mask += identity_kernel_for_groupwise_kernel(in_channels, kernel_size, groups)
            if self.cpu_mode:
                grad_mult_map[para] = mask
            else:
                grad_mult_map[para] = mask.cuda()
        return grad_mult_map


def build_RepOptGhostNet_SGD_optimizer(model, scales, lr, momentum=0.9, weight_decay=1e-5):
    from optimizer import set_weight_decay
    handler = RepOptGhostNetHandler(model, scales, reinit=True, update_rule='sgd')
    handler.reinitialize()
    params = set_weight_decay(model)
    optimizer = RepOptimizerSGD(handler.generate_grad_mults(), params, lr=lr,
                                momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optimizer


def extract_RepOptGhostNet_0_5x_scales_from_pth(scales_path):
    trained_hs_model = repoptghostnet_0_5x(mode='hs', num_classes=100)
    weights = torch.load(scales_path, map_location='cpu')
    if 'model' in weights:
        weights = weights['model']
    if 'state_dict' in weights:
        weights = weights['state_dict']
    for ignore_key in ['linear.weight', 'linear.bias']:
        if ignore_key in weights:
            weights.pop(ignore_key)
    scales = extract_scales(trained_hs_model)
    print('check: before loading scales ', scales[-2][-1].mean(), scales[-2][-2].mean())
    trained_hs_model.load_state_dict(weights, strict=False)
    scales = extract_scales(trained_hs_model)
    print('========================================== loading scales from', scales_path)
    print('check: after loading scales ', scales[-2][-1].mean(), scales[-2][-2].mean())
    return scales

def build_RepOptGhostNet_0_5x_and_SGD_optimizer_from_pth(scales_path, lr, momentum=0.9, weight_decay=1e-5, num_classes=1000):
    model = repoptghostnet_0_5x(mode='target', num_classes=num_classes)
    scales = extract_RepOptGhostNet_0_5x_scales_from_pth(scales_path=scales_path)
    optimizer = build_RepOptGhostNet_SGD_optimizer(model, scales, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return model, optimizer


