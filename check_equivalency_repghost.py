from repoptimizer.repoptimizer_sgd import RepOptimizerSGD
from repoptimizer.repoptimizer_adamw import RepOptimizerAdamW
from repoptimizer.repoptghostnet_model import RepOptGhostModule
from repoptimizer.repoptghostnet_impl import RepOptGhostNetHandler, identity_kernel_for_groupwise_kernel

import torch.nn as nn
import torch


num_train_iters = 5
lr = 0.1
momentum = 0.9
weight_decay = 0.1
nest = True

channels = 4

train_data = []
for _ in range(num_train_iters):
    train_data.append(torch.randn(10, channels, 14, 14))


def get_model(mode):
    return nn.Sequential(
        RepOptGhostModule(channels, channels, mode=mode),
        RepOptGhostModule(channels, channels, mode=mode)
    )

test_scales = [
    (torch.rand(channels), torch.rand(channels)),
    (torch.rand(channels), torch.rand(channels))
]

def get_equivalent_kernel(csla_m):
    return csla_m.cheap_operation[0].weight.data * csla_m.cheap_operation[1].weight.data.view(-1, 1, 1, 1) \
           + identity_kernel_for_groupwise_kernel(csla_m.cheap_operation[0].in_channels, csla_m.cheap_operation[0].kernel_size, csla_m.cheap_operation[0].groups) \
           * csla_m.fusion_scale.weight.data.view(-1, 1, 1, 1)



def check_equivalency(update_rule):

    assert update_rule in ['sgd', 'adamw']
    print('################################# testing optimizer: ', update_rule)

    csla_model = get_model('csla')
    target_model = get_model('target')

    #   load scales
    for m, s in zip(csla_model, test_scales):
        m.fusion_scale.weight.data = s[0]
        m.cheap_operation[1].weight.data = s[1]

    # remove irrelevant components and make identical initialization
    for csla_m, target_m in zip(csla_model, target_model):
        csla_m.primary_conv = nn.Identity()
        target_m.primary_conv = nn.Identity()
        csla_m.final_bn = nn.Identity()
        target_m.final_bn = nn.Identity()
        target_m.cheap_operation[0].weight.data = get_equivalent_kernel(csla_m).detach().clone()

    handler = RepOptGhostNetHandler(model=target_model, scales=test_scales, reinit=False, cpu_mode=True, update_rule=update_rule)

    if update_rule == 'sgd':
        csla_optimizer = torch.optim.SGD(params=csla_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        target_optimizer = RepOptimizerSGD(handler.generate_grad_mults(), target_model.parameters(), lr=lr, momentum=momentum,
                                       weight_decay=weight_decay)
    else:
        csla_optimizer = torch.optim.AdamW(params=csla_model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=lr,
                                      weight_decay=weight_decay)
        target_optimizer = RepOptimizerAdamW(handler.generate_grad_mults(), target_model.parameters(), eps=1e-8,
                                         betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)

    csla_model.train()
    target_model.train()

    def train(model, optimizer):
        for i in range(num_train_iters):
            x = train_data[i]
            y = model(x)
            optimizer.zero_grad()
            loss = y.var()  # just an arbitrary loss function.
            loss.backward()
            optimizer.step()

    train(csla_model, csla_optimizer)
    print('============== finished training the original model')
    train(target_model, target_optimizer)
    print('============== finished training the equivalent model')

    target_sample_kernel = target_model[-1].cheap_operation[0].weight
    csla_sample_kernel = get_equivalent_kernel(csla_model[-1])

    print('============== the relative difference is ')
    print((target_sample_kernel - csla_sample_kernel).abs().sum() / target_sample_kernel.abs().sum())


check_equivalency('sgd')
check_equivalency('adamw')