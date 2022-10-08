# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
from torch.optim.sgd import SGD


class RepOptimizerSGD(SGD):

    def __init__(self,
                 grad_mult_map,
                 params,
                 lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(RepOptimizerSGD, self).__init__(params, lr, momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.grad_mult_map = grad_mult_map
        print('============ Grad Mults generated. There are ', len(self.grad_mult_map))

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

                if p in self.grad_mult_map:
                    d_p = p.grad.data * self.grad_mult_map[p]   # Note: multiply here
                else:
                    d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p + buf * momentum #  d_p.add(buf, momentum)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss
