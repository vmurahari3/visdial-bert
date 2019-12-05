import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLinearScheduleNonZero(_LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
        Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor =  float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

        return [base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs]