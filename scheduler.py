import math

from torch.optim.lr_scheduler import LRScheduler

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.d_model = d_model
        self._last_lr = []

    def step(self):
        self.step_num += 1
        calc_lr = (self.d_model ** -0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))

        values = [calc_lr]

        for data in zip(self.optimizer.param_groups, values):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr

