from torch.optim.lr_scheduler import LRScheduler


class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

        # manually initialize base_lrs and last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    # 更新 after_scheduler 的 base_lrs
                    for i, group in enumerate(self.after_scheduler.optimizer.param_groups):
                        group['lr'] = self.base_lrs[i] * self.multiplier
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        # linear warmup
        progress = self.last_epoch / self.total_epoch
        return [base_lr * (1 + (self.multiplier - 1) * progress) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            super().step(epoch)
            if self.last_epoch >= self.total_epoch and self.after_scheduler:
                self.after_scheduler.step(0)
                self._last_lr = self.after_scheduler.get_last_lr()
