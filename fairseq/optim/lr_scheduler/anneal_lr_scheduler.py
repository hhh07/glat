from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List
from fairseq.dataclass import FairseqDataclass

@dataclass
class AnnealingSchedulerConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    anneal_steps: int = field(
        default=250000
    )
    init_lr: float = field(
        default=3e-4
    )
    end_lr: float = field(
        default=1e-5
    )
    

@register_lr_scheduler('anneal', dataclass=AnnealingSchedulerConfig)
class AnnealingScheduler(FairseqLRScheduler):
    def __init__(self, cfg:AnnealingSchedulerConfig, optimizer):
        super().__init__(cfg, optimizer)
        self.warmup_updates = cfg.warmup_updates
        self.anneal_steps = cfg.anneal_steps
        self.init_lr = cfg.init_lr
        self.end_lr = cfg.end_lr

    def step(self, epoch, val_loss=None):
        print("11111111111111111")
        return self.optimizer.get_lr()

    # @staticmethod
    # def add_args(parser):
    #     parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
    #                         help='warmup the learning rate linearly for the first N updates')
    #     parser.add_argument('--anneal-steps', default=250000, type=int)
    #     parser.add_argument('--init-lr', type=float, default=3e-4)
    #     parser.add_argument('--end-lr', type=float, default=1e-5)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.cfg.warmup_updates > 0 and num_updates <= self.cfg.warmup_updates:
            self.optimizer.set_lr(self.cfg.init_lr)
        else:
            args = self.cfg
            lr = max(0, (args.init_lr - args.end_lr) * (
                    args.anneal_steps - num_updates) / args.anneal_steps) + args.end_lr
            self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()