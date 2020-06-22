import torch

from torchpack.mtpack.utils.config import Config, configs

# scheduler
configs.train.scheduler = Config(torch.optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs - configs.train.warmup_lr_epochs
