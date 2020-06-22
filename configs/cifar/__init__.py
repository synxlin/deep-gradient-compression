import torch

from torchpack.mtpack.datasets.vision import CIFAR
from torchpack.mtpack.utils.config import Config, configs

# dataset
configs.dataset = Config(CIFAR)
configs.dataset.root = './data/cifar10'
configs.dataset.num_classes = 10
configs.dataset.image_size = 32

# training
configs.train.num_epochs = 200
configs.train.batch_size = 128

# optimizer
configs.train.optimizer.lr = 0.1
configs.train.optimizer.weight_decay = 1e-4

# scheduler
configs.train.scheduler = Config(torch.optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs - configs.train.warmup_lr_epochs
