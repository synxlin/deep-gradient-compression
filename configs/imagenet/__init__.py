import torch

from torchpack.mtpack.datasets.vision import ImageNet
from torchpack.mtpack.utils.config import Config, configs

# dataset
configs.dataset = Config(ImageNet)
configs.dataset.root = '/dataset/imagenet'
configs.dataset.num_classes = 1000
configs.dataset.image_size = 224

# training
configs.train.num_epochs = 90
configs.train.batch_size = 32

# optimizer
configs.train.optimize_bn_separately = False
configs.train.optimizer.lr = 0.0125
configs.train.optimizer.weight_decay = 5e-5

# scheduler
configs.train.scheduler = Config(torch.optim.lr_scheduler.MultiStepLR)
configs.train.scheduler.milestones = [e - configs.train.warmup_lr_epochs 
                                      for e in [30, 60, 80]]
configs.train.scheduler.gamma = 0.1
