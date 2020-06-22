import torch

from torchpack.mtpack.utils.config import Config, configs
from torchpack.mtpack.meters import TopKClassMeter

from dgc.horovod.compression import Compression


configs.seed = 42
configs.data = Config()
configs.data.num_threads_per_worker = 4

# criterion
configs.train = Config()
configs.train.dgc = False
configs.train.compression = Compression.none
configs.train.criterion = Config(torch.nn.CrossEntropyLoss)

# optimizer
configs.train.optimizer = Config(torch.optim.SGD)
configs.train.optimizer.momentum = 0.9

# scheduler
configs.train.schedule_lr_per_epoch = True
configs.train.warmup_lr_epochs = 5

# metrics
configs.train.metric = 'acc/test_top1'
configs.train.meters = Config()
configs.train.meters['acc/{}_top1'] = Config(TopKClassMeter)
configs.train.meters['acc/{}_top1'].k = 1
configs.train.meters['acc/{}_top5'] = Config(TopKClassMeter)
configs.train.meters['acc/{}_top5'].k = 5
