from torchpack.mtpack.utils.config import Config, configs

from dgc.compression import DGCCompressor
from dgc.memory import DGCSGDMemory
from dgc.optim import DGCSGD


configs.train.dgc = True
configs.train.compression = Config(DGCCompressor)
configs.train.compression.compress_ratio = 0.001
configs.train.compression.sample_ratio = 0.01
configs.train.compression.strided_sample = True
configs.train.compression.compress_upper_bound = 1.3
configs.train.compression.compress_lower_bound = 0.8
configs.train.compression.max_adaptation_iters = 10
configs.train.compression.resample = True

old_optimizer = configs.train.optimizer
configs.train.optimizer = Config(DGCSGD)
for k, v in old_optimizer.items():
    configs.train.optimizer[k] = v

configs.train.compression.memory = Config(DGCSGDMemory)
configs.train.compression.memory.momentum = configs.train.optimizer.momentum
