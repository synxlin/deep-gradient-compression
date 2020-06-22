from torchvision.models import resnet18

from torchpack.mtpack.utils.config import Config, configs

configs.train.batch_size = 64
configs.train.optimizer.lr = 0.025

# model
configs.model = Config(resnet18)
configs.model.num_classes = configs.dataset.num_classes
configs.model.zero_init_residual = True
