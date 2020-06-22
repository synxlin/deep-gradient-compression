from torchvision.models import resnet50

from torchpack.mtpack.utils.config import Config, configs

configs.train.optimizer.weight_decay = 1e-4
configs.train.optimizer.nesterov = True
configs.train.optimize_bn_separately = True

# model
configs.model = Config(resnet50)
configs.model.num_classes = configs.dataset.num_classes
configs.model.zero_init_residual = True
