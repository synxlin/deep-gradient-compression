from torchpack.mtpack.models.vision.resnet import resnet20

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(resnet20)
configs.model.num_classes = configs.dataset.num_classes
