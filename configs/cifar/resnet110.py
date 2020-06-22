from torchpack.mtpack.models.vision.resnet import resnet110

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(resnet110)
configs.model.num_classes = configs.dataset.num_classes
