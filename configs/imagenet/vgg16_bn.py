from torchvision.models import vgg16_bn

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(vgg16_bn)
configs.model.num_classes = configs.dataset.num_classes
