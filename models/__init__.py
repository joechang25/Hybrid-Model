""" The model implementations are downloaded from
    1. https://github.com/keras-team/keras-applications
    2. https://github.com/keras-team/keras-contrib (resnet18,renset34)
"""

from .utils import get_submodules_from_kwargs
from .utils import correct_pad

from .xception import Xception
from .vgg16 import VGG16
from .vgg19 import VGG19
from .resnet1834 import ResNet18
from .resnet1834 import ResNet34
from .resnet import ResNet50
from .resnet import ResNet101
from .resnet import ResNet152
from .resnet_v2 import ResNet50V2
from .resnet_v2 import ResNet101V2
from .resnet_v2 import ResNet152V2
from .resnext import ResNeXt50
from .resnext import ResNeXt101
from .inception_v3 import InceptionV3
from .inception_resnet_v2 import InceptionResNetV2
from .mobilenet import MobileNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3Small
from .mobilenet_v3 import MobileNetV3Large
from .densenet import DenseNet121
from .densenet import DenseNet169
from .densenet import DenseNet201
from .nasnet import NASNetMobile
from .nasnet import NASNetLarge
from .efficientnet import EfficientNetB0
from .efficientnet import EfficientNetB1
from .efficientnet import EfficientNetB2
from .efficientnet import EfficientNetB3
from .efficientnet import EfficientNetB4
from .efficientnet import EfficientNetB5
from .efficientnet import EfficientNetB6
from .efficientnet import EfficientNetB7
from .customnet import CustomNet

model_class = {
    'xception': Xception,
    'vgg16': VGG16,
    'vgg19': VGG19,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'resnet50_v2': ResNet50V2,
    'resnet101_v2': ResNet101V2,
    'resnet152_v2': ResNet152V2,
    'resnext50': ResNeXt50,
    'resnext101': ResNeXt101,
    'inception_v3': InceptionV3,
    'inception_resnet_v2': InceptionResNetV2,
    'mobilenet': MobileNet,
    'mobilenet_v2': MobileNetV2,
    'mobilenet_v3_small': MobileNetV3Small,
    'mobilenet_v3_large': MobileNetV3Large,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201,
    'nasnet_mobile': NASNetMobile,
    'nasnet_large': NASNetLarge,
    'efficientnet_b0': EfficientNetB0,
    'efficientnet_b1': EfficientNetB1,
    'efficientnet_b2': EfficientNetB2,
    'efficientnet_b3': EfficientNetB3,
    'efficientnet_b4': EfficientNetB4,
    'efficientnet_b5': EfficientNetB5,
    'efficientnet_b6': EfficientNetB6,
    'efficientnet_b7': EfficientNetB7,
    'customnet': CustomNet,
}


