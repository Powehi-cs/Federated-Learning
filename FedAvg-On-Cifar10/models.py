import sys
import logging

import torch
from torchvision import models


def get_model(name='vgg16'):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    elif name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    elif name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    else:
        logging.error("model name error")
        sys.exit(1)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        logging.warning("----Can't find GPU----")
        return model
