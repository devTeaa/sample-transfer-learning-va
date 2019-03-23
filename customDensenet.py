import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.utils as vutils

import pdb

__all__ = ['DenseNet', 'densenet121',
           'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        origin_model = model_zoo.load_url(model_urls['densenet121'])

        for key in list(origin_model.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                origin_model[new_key[9:]] = origin_model[key]
                del origin_model[key]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        origin_model = {k: v for k, v in origin_model.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(origin_model)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

def interpolate(x, multiplier=2, divider=2, absolute_channel = 0, mode='nearest'):
    return F.interpolate(x.view(1, x.size()[0], x.size()[1], x.size()[2], x.size()[3]),
                            size=(x.size()[1] // divider if absolute_channel == 0 else absolute_channel, x.size()[2] * multiplier, x.size()[3] * multiplier),
                            mode=mode)[0]


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
 
        self.conv2d1x1fp4 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp3 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2d1x1fp1 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
      
        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)
        
        # # Each denseblock
        # num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
            # if (num_layers != 0):
                # block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    # bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                # self.features.add_module('denseblock%d' % (i + 1), block)
                # num_features = num_features + num_layers * growth_rate
                # if i != len(block_config) - 1:
                    # trans = _Transition(
                        # num_input_features=num_features, num_output_features=num_features // 2)
                    # self.features.add_module('transition%d' % (i + 1), trans)
                    # num_features = num_features // 2

        # # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # # 7
        # self.transconv1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        # # 14
        # self.transconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        # # 28
        # self.transconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        # # 56
        # self.transconv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        # # 112
        # self.transconv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        # # 224

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # # Visual attention layer (output 1 x 7 x 7)
        # self.valinear = nn.Linear(1024 * 7 * 7, 49)
        # self.valinear = nn.Conv2d(1024, 1, 3, 1, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):
        # Block 1
        f1 = self.denseblock1(self.features(x))

        # Block 2
        f2 = self.denseblock2(self.transition1(f1))

        # Block 3
        f3 = self.denseblock3(self.transition2(f2))

        # Block 4
        f4 = self.denseblock4(self.transition3(f3))

        # Feature Pyramid
        fp3 = interpolate(f4, divider=1) + self.conv2d1x1fp3(f3)
        fp2 = interpolate(f3) + self.conv2d1x1fp2(f2)
        fp1 = interpolate(f2) + self.conv2d1x1fp1(f1)

        x = x + interpolate(fp1, multiplier = 4, absolute_channel = 3)

        # =============================================================
        # Phase 2 normal Densenet Sequence
        x = self.features(x)
        x = self.denseblock1(x)
        x = self.denseblock2(self.transition1(x))
        x = self.denseblock3(self.transition2(x))
        x = self.denseblock4(self.transition3(x))
        x = self.batchNorm5(x)

        # if epoch != -1:
        #     writer.add_image('Image', vutils.make_grid(x, normalize=True, scale_each=True), epoch)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x
