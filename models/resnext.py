# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from models.initializers import xavier_normal_, kaiming_normal_, constant_

import paddle.nn as nn
import paddle.nn.functional as f

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    #return x * torch.tanh(F.softplus(x))
    return x * nn.Tanh()(f.softplus(x))


class BatchNorm2d(nn.BatchNorm2D):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class ResNeXtBottleneck(nn.Layer):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride,
                 cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super().__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2D(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn_reduce = nn.BatchNorm2D(D, momentum=0.001)
        self.conv_conv = nn.Conv2D(D, D,
                                   kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias_attr=False)
        self.bn = nn.BatchNorm2D(D, momentum=0.001)
        self.act = mish
        self.conv_expand = nn.Conv2D(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn_expand = nn.BatchNorm2D(out_channels, momentum=0.001)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_sublayer('shortcut_conv',
                                     nn.Conv2D(in_channels, out_channels,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias_attr=False))
            self.shortcut.add_sublayer(
                'shortcut_bn', nn.BatchNorm2D(out_channels, momentum=0.001))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = self.act(self.bn_reduce.forward(bottleneck))
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.act(self.bn.forward(bottleneck))
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return self.act(residual + bottleneck)


class CifarResNeXt(nn.Layer):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, num_classes,
                 base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super().__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 *
                       self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2D(3, 64, 3, 1, 1, bias_attr=False)
        self.bn_1 = nn.BatchNorm2D(64, momentum=0.001)
        self.act = mish
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2D):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                constant_(m.bias, 0.0)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_sublayer(name_, ResNeXtBottleneck(in_channels,
                                                          out_channels,
                                                          pool_stride,
                                                          self.cardinality,
                                                          self.base_width,
                                                          self.widen_factor))
            else:
                block.add_sublayer(name_,
                                 ResNeXtBottleneck(out_channels,
                                                   out_channels,
                                                   1,
                                                   self.cardinality,
                                                   self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = self.act(self.bn_1.forward(x))
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = f.adaptive_avg_pool2d(x, 1)
        x = x.reshape([-1, self.stages[3]])
        return self.classifier(x)


def build_resnext(cardinality, depth, width, num_classes):
    logger.info(f"Model: ResNeXt {depth+1}x{width}")
    return CifarResNeXt(cardinality=cardinality,
                        depth=depth,
                        base_width=width,
                        num_classes=num_classes)
