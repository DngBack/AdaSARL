# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d
from typing import List, Tuple
import math


class Gelu(nn.Module):
    """GELU activation function implementation"""
    def __init__(self):
        super(Gelu, self).__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GeluBasicBlock(nn.Module):
    """
    The basic block of ResNet with GELU activation.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, apply_gelu=True) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        :param apply_gelu: whether to apply GELU activation
        """
        super(GeluBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.apply_gelu = apply_gelu

        self.gelu1 = Gelu()
        self.gelu2 = Gelu()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """

        out = self.bn1(self.conv1(x))
        if self.apply_gelu:
            out = self.gelu1(out)
        else:
            out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.apply_gelu:
            out = self.gelu2(out)
        else:
            out = F.relu(out)
        return out


class GeluResNet(nn.Module):
    """
    ResNet network architecture with GELU activations. Designed for complex datasets.
    """

    def __init__(self, block: GeluBasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, apply_gelu=(1, 1, 1, 1)) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        :param apply_gelu: tuple indicating which layers to apply GELU to
        """
        super(GeluResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.num_blocks = num_blocks

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, apply_gelu=apply_gelu[0])
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, apply_gelu=apply_gelu[1])
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, apply_gelu=apply_gelu[2])
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, apply_gelu=apply_gelu[3])
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

        self.classifier = self.linear

    def _make_layer(self, block: GeluBasicBlock, planes: int,
                    num_blocks: int, stride: int, apply_gelu: bool) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :param apply_gelu: whether to apply GELU activation
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, apply_gelu))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_activations=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        activations = {}
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        activations['feat'] = out
        out = self.linear(out)
        if return_activations:
            return out, activations
        else:
            return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        feat = self._features(x)
        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feat, out

    def extract_features(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        feat1 = out
        out = self.layer1(out)
        feat2 = out
        out = self.layer2(out)
        feat3 = out
        out = self.layer3(out)
        feat4 = out
        out = self.layer4(out)
        out = avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (feat1, feat2, feat3, feat4), out

    def get_features_only(self, x: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        Returns the non-activated output of the specified layer.
        :param x: input tensor (batch_size, *input_shape)
        :param feat_level: level of features to extract (1-4)
        :return: output tensor (??)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        if feat_level == 1:
            return out
        out = self.layer1(out)
        if feat_level == 2:
            return out
        out = self.layer2(out)
        if feat_level == 3:
            return out
        out = self.layer3(out)
        if feat_level == 4:
            return out
        out = self.layer4(out)
        return out

    def predict_from_features(self, feats: torch.Tensor, feat_level: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Predicts from features at a specific level.
        :param feats: input features
        :param feat_level: level of features (1-4)
        :return: output tensor (??)
        """
        if feat_level == 1:
            out = self.layer1(feats)
            feat2 = out
            out = self.layer2(out)
            feat3 = out
            out = self.layer3(out)
            feat4 = out
            out = self.layer4(out)
        elif feat_level == 2:
            out = self.layer2(feats)
            feat3 = out
            out = self.layer3(out)
            feat4 = out
            out = self.layer4(out)
        elif feat_level == 3:
            out = self.layer3(feats)
            feat4 = out
            out = self.layer4(out)
        elif feat_level == 4:
            out = self.layer4(feats)
        else:
            raise ValueError(f"Invalid feat_level: {feat_level}")

        out = avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (feats, feat2, feat3, feat4), out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress:progress + torch.tensor(pp.size()).prod()].view(pp.size())
            pp.detach().copy_(cand_params)
            progress += torch.tensor(pp.size()).prod()

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            if pp.grad is not None:
                grads.append(pp.grad.view(-1))
            else:
                grads.append(torch.zeros_like(pp.view(-1)))
        return torch.cat(grads)


def gelu_resnet18(nclasses: int, nf: int=64, apply_gelu=(1, 1, 1, 1)) -> GeluResNet:
    """
    Constructs a ResNet-18 model with GELU activations.
    :param nclasses: number of classes
    :param nf: number of filters
    :param apply_gelu: tuple indicating which layers to apply GELU to
    :return: ResNet model
    """
    return GeluResNet(GeluBasicBlock, [2, 2, 2, 2], nclasses, nf, apply_gelu) 