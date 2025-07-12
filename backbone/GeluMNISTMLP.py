# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Gelu(nn.Module):
    """GELU activation function implementation"""
    def __init__(self):
        super(Gelu, self).__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to the parameters of the given module.
    :param m: module whose parameters will be initialized
    """
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GeluMNISTMLP(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 GELU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(GeluMNISTMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()

        fc1 = nn.Linear(self.input_size, 100)
        fc2 = nn.Linear(100, 100)

        gelu1 = Gelu()
        gelu2 = Gelu()

        self._layers = nn.ModuleList([fc1, fc2])
        self._activations = nn.ModuleList([gelu1, gelu2])
        self.classifier = nn.Linear(100, self.output_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self._layers.apply(xavier)
        self.classifier.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :param returnt: return type (out, features, or all)
        :return: output tensor (output_classes)
        """
        x = x.view(-1, self.input_size)

        features = []
        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            x = layer(x)
            x = activation(x)
            features.append(x)

        x = self.classifier(x)

        if returnt == 'features':
            return features
        elif returnt == 'all':
            return x, features
        else:
            return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (??)
        """
        x = x.view(-1, self.input_size)

        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            x = layer(x)
            if i < len(self._layers) - 1:  # Don't apply activation to last layer
                x = activation(x)

        return x

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