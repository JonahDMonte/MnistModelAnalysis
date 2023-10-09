from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time


class LinearNet1(nn.Module): #2 linears
    def __init__(self, use_softmax=False, use_relu=False, two_layers_neuron_count=2, intermediate_layer_count=1, test_num=0):
        super(LinearNet1, self).__init__()
        self.use_softmax = use_softmax
        self.use_relu = use_relu
        self.two_layers_neuron_count = two_layers_neuron_count
        self.intermediate_layer_count = intermediate_layer_count
        self.test_num = test_num

        self.fc0 = nn.Linear(784, 10)

        self.fc1 = nn.Linear(784, self.two_layers_neuron_count)
        self.fc2 = nn.Linear(self.two_layers_neuron_count, 10)


    def test0(self, x): # one basic layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.use_relu:
            x = F.relu(x)
        if self.use_softmax:
            x = F.log_softmax(x, dim=1)
        return x

    def test1(self, x): # two layers, changing neuron count (matrix size)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.use_relu:
            x = F.relu(x)
        x = self.fc2(x)
        if self.use_relu:
            x = F.relu(x)
        if self.use_softmax:
            x = F.log_softmax(x, dim=1)
        return x

    def test2(self, x): #changing layer count by powers of two
        nums = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

        x = torch.flatten(x, 1)
        x = nn.Linear(784, 512)(x)
        if self.use_relu:
            x = F.relu(x)

        for i in range(self.intermediate_layer_count + 1):
            x = nn.Linear(nums[i], nums[i+1])
            if self.use_relu:
                x = F.relu(x)

        if self.use_softmax:
            x = F.log_softmax(x, dim=1)
        return x

    def forward(self, x):
        if self.test_num == 0:
            return (self.test0(x))
        elif self.test_num == 1:
            return (self.test1(x))
        elif self.test_num == 2:
            return(self.test2(x))




class T0_0(nn.Module): #2 linears
    def __init__(self):
        super(T0_0, self).__init__()
        self.fc0 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        return x



