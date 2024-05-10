import torch.nn as nn
from typing import Optional

NUM_RESIDUAL_BLOCKS = 10

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, down_sample: Optional[nn.Module]=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample:
            identity = self.down_sample(identity)

        out += identity
        out = self.relu(out)
        return out
    
class Net(nn.Module):
    def __init__(self, input_dims, n_actions, conv_units):
        super().__init__()

        self.down_sample = nn.Sequential(
                conv1x1(1, conv_units, stride=1),
                nn.BatchNorm2d(conv_units),
                nn.ReLU(inplace=True))

        self.conv_units = conv_units
        self.in_planes = self.conv_units
        self.input_dims = input_dims

        self.residual_layers = self._make_layers(BasicBlock, conv_units, NUM_RESIDUAL_BLOCKS)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_units * input_dims[-1] * input_dims[-2], n_actions)
        self.avgpool = nn.AdaptiveAvgPool2d((input_dims[-1], input_dims[-2]))
        self.tanh = nn.Tanh()

    def _make_layers(self, block, planes, blocks, stride=1):

        layers = []

        layers.append(block(self.conv_units, self.conv_units, stride))

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.down_sample(x)

        x = self.residual_layers(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x