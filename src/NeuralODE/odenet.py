import numpy as np

from torch import nn

from .ode_solvers import euler_step
from .neural_ode_solvers import AdjointODE


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        stride = (1, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ODEResFunc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEResFunc, self).__init__()
        stride = (1, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = x[1]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEBlock, self).__init__()
        ode_solver = euler_step
        timestamps = np.linspace(0, 1., 11)
        self.neural_ode = AdjointODE(ode_func=ODEResFunc(in_channels, out_channels),
                                     timestamps=timestamps,
                                     ode_solver=ode_solver)

    def forward(self, x):
        self.neural_ode(x)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self, block_type, channels=(1, 64, 64, 128, 128, 10)):
        super(MNISTClassifier, self).__init__()
        stride = (1, 1)
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)

        self.layer1 = nn.Sequential(
            block_type(channels[1], channels[1]),
            block_type(channels[1], channels[1]))

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.layer2 = nn.Sequential(
            block_type(channels[2], channels[2]),
            block_type(channels[2], channels[2]))

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)

        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.layer3 = nn.Sequential(
            block_type(channels[3], channels[3]),
            block_type(channels[3], channels[3]))

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)

        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)

        self.layer4 = nn.Sequential(
            block_type(channels[4], channels[4]),
            block_type(channels[4], channels[4]))

        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(channels[4], channels[5])
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.layer2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.layer3(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.layer4(out)
        out = self.maxpool4(out)

        out = self.avgpool(out)
        out = self.dropout(out)
        out = out[:, :, 0, 0]
        out = self.fc(out)
        out = self.softmax(out)
        return out
