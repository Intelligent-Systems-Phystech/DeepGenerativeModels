# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim

# GatedPixelCNN


class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, *args, vertical=False, mask="B", **kwargs):
        """
        Masked conv to PixelCNN
        Kernel of conv is masked like this:
        | 1 1    1    1 1 |
        | 1 1    1    1 1 |
        | 1 1 1 if B  0 0 |
        | 0 0    0    0 0 |
        | 0 0    0    0 0 |
        """
        super(MaskedConvolution2D, self).__init__(
            in_channels, out_channels,
            kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pr_m = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        yc, xc = kh // 2, kw // 2
        if vertical:
            if mask == "A":
                pr_m[:, :, yc:, :] = 0.0
            if mask == "B":
                pr_m[:, :, yc+1:, :] = 0.0
        else:
            pr_m[:, :, yc+1:, :] = 0.0
            pr_m[:, :, yc, xc:] = 0.0

        self.register_buffer("mask", torch.from_numpy(pr_m))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)


class GatedPixelCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size,
                 residual_vertical=True, mask="B"):
        super().__init__()
        self.out_channels = out_channels
        if mask == "A":
            assert not residual_vertical
        else:
            assert mask == "B"
        padding = filter_size // 2
        self.vertical_conv = MaskedConvolution2D(
            in_channels, 2 * out_channels, (filter_size, filter_size),
            padding=padding, vertical=True, mask=mask)
        self.v_to_h_conv = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)

        self.horizontal_conv = MaskedConvolution2D(
            in_channels, 2 * out_channels, (1, filter_size),
            padding=(0, padding), vertical=False, mask=mask)
        self.residual_vertical = None

        self.horizontal_output = nn.Conv2d(out_channels, out_channels, 1)

    def _gate(self, x):
        return torch.tanh(x[:, :self.out_channels, :, :]) *\
               F.sigmoid(x[:, self.out_channels:, :, :])

    def forward(self, v, h, conditional_image=None, conditional_vector=None):
        horizontal_preactivation = self.horizontal_conv(h)  # 1xN
        vertical_preactivation = self.vertical_conv(v)  # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation)  # 1x1

        horizontal_preactivation = horizontal_preactivation + v_to_h
        v_out = self._gate(vertical_preactivation)
        h_activated = self._gate(horizontal_preactivation)
        return v_out, h_activated


class GatedPixelCNNBlock(nn.Module):
    def __init__(self, kernel_size, channels, is_activation=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        self.GatedPixelLayer = GatedPixelCNNLayer(channels,
                                                  channels, kernel_size)
        self.BatchNorm = nn.BatchNorm2d(channels)
        self.is_activation = is_activation
        if is_activation:
            self.activation = nn.ReLU()

    def forward(self, inputs):
        v_prev_layer, h_prev_layer = inputs
        v_layer, h_layer = self.GatedPixelLayer(v_prev_layer, h_prev_layer)
        v_layer = self.BatchNorm(v_layer)
        h_layer = self.BatchNorm(h_layer)
        if self.is_activation:
            v_layer = self.activation(v_layer)
            h_layer = self.activation(h_layer)
        return (v_layer, h_layer)


class MNISTGatedPixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, num_layers, num_colors=2):
        super().__init__()
        padding = kernel_size//2
        self.MaskedConv_A = MaskedConvolution2D(
            in_channels, out_channels, (kernel_size, kernel_size),
            padding=padding, mask="A")
        self.layers = []
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            self.layers.append(GatedPixelCNNBlock(kernel_size, out_channels))
            self.add_module('Block'+str(i), self.layers[i])
        self.masked_convB1 = MaskedConvolution2D(
            out_channels, out_channels, kernel_size=1, mask='B')
        self.masked_convB2 = MaskedConvolution2D(
            out_channels, num_colors, kernel_size=1, mask='B')

    def implement_model(self, inputs):
        out = self.MaskedConv_A(inputs)
        out = (out, out)
        for layer in self.layers:
            out = layer(out)

        out = out[1]
        out = self.masked_convB1(out)
        out = self.masked_convB2(out)

        out = torch.log_softmax(out, dim=1)
        return out

    def forward(self, inputs):
        if module.training:
            out = implement_model(self, inputs)
        else:
            N = inputs.size()[2]
            M = inputs.size()[3]
            out = inputs
            for i in range(M*N):
                out = implement_model(self, inputs)
                out = torch.argmax(out, dim=1)
        return out
