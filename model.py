import torch
from torch import nn

# from https://github.com/jzbontar/pixelcnn-pytorch

class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv, self).forward(x)
    
    
class PixelCNN(nn.Module):
    def __init__(self, n_layers=4, kernel_size=7, out_channels=64):
        super(PixelCNN, self).__init__()
        self.n_layers = n_layers
        padding = kernel_size // 2
        
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            Mask = 'A' if i == 0 else 'B'
            in_channels = 1 if i == 0 else out_channels
        
            if (i == self.n_layers - 1):
                self.layers.append(nn.Conv2d(out_channels, 256, 1))
            else:
                block = nn.Sequential(
                    MaskedConv(Mask, in_channels , out_channels , kernel_size, 1, padding, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True))
                self.layers.append(block)
                
    def forward(self, t):
        for i in range(self.n_layers):
            t = self.layers[i](t)
        return t