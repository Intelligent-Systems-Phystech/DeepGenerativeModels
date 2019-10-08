# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch import optim

# Variational Auto Encoder


class NormalizingFlow(nn.Module):
    def __init__(self, device='cpu'):
        """
        Standart model of Normalizing Flow.
        Input: device,      string  - the type of computing device: 'cpu' or 'gpu'.
        """
        super(NormalizingFlow, self).__init__()
        self.device = device