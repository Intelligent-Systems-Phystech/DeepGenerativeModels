from model import PixelCNN

import numpy as np
import time

import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms
from torchvision import datasets, utils, transforms

Pixel = PixelCNN(n_layers=8, kernel_size=7, out_channels=64)

train = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test =  torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_batches = torch.utils.data.DataLoader(train, batch_size=100)
test = torch.utils.data.DataLoader(test)

optimizer = optim.Adam(Pixel.parameters(), lr=0.01)

for epoch in range(10):
    # train
    err_train = []
    time_train = time.time()
    Pixel.train(True)
    for images, _ in train_batches:
        images = images.to(device='cuda')
        
        targets = (images.data[:,0] * 255).long()
        targets = targets.to(device='cuda')
        
        preds = Pixel(images)
        
        loss = F.cross_entropy(preds, targets)
        err_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    time_train = time.time() - time_train

    # test
    err_test = []
    time_test = time.time()
    Pixel.train(False)
    for images, _ in test:
        images = images.to(device='cuda')
        
        targets = (images.data[:,0] * 255).long()
        targets = targets.to(device='cuda')
       
        preds = Pixel(images)
        
        loss = F.cross_entropy(preds, targets)
        err_test.append(loss.item())
    
    time_test = time.time() - time_test

    print ('epoch={}; error_train={:.7f}; error_test={:.7f}; time_train={:.1f}s; time_test={:.1f}s'.format(
        epoch, np.mean(err_train), np.mean(err_test), time_train, time_test))

# sample
sample = torch.Tensor(256, 1, 28, 28).fill_(1)
Pixel.train(False)
for i in range(28):
    for j in range(28):
        out = Pixel(sample)
        probs = F.softmax(out[:, :, i, j]).data
        sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
    utils.save_image(sample, 'sample.png', nrow=32, padding=0)