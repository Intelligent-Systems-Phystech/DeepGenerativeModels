# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import defaultdict


def idx2onehot(idx, num_labels):

    if num_labels and torch.max(idx).item() >= num_labels:
        raise ValueError('all idx must be less then num_labels')
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), num_labels)
    onehot.scatter_(1, idx, 1)
    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()
        if conditional:
            if num_labels <= 0:
                raise ValueError('CVAE demands positive num_labels')

        if type(encoder_layer_sizes) != list:
            raise TypeError('encoder_layer_sizes must be list')
        if type(latent_size) != int:
            raise TypeError('latent_size must be int')
        if type(decoder_layer_sizes) != list:
            raise TypeError('decoder_layer_sizes must be list')

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)
        batch_size = x.size(0)
        means, log_var = self.encoder(x, c)
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * sigma + means
        recon_x = self.decoder(z, c)
        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        recon_x = self.decoder(z, c)
        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, num_labels=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()
        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, num_labels=10)
            z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


def train_vae(vae, device, data_loader, loss_fn, freq_print, learning_rate, epochs):

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    logs = defaultdict(list)
    for epoch in range(epochs):
        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        for iteration, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            if vae.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id_ = len(tracker_epoch)
                tracker_epoch[id_]['x'] = z[i, 0].item()
                tracker_epoch[id_]['y'] = z[i, 1].item()
                tracker_epoch[id_]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs['loss'].append(loss.item())

            if iteration % freq_print == 0 or iteration == len(data_loader) - 1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, epochs, iteration, len(data_loader) - 1, loss.item()))

                if vae.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=10)
