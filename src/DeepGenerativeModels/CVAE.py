# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import defaultdict


def idx2onehot(idx, num_labels):
    """
    One-hot encoding of array
    :param idx: torch array of labels
    :param num_labels: int, number of possible labels
    :return: array of 0-1 codes for labels
    """
    if num_labels and torch.max(idx).item() >= num_labels:
        raise ValueError('all idx must be less then num_labels')
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), num_labels)
    onehot.scatter_(1, idx, 1)
    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditional=False, num_labels=0):
        """
        Creation of VAE by parameters
        :param encoder_layer_sizes: list of int, layers of MLP
        :param latent_size: int, dimension of latent space
        :param decoder_layer_sizes: list of int, layers of MLP
        :param conditional: bool, True for CVAE, False for simple VAE
        :param num_labels: int, 0 for VAE, positive for CVAE
        """
        super().__init__()
        if conditional:
            if num_labels <= 0:
                raise ValueError('CVAE demands positive num_labels')

        if not isinstance(encoder_layer_sizes, list):
            raise TypeError('encoder_layer_sizes must be list')
        if not isinstance(latent_size, int):
            raise TypeError('latent_size must be int')
        if not isinstance(decoder_layer_sizes, list):
            raise TypeError('decoder_layer_sizes must be list')

        self.latent_size = latent_size
        self.conditional = conditional
        self.num_labels = num_labels
        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, cond=None):
        """
        Forward: calculates result of encoder, latent variable and reconstructed value
        :param x: image
        :param cond: label (None for VAE)
        :return: tuple (recon_x, means, log_var, z)
        :return: recon_x --- reconstructed by decoder x
        :return: means, log_var --- result of encoder
        :return: z --- latent vector for x
        """
        if x.dim() > 2:
            x = x.view(-1, 28*28)
        batch_size = x.size(0)
        means, log_var = self.encoder(x, cond)
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * sigma + means
        recon_x = self.decoder(z, cond)
        return recon_x, means, log_var, z

    def inference(self, batch_size=1, cond=None):
        """
        Samples and decodes from latent space
        :param batch_size: number of samples
        :param cond: label (for VAE)
        :return: x reconstructed from sampled z image 28x28
        """
        z = torch.randn([batch_size, self.latent_size])
        recon_x = self.decoder(z, cond)
        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        """
        Creates encoder image to latent space
        :param layer_sizes: list of int, layers of MLP
        :param latent_size: int, dimension of latent space
        :param conditional: bool, True for CVAE, False for simple VAE
        :param num_labels: int, number of possible labels
        """
        super().__init__()
        self.num_labels = num_labels
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, cond=None):
        """
        Encodes image
        :param x: image 28x28
        :param cond: label (for VAE)
        :return: parameters of distribution for latent space
        """
        if self.conditional:
            cond = idx2onehot(cond, num_labels=self.num_labels)
            x = torch.cat((x, cond), dim=-1)

        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        """
        Creates decoder image from latent space
        :param layer_sizes: list of int, layers of MLP
        :param latent_size: int, dimension of latent space
        :param conditional: bool, True for CVAE, False for simple VAE
        :param num_labels: int, 0 for VAE, positive for CVAE
        """
        super().__init__()
        self.num_labels = num_labels
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

    def forward(self, z, cond):
        """
        Decodes from latent space
        :param z: latent vector
        :param cond: label (for VAE)
        :return: reconstructed image 28x28
        """
        if self.conditional:
            cond = idx2onehot(cond, num_labels=self.num_labels)
            z = torch.cat((z, cond), dim=-1)
        x = self.MLP(z)
        return x


def loss_VAE(recon_x, x, mean, log_var):
    """
    loss function for VAE
    :param recon_x: reconstructed image 28x28
    :param x: original image 28x28
    :param mean, log_var: results of encoder
    :return: loss including cross entropy and KL-divergence
    """
    BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)


def train_VAE(vae, device, data_loader, loss_fn=loss_VAE, verbose=True, freq_print=100, learning_rate=0.001, epochs=50):
    """
    Trains VAE instance
    :param vae: instance to train
    :param device: device for calculations
    :param data_loader: loader data for train
    :param loss_fn: loss function, should take recon_x, x, mean, log_var
    :param freq_print: number of elements processed between prints
    :param learning_rate: lr parameter for Adam optimizer
    :param epochs: number of epochs
    :return: log of loss values
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    logs = defaultdict(list)
    for epoch in range(epochs):
        for iteration, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            if vae.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            loss = loss_fn(recon_x, x, mean, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs['loss'].append(loss.item())

            if iteration % freq_print == 0 or iteration == len(data_loader) - 1:
                if verbose:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch, epochs, iteration, len(data_loader) - 1, loss.item()))

                if vae.conditional:
                    conds = torch.arange(0, 10).long().unsqueeze(1)
                    x = vae.inference(batch_size=conds.size(0), cond=conds)
                else:
                    x = vae.inference(batch_size=10)
    return logs
