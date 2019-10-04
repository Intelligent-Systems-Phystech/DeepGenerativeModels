import math

from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch import optim

from scipy.stats import norm
import numpy as np

from matplotlib import pyplot as plt

# Additional function for training


def train_on_batch(model,
			       batch_of_x,
			       batch_of_y,
			       optimizer):
    """
    Function for optimize model parameters by using one batch.
    Input: model,                    - training model.
    Input: batch_of_x,   FloatTensor - the matrix of shape batch_size x input_dim.
    Input: batch_of_y,   FloatTensor - the matrix of shape batch_size x ?.
    Input: optimizer,    Optimimizer - optimizer from torch.optim.
    """
    model.zero_grad()

    loss = model.loss(batch_of_x, batch_of_y)

    loss.backward()

    optimizer.step()

    return


def train_epoch(model,
	        	train_generator,
	        	optimizer,
	        	callback=None):
    """
    Function for optimize model parameters by using all Dataset.
    Input: train_generator, DataLoader  - generator of samples from all Dataset.
    Input: model,                       - training model.
    Input: optimizer,       Optimimizer - optimizer from torch.optim.
    Input: callback,        <function>  - function wich call after each epoch.
    """
    model.train()
    for it, (batch_of_x, batch_of_y) in enumerate(train_generator):
        train_on_batch(model, batch_of_x, batch_of_y, optimizer)

    if callback is not None:
        callback(model)
    return


def trainer(model,
            optimizer,
            dataset,
            count_of_epoch=5,
            batch_size=64,
            callback=None,
            progress=None):
    """
    Function for optimize model parameters by using all Dataset count_of_epoch times.
    Input: model,                         - training model.
    Input: optimizer,       Optimimizer   - optimizer from torch.optim.
    Input: dataset,         TensorDataset - train dataset.
    Input: count_of_epoch,  int           - a number of epoch.
    Input: batch_size,      int           - the size of batch.
    Input: callback,        <function>    - function wich call after each epoch.
    Input: progress,        yield         - function to display progress (for example tqdm).
    """
    iterations = range(count_of_epoch)

    if progress is not None:
        iterations = progress(iterations)

    for it in iterations:

        batch_generator = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True)

        train_epoch(
        	model=model,
            train_generator=batch_generator,
            optimizer=optimizer,
            callback=callback)

    return


def draw_samples_grid_vae(model,
    					  num_row=15,
    					  num_colum=15,
    					  images_size=(28, 28)):
    """
    Illustrate how change digits x where change point in latent space z.
    Input: model,                                          - model VAE or IWAE.
    Input: num_row,                        int             - the number of row.
    Input: num_colum,                      int             - the number of column.
    Input: images_size = (x_size, y_size), tuple(int, int) - a size of input image.
    """

    grid_x = norm.ppf(np.linspace(0.05, 0.95, num_colum))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, num_row))

    figure = np.zeros((images_size[0] * num_colum, images_size[1] * num_row))
    for i, y_i in enumerate(grid_x):
        for j, x_i in enumerate(grid_y):
            z_sample = np.array([[x_i, y_i]])

            x_sample = model.q_x(torch.from_numpy(z_sample).float()).view(
                images_size).cpu().data.numpy()

            image = x_sample
            figure[i * images_size[0]: (i + 1) * images_size[0],
                   j * images_size[1]: (j + 1) * images_size[1]] = image

    return figure
