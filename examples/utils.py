import math

from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch import optim

from scipy.stats import norm
import numpy as np

from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# Additional function for training


def train_on_batch(model,
			       batch_of_x,
			       batch_of_y,
			       optimizer):
    """
    Function for optimize model parameters by using one batch.
    Args: 
        model:                    - training model.
        batch_of_x:   FloatTensor - the matrix of shape batch_size x input_dim.
        batch_of_y:   FloatTensor - the matrix of shape batch_size x ?.
        optimizer:    Optimimizer - optimizer from torch.optim.

    Returns:
        None

    Example:
        >>>
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
    Args: 
        train_generator: DataLoader  - generator of samples from all Dataset.
        model:                       - training model.
        optimizer:       Optimimizer - optimizer from torch.optim.
        callback:        <function>  - function wich call after each epoch.

    Returns:
        None

    Example:
        >>>
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
    Args: 
        model:                         - training model.
        ptimizer:       Optimimizer   - optimizer from torch.optim.
        dataset:         TensorDataset - train dataset.
        count_of_epoch:  int           - a number of epoch.
        batch_size:      int           - the size of batch.
        callback:        <function>    - function wich call after each epoch.
        progress:        yield         - function to display progress (for example tqdm).

    Returns:
        None

    Example:
        >>>
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
    Args: 
        model:                                          - model VAE or IWAE.
        num_row:                        int             - the number of row.
        num_colum:                      int             - the number of column.
        images_size = (x_size, y_size): tuple(int, int) - a size of input image.

    Returns: 
        figure: float - the picture

    Example:
        >>>
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


def draw_reconstucted_samples(model, batch_x, num_samples=15, images_size=(28, 28), IW_sampler = True):
    """
    Illustrate how change digits x where change point in latent space z.
    Args: 
        model:                                          - model VAE or IWAE.
        batch_x:                        Tensor          - the tensor of shape batch_size x input_dim.
        num_samples:                    int             - the number of sampled values for each image.
        images_size = (x_size, y_size): tuple(int, int) - a size of input image.
        IW_sampler:					   bool            - the flag: use Importance sampling or q_z
    
    Returns: 
        figure: float - the picture
    
    Example:
        >>>
    """
    num_row = batch_x.shape[0]
    
    figure = np.zeros((images_size[0] * num_row, images_size[1] * (num_samples + 2)))
        
    for i, x in enumerate(batch_x):
        x = x.view([1, -1])
        image = x.view((28, 28)).cpu().data.numpy()

        # draw real image
        j = 0
        figure[i * images_size[0]: (i + 1) * images_size[0],
               j * images_size[1]: (j + 1) * images_size[1]] = image
        # draw saparator betwean real and generate image
        j = 1
        figure[i * images_size[0]: (i + 1) * images_size[0],
               j * images_size[1]: (j + 1) * images_size[1]] = 1

        for j in range(2, num_samples + 2):
            if IW_sampler == True:
                z = model.sample_z_IW(x)
            else: 
                distr = model.q_z(x)
                z = model.sample_z(distr)
            x_sample = model.q_x(z).view((28, 28)).cpu().data.numpy()

            image = x_sample
            figure[i * images_size[0]: (i + 1) * images_size[0],
                   j * images_size[1]: (j + 1) * images_size[1]] = image

    return figure



def q_IW(z, K=10, latent_dim=2, q_z = None, p_z = None):
    """
    Return density probability for z from q_IW.
    Args: 
        z: numpy.array       - the matrix of shape 1 x latent_dim.
        K: int               - scalar, the number of samples in Importance Sampling.
        latent_dim:           - the space dimensional.
        q_z: <class>         - the class of approximated distribution from normal. 
                                  It's need to have method rvs for generate samples.
                                  It's need to have method logpdf for generate log of density function.
                                  Default it's multivariate_normal from scipy.stats.
        p_z: <function>      - the function witch return real density probability of z.
                                  Default it is a mixture of gaussian.
    
    Returns: 
        q_IW: numpy.array   - the scalar. The density probability for z from q_IW

    Example:
        >>>
    """
    if q_z is None:
        q_z = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
    if p_z is None:
        mixture = [multivariate_normal(mean=3*np.array([0, 1]), cov=2*np.eye(2)), 
                   multivariate_normal(mean=-1.5*np.ones(2), cov=0.75*np.eye(2)), 
                   multivariate_normal(mean=1.5*np.ones(2), cov=np.eye(2)), 
                   multivariate_normal(mean=np.zeros(2), cov=np.eye(2)), 
                   multivariate_normal(mean=3*np.array([0, 1]), cov=np.eye(2))]
        p_z = lambda x: np.log(np.mean(np.array([mix.pdf(x) for mix in mixture]), axis = 0))
    
    z_latent = q_z.rvs(K).reshape([-1, latent_dim])
    z_latent[0] = z

    exponent = np.array([p_z(z_latent)]).reshape([-1]) - np.array([q_z.logpdf(z_latent)]).reshape([-1])

    expectation = logsumexp(exponent) - np.log(K)

    proba = np.exp(np.array([p_z(z_latent)]).reshape([-1]) - expectation)
    return proba[0]


def score_ae(model, batch_x, batch_y, images_size=(28,28)):
    """
    Args: 
        model:                                              - model VAE or IWAE.
        batch_x:                        Tensor              - the tensor of shape batch_size x input_dim.
        batch_y:                        Tensor              - dont uses in the ae score computation.
        images_size = (x_size, y_size): tuple(int, int)     - a size of input image.

    Returns: 
        (mse_loss, model_loss): tuple(float, float) - the model quality

    Example:
        >>>
    """
    mse_arr = []
    
    qz = model.q_z(batch_x)
    qzsample = model.sample_z(qz)
    model_x = model.q_x(qzsample)[:,0,:]
    
    for i in range(batch_x.shape[0]):
        original_x = batch_x[i].view(images_size).cpu().data.numpy()
        reconstructed_x = model_x[i].view(images_size).cpu().data.numpy()
        
        mse_arr.append((original_x - reconstructed_x)**2)
        
    return np.array(mse_arr).mean(), model.loss(batch_x, batch_y).item()

def draw_table(data, title = ['MSE', 'Model Loss'], width = 20):
    """
    Args: 
        data: dict - is a dict with format
                    {row_name_1: (title[0], title[1], ...), 
                     row_name_2: (title[0], title[1], ...), 
                     ...}
        title: list - is the list of column name

    Returns: 
        string: str - is a string of formating data in table

    Example:
        >>>
    """
    string = ""
    row_format =("|{:>"+str(width)+"}|") * (len(title) + 1)
    string += str(row_format.format("-"*width, *["-"*width for _ in title])) + '\n'
    string += str(row_format.format("", *title)) + '\n'
    string += str(row_format.format("-"*width, *["-"*width for _ in title])) + '\n'
    for key in data:
        if len(key) > width:
            row_name = '...' + key[len(key)-width+3:]
        else:
            row_name = key
        string += str(row_format.format(row_name, *[round(x, 5) for x in data[key]]))+ '\n'
        string += str(row_format.format("-"*width, *["-"*width for _ in title]))+ '\n'

    return string

    