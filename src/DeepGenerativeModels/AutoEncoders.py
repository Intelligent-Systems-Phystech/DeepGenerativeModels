# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch import optim

# Variational Auto Encoder


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim=200, device='cpu'):
        """
        Standart model of VAE with ELBO optimization.
        Input: latent_dim,  int     - the dimension of latent space.
        Input: input_dim,   int     - the dimension of input space.
        Input: device,      string  - the type of computing device: 'cpu' or 'gpu'.
        Input: hidden_dim,  int     - the size of hidden_dim neural layer.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.proposal_z = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.proposal_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.proposal_sigma = nn.Linear(hidden_dim, self.latent_dim)

        self.generative_network = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid()
        )

        self.to(device)

    def q_z(self, x):
        """
        Generates distribution of z provided x.
        Input: x, Tensor - the matrix of shape batch_size x input_dim.

        Return: (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape batch_size x latent_dim.
            sigma, Tensor - the matrix of shape batch_size x latent_dim.
        """
        x = x.to(self.device)

        proposal = self.proposal_z(x)
        mu = self.proposal_mu(proposal)
        sigma = torch.nn.Softplus()(self.proposal_sigma(proposal))
        return mu, sigma

    def p_z(self, num_samples):
        """
        Generetes prior distribution of z.
        Input: num_samples, int - the number of samples.

        Return: (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape num_samples x latent_dim.
            sigma, Tensor - the matrix of shape num_samples x latent_dim.
        """
        mu = torch.zeros([num_samples, self.latent_dim], device=self.device)
        sigma = torch.ones([num_samples, self.latent_dim], device=self.device)
        return mu, sigma

    def sample_z(self, distr, num_samples=1):
        """
        Generates samples from normal distribution q(z|x).
        Input: distr = (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape batch_size x latent_dim.
            sigma, Tensor - the matrix of shape batch_size x latent_dim.
        Input: num_samples, int - the number of samples for each element.


        Return: Tensor - the tensor of shape n x num_samples x latent_dim - samples from normal distribution in latent space.
        """
        mu, sigma = distr
        mu = mu.to(self.device)
        sigma = sigma.to(self.device)

        batch_size = mu.shape[0]

        bias = mu.view([batch_size, 1, self.latent_dim])

        epsilon = torch.randn([batch_size, num_samples, self.latent_dim], 
                                requires_grad=True, 
                                device=self.device)
        scale = sigma.view([batch_size, 1, self.latent_dim])

        return bias + epsilon * scale

    def q_x(self, z):
        """
        Given the latent representation matrix z, returns the matrix of Bernoulli distribution parameters for sampling x objects.
        Input: z, Tensor - the tensor of shape batch_size x num_samples x latent_dim, samples from latent space.

        Return: Tensor - the tensor of shape batch_size x num_samples x input_dim, Bernoulli distribution parameters.
        """
        z = z.to(self.device)
        out = self.generative_network(z)

        return torch.clamp(out, 0.01, 0.99)

    def loss(self, batch_x, batch_y):
        """
        Calculate ELBO approximation of log likelihood for given batch with negative sign.
        Input: batch_x, FloatTensor - the matrix of shape batch_size x input_dim.
        Input: batch_y, FloatTensor - dont uses parameter in this model.

        Return: Tensor - scalar, ELBO approximation of log likelihood for given batch with negative sign.
        """
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        batch_size = batch_x.shape[0]

        propos_distr = self.q_z(batch_x)
        pri_distr = self.p_z(batch_size)

        x_distr = self.q_x(self.sample_z(propos_distr))

        expectation = torch.mean(
            self.log_mean_exp(
                self.log_likelihood(
                    batch_x, x_distr)), dim=0)

        divergence = self.divergence_KL_normal(propos_distr, pri_distr)

        return -1 * torch.mean(expectation - divergence, dim=0)

    def generate_samples(self, num_samples):
        """
        Generate samples of object x from noises in latent space.
        Input: num_samples, int - the number of samples, wich need to generate.

        Return: Tensor - the matrix of shape num_samples x input_dim.
        """
        distr_z = self.p_z(num_samples=1)

        z = self.sample_z(distr, num_samples=num_samples)

        distr_x = self.q_x(z).view([num_samples, -1])

        return torch.bernoulli(distr_x, device = self.device)

    @staticmethod
    def log_pdf_normal(distr, samples):
        """
        The function calculates the logarithm of the probability density at a point relative to the corresponding normal distribution given componentwise by its mean and standard deviation.
        Input: distr = (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape batch_size x latent_dim.
            sigma, Tensor - the matrix of shape batch_size x latent_dim.
        Input: samples, Tensor - the tensor of shape batch_size x num_samples x latent_dim, samples in latent space.

        Return: Tensor - the matrix of shape batch_size x num_samples, each element of which is the logarithm of the probability density of a point relative to the corresponding distribution.
        """
        mu, sigma = distr

        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]

        f1 = torch.sum(((samples -
                         mu.view([batch_size, 1, latent_dim]))**2) /
                       sigma.view([batch_size, 1, latent_dim])**2, dim=2)
        f2 = mu.shape[1] * (math.log(2) + math.log(math.pi))
        f3 = torch.sum(torch.log(sigma), dim=1).view(batch_size, 1)
        return -0.5 * (f1 + f2) - f3

    @staticmethod
    def log_likelihood(x_true, x_distr):
        """
        Calculate log likelihood between x_true and x_distr.
        Input: x_true,  Tensor - the matrix of shape batch_size x input_dim.
        Input: x_distr, Tensor - the tensor of shape batch_size x num_samples x input_dim, Bernoulli distribution parameters.

        Return: Tensor - the matrix of shape batch_size x num_samples - log likelihood for each sample.
        """
        batch_size = x_distr.shape[0]
        input_dim = x_distr.shape[2]

        bernoulli_log_likelihood = torch.log(
            x_distr) * x_true.view([batch_size, 1, input_dim])
        bernoulli_log_likelihood += torch.log(1 - x_distr) * (
            1 - x_true).view([batch_size, 1, input_dim])

        return torch.sum(bernoulli_log_likelihood, dim=2)

    @staticmethod
    def log_mean_exp(data):
        """
        Input: data, Tensor - the tensor of shape n_1 x n_2 x ... x n_K.

        Return: Tensor - the tensor of shape n_1 x n_2 x ,,, x n_{K - 1}.
        """

        return torch.logsumexp(data, dim=-1) - \
            torch.log(torch.Tensor([data.shape[-1]]).to(data.device))

    @staticmethod
    def divergence_KL_normal(q_distr, p_distr):
        """
        Calculate KL-divergence KL(q||p) between n-pairs of normal distribution.
        Input: q_distr = (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape batch_size x latent_dim.
            sigma, Tensor - the matrix of shape batch_size x latent_dim.
        Input: p_distr = (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape batch_size x latent_dim.
            sigma, Tensor - the matrix of shape batch_size x latent_dim.
        Return: Tensor - the vector of shape n, each value of which is a KL-divergence between pair of normal distribution.
        """
        q_mu, q_sigma = q_distr
        p_mu, p_sigma = p_distr

        D_KL = torch.sum((q_sigma / p_sigma)**2, dim=1)
        D_KL -= p_mu.shape[1]
        D_KL += 2 * torch.sum(torch.log(p_sigma), dim=1) - \
            2 * torch.sum(torch.log(q_sigma), dim=1)
        D_KL += torch.sum((p_mu - q_mu) * (p_mu - q_mu) / (p_sigma**2), dim=1)
        return 0.5 * D_KL

# Variational Auto Encoder with Importance Sampling


class IWAE(VAE):
    def __init__(
            self,
            latent_dim,
            input_dim,
            K=2,
            hidden_dim=200,
            device='cpu'):
        """
        Standart model of IWAE.
        Input: latent_dim,     int     - the dimension of latent space.
        Input: input_dim,      int     - the dimension of input space.
        Input: K,              int     - the rang of Lower Bound.
        Input: device,         string  - the type of computing device: 'cpu' or 'gpu'.
        Input: hidden_dim,     int     - the size of hidden_dim neural layer.
        """
        super(IWAE, self).__init__(latent_dim, input_dim, hidden_dim, device)

        self.K = K

        self.to(device)

    def q_IW(self, z, x):
        """
        Return density value for sample z for q_IW(z|x).
        Input: x, FloatTensor - the matrix of shape batch_size_x x input_dim.
        Input: z, FloatTensor - the matrix of shape 1 x latent_dim.
        
        Return: FloatTensor - matrix of shape batch_size_x x 1.
        """
        z = z.to(self.device)
        x = x.to(self.device)

        propos_distr = self.q_z(x)
        pri_distr = self.p_z(x.shape[0])
        
        z_latent = self.sample_z(propos_distr, self.K)
        z_latent[:, 0, :] = z

        x_distr = model.q_x(z_latent)

        log_likelihood_true_distr = model.log_likelihood(batch_x, x_distr)

        normal_log_pdf_prior = model.log_pdf_normal(pri_distr, z_latent)

        normal_log_pdf_propos = model.log_pdf_normal(propos_distr, z_latent)

        exponent = log_likelihood_true_distr + normal_log_pdf_prior - normal_log_pdf_propos

        expectation = model.log_mean_exp(exponent)

        log_weight = log_likelihood_true_distr + normal_log_pdf_prior - expectation.view(-1, 1)
        
        proba = torch.exp(log_weight)
        
        return proba[:, :1]

    def sample_z_IW(self, distr, num_samples=1):
        """
        Generates samples z from q_IW(z|x).
        Input: distr = (mu, sigma), tuple(Tensor, Tensor) - the normal distribution parameters.
            mu,    Tensor - the matrix of shape 1 x latent_dim.
            sigma, Tensor - the matrix of shape 1 x latent_dim.
        Input: 1 x latent_dim, int - the number of samples for each element.


        Return: Tensor - the tensor of shape n x num_samples x latent_dim - samples from normal distribution in latent space.
        """
        
        pri_distr = self.p_z(batch_x.shape[0])
       
        z_latent = self.sample_z(distr, self.K)
        
        x_distr = self.q_x(z_latent)
        
        log_likelihood_true_distr = self.log_likelihood(batch_x, x_distr)

        normal_log_pdf_prior = self.log_pdf_normal(pri_distr, z_latent)

        normal_log_pdf_propos = self.log_pdf_normal(propos_distr, z_latent)

        log_weight = log_likelihood_true_distr + normal_log_pdf_prior - normal_log_pdf_propos

        weight = torch.softmax(log_weight, dim = 1)

        cat = torch.distributions.Categorical(weight)
        
        indexes = cat.sample_n(1)

        return z_latent[:, indexes, :][0][0]


    def loss(self, batch_x, batch_y):
        """
        Calculate k-sample lower bound approximation of log likelihood for given batch with negative sign.
        Input: batch_x, FloatTensor - the matrix of shape batch_size x input_dim.
        Input: batch_y, FloatTensor - dont uses parameter in this model.

        Return: Tensor - scalar, k-sample lower bound approximation of log likelihood for given batch with negative sign.
        """
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        propos_distr = self.q_z(batch_x)
        pri_distr = self.p_z(batch_x.shape[0])

        z_latent = self.sample_z(propos_distr, num_samples=self.K)
        x_distr = self.q_x(z_latent)

        log_likelihood_true_distr = self.log_likelihood(batch_x, x_distr)
        normal_log_pdf_prior = self.log_pdf_normal(pri_distr, z_latent)
        normal_log_pdf_propos = self.log_pdf_normal(propos_distr, z_latent)

        exponent = log_likelihood_true_distr + \
            normal_log_pdf_prior - normal_log_pdf_propos

        expectation = torch.mean(self.log_mean_exp(exponent), dim=0)

        return -1 * torch.mean(expectation, dim=0)





