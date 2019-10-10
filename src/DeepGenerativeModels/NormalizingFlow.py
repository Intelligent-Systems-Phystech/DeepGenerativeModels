# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Normalizing Flows Model

# Used code from the repository https://github.com/tonyduan/normalizing-flows/

class NormalizingFlowModel(nn.Module):
    r"""Interface for Flow models."""
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        
        return z, prior_logprob, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        x = z
        
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.backward(z)
        
        return x
    

# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

# Implemented Flows

class Planar(nn.Module):
    def __init__(self, dim, nonlinearity=torch.tanh): # mb add device='cpu'
        r"""Planar flow.
        
        .. math:: 
            z = f(x) = x + u h(wᵀx + b)
            
        [Rezende and Mohamed, 2015]
        
        Args:
            dim (int): The dimension of input space (data dimensionality).
            nonlinearity (torch.nn.functional): The smooth element-wise non-linearity to use.
                Default: ``'tanh'``
            device (string): The type of computing device: 'cpu' or 'gpu'.
        """
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        r"""Reset Flow parameters. Utility method for Flow initialisation.
        
        Args: dim (int): The dimension of input space (data dimensionality).
        """
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Applies a Normalizing Flow (mapping from data `x` to the noise `z`).
        
        Args:
            x (torch.Tensor): tensor containing input data.

        Returns: (z, log_det), tuple(Tensor, Tensor) - f(x) and log|det(df/dx)|
            z (torch.Tensor): :math:`z = f(x)` - result of applying NF (same dimension as `x`), where
            :math:`f(x) = x + u h \left( w^{\top} x + b \right)` 
            log_det (torch.Tensor): logdet-Jacobian term where |det(df/dx)| equals to:
            
            .. math::
                \begin{array}{c} \\
                    \psi(z) = h^{\prime}(w^{\top} z + b) w \\
                    \left| \det \frac{\partial f}{\partial z} \right| = 
                    \left| \det \left(I + u \psi(z)^{\top}\right) \right| = 
                    \left| 1 + u^{\top} \psi(z) \right|
                \end{array}
        """
        if self.h in (F.elu, F.leaky_relu):
            u = self.u
        elif self.h == torch.tanh:
            scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
            u = self.u + scal * self.w / torch.norm(self.w)
        else:
            raise NotImplementedError("Non-linearity is not supported.")
            
        lin = torch.unsqueeze(x @ self.w, 1) + self.b
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        
        return z, log_det

    def backward(self, z):
        if self.w @ self.u >= -1:
            print("Not invertible.")
        raise NotImplementedError("Planar flow has no algebraic inverse.")


class Radial(nn.Module):
    def __init__(self, dim):
        r"""Radial flow.
        
        .. math:: 
            z = f(x) = x + β h(α, r)(z − z_0)
            
        [Rezende and Mohamed, 2015]
        
        Args:
            dim (int): The dimension of input space (data dimensionality).
            nonlinearity (torch.nn.functional): The smooth element-wise non-linearity to use.
                Default: ``'tanh'``
            device (string): The type of computing device: 'cpu' or 'gpu'.
        """
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(dim):
        init.uniform_(self.z0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Applies a Normalizing Flow (mapping from data `x` to the noise `z`).
        
        Args:
            x (torch.Tensor): tensor containing input data.
            
        Returns: (z, log_det), tuple(Tensor, Tensor) - f(x) and log|det(df/dx)|
            z (torch.Tensor): :math:`z = f(x)` - result of applying NF (same dimension as `x`), where
            :math:`f(x) = x + \beta h(\alpha, r) \left( x - x_0 \right) ` 
            log_det (torch.Tensor): logdet-Jacobian term where |det(df/dx)| equals to:
            .. math::
                \left| \det \frac{\partial f}{\partial z} \right| = 
                [1 + \beta h(\alpha, r)]^{d-1} [1 + \beta h(\alpha, r) + \beta h^{\prime}(\alpha, r) r)],
                
            where :math:`r = |x − x_0|, h(\alpha, r) = 1 / (\alpha + r)`
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det