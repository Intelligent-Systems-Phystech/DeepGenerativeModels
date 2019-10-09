import torch

from torch import nn
from torch import autograd


def forward_dynamics(state, ode_func):
    return [1.0, ode_func(state)]


def backward_dynamics(state, parameters, ode_func):
    with torch.set_grad_enabled(True):
        t = state[0]
        ht = state[1].requires_grad_(True)
        at = -state[2]
        ht_new = ode_func(inputs=[t, ht])
        gradients = autograd.grad(outputs=ht_new,
                                  inputs=[ht, parameters],
                                  grad_outputs=at,
                                  allow_unused=True)
    if len(parameters) == 0:
        return [1.0, ht_new, gradients[0]]
    else:
        return [1.0, ht_new, *gradients]


class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, timestamps, parameters, ode_func, ode_solver):
        time_intervals = timestamps[1:] - timestamps[:-1]
        states = [inputs]
        state = [timestamps[0], inputs]

        for dt in time_intervals:
            state = ode_solver(func=lambda state: forward_dynamics(state, ode_func), dt=dt, state=state)
            states.append(state[1])
        states = torch.stack(states)

        ctx.func = ode_func
        ctx.timestamps = timestamps
        ctx.ode_solver = ode_solver
        ctx.save_for_backward(states.clone(), parameters)

        return states[-1]

    @staticmethod
    def backward(ctx, output_gradients):
        ode_func = ctx.func
        timestamps = ctx.timestamps
        ode_solver = ctx.ode_solver

        states, parameters = ctx.saved_tensors
        time_intervals = timestamps[1:] - timestamps[:-1]
        outputs = states[-1]
        grad_weights = torch.zeros_like(parameters)

        if output_gradients is None:
            output_gradients = torch.zeros_like(outputs)

        t0 = timestamps[-1]
        state = [t0, outputs, output_gradients, grad_weights]

        for dt in time_intervals[::-1]:
            state = ode_solver(lambda state: backward_dynamics(state, parameters, ode_func),
                               dt=dt, state=state)
        if len(state) == 3:
            print(state[2])
            return -state[2], None, None, None, None
        else:
            return -state[2], None, state[3], None, None


class NeuralODE(nn.Module):
    def __init__(self, ode_func, timestamps, ode_solver):
        super(NeuralODE, self).__init__()
        self.timestamps = timestamps
        self.ode_func = ode_func
        self.time_intervals = timestamps[1:] - timestamps[:-1]
        self.ode_solver = ode_solver

    def forward(self, inputs):
        z = ODEAdjoint.apply(inputs,
                             self.timestamps,
                             self.get_flat_parameters(self.ode_func),
                             self.ode_func,
                             self.ode_solver)
        return z

    @staticmethod
    def get_flat_parameters(model):
        flat_parameters = []
        for p in model.parameters():
            flat_parameters.append(p.flatten())
        if len(flat_parameters) == 0:
            return torch.Tensor([]).requires_grad_(True)
        else:
            return torch.cat(flat_parameters)
