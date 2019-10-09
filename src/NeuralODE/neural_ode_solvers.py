import torch

from torch import nn
from torch import autograd


def get_flat_parameters(parameters):
    flat_parameters = []
    for p in parameters:
        flat_parameters.append(p.flatten())
    if len(flat_parameters) == 0:
        return torch.Tensor([]).requires_grad_(True)
    else:
        return torch.cat(flat_parameters)


def forward_dynamics(state, ode_func):
    return [1.0, ode_func(state)]


def backward_dynamics(state, parameters, ode_func):
    with torch.set_grad_enabled(True):
        t = state[0]
        ht = state[1].requires_grad_(True)
        at = -state[2]
        ht_new = ode_func([t, ht])
        gradients = autograd.grad(outputs=ht_new,
                                  inputs=[ht] + list(ode_func.parameters()),
                                  grad_outputs=at)
    gradients_w = get_flat_parameters(gradients[1:])
    return [1.0, ht_new, gradients[0], gradients_w]


class AdjointODEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, timestamps, parameters, ode_func, ode_solver):
        time_intervals = timestamps[1:] - timestamps[:-1]
        states = [inputs]
        state = [timestamps[0], inputs]

        for dt in time_intervals:
            state = ode_solver(func=lambda input_state: forward_dynamics(input_state, ode_func), dt=dt, state=state)
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
            state = ode_solver(lambda input_state: backward_dynamics(input_state, parameters, ode_func),
                               dt=dt, state=state)

        return -state[2], None, -state[3], None, None


class AdjointODE(nn.Module):
    def __init__(self, ode_func, timestamps, ode_solver):
        super(AdjointODE, self).__init__()
        self.timestamps = timestamps
        self.ode_func = ode_func
        self.time_intervals = timestamps[1:] - timestamps[:-1]
        self.ode_solver = ode_solver

    def forward(self, inputs):
        output = AdjointODEFunc.apply(inputs,
                                      self.timestamps,
                                      get_flat_parameters(self.ode_func.parameters()),
                                      self.ode_func,
                                      self.ode_solver)
        return output


class AutogradODE(nn.Module):
    def __init__(self, timestamps, ode_solver, ode_func):
        super(AutogradODE, self).__init__()
        self.timestamps = timestamps
        self.ode_func = ode_func
        self.time_intervals = timestamps[1:] - timestamps[:-1]
        self.ode_solver = ode_solver

    def forward_dynamics(self, state):
        return [1.0, self.ode_model(state)]

    def forward(self, inputs):
        states = [inputs]
        state = [self.timestamps[0], inputs]
        for dt in self.time_intervals:
            state = self.ode_solver(func=self.forward_dynamics, dt=dt, state=state)
            states.append(state[1])
        return states
