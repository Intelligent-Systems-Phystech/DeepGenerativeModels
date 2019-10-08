import torch

from torch import nn


def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]


def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)


class AdjointODE(nn.Module):
    def __init__(self, timestamps, ode_solver, ode_model):
        super(AdjointODE, self).__init__()
        self.timestamps = timestamps
        self.ode_model = ode_model
        self.time_intervals = timestamps[1:] - timestamps[:-1]
        self.ode_solver = ode_solver

    def forward_dynamics(self, state):
        return [1.0, self.ode_model(state)]

    def backward_dynamics(self, state):
        t = state[0]
        ht = state[1]
        at = -state[2]
        ht_new = self.ode_model(inputs=[t, ht])
        gradients = torch.autograd.grad(outputs=ht_new, inputs=[ht] + list(self.ode_model.parameters()),
                                        grad_outputs=at)
        return [1.0, ht_new, *gradients]

    def forward(self, inputs):
        states = [inputs]
        state = [self.timestamps[0], inputs]
        for dt in self.time_intervals:
            state = self.ode_solver(func=self.forward_dynamics, dt=dt, state=state)
            states.append(state[1])
        return states

    def backward(self, outputs, output_gradients=None):

        grad_weights = [torch.zeros_like(w) for w in self.ode_model.parameters()]
        t0 = self.timestamps[-1]
        if output_gradients is None:
            output_gradients = torch.zeros_like(outputs)

        state = [t0, outputs, output_gradients, *grad_weights]

        for dt in self.time_intervals[::-1]:
            state = self.ode_solver(self.backward_dynamics, dt=dt, state=state)
        return state[1], state[2], state[3:]
