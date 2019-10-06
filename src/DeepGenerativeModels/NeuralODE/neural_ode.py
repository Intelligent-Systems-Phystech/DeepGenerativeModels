from torch import nn


def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]


def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)


class NeuralODE(nn.Module):
    def __init__(self, timestamps, ode_solver, ode_model):
        super(NeuralODE, self).__init__()
        self.timestamps = timestamps
        self.ode_model = ode_model
        self.time_intervals = timestamps[1:] - timestamps[:-1]
        self.ode_solver = ode_solver

    def forward_dynamics(self, state):
        return [1.0, self.ode_model(inputs=state)]

    def forward(self, inputs):
        states = []
        states = [inputs]
        state = [self.timestamps[0], inputs]
        for dt in self.time_intervals:
            state = self.ode_solver(func=self.forward_dynamics, dt=dt, state=state)
            states.append(state[1])
        outputs = state[1]
        return outputs, states
