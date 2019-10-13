def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]


def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)
