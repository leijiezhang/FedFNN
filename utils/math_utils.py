import numpy as np


def mapminmax(x, l_range=-1, u_range=1):
    xmax = np.expand_dims(x.max(axis=0), axis=0)
    xmin = np.expand_dims(x.min(axis=0), axis=0)
    xmin = np.repeat(xmin, x.shape[0], axis=0)
    xmax = np.repeat(xmax, x.shape[0], axis=0)

    if (xmax == xmin).any():
        raise ValueError("some rows have no variation")
    x_proj = ((u_range - l_range) * (x - xmin) / (xmax - xmin)) + l_range

    return x_proj
