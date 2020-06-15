import numpy as np


def gaussian_tuning(s, s_a, sigma_a, r_max=1):
    return r_max * np.exp(-.5 * ((s - s_a) / sigma_a) ** 2)
