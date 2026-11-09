import numpy as np
from scipy.stats import qmc


def initial_points(n, dim, seed):
    """Scrambled Sobol design in [-5,5]^dim, identical for every method given the seed."""
    u = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)
    return -5.0 + 10.0 * u
