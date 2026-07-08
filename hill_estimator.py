import numpy as np
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from hill_estimator import hill_estimator


def weibull_hill_estimator(data, k, right_endpoint=None):

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = len(x)

    if right_endpoint is None:
        x_F = _estimate_endpoint(x)
    else:
        x_F = right_endpoint

    if np.any(x >= x_F):
        raise ValueError()

    y = 1.0 / (x_F - x)          
    gamma_hat = hill_estimator(y, k)   
    xi_hat = -gamma_hat

    return xi_hat, x_F


def _estimate_endpoint(x_sorted, gap=1):
 
    n = len(x_sorted)
    return x_sorted[-1] + (x_sorted[-1] - x_sorted[-1 - gap])