import numpy as np


def pwm_estimator(exceedances):
   

    y = np.sort(np.asarray(exceedances))

    n = len(y)

    if n < 3:
        raise ValueError("Need at least 3 exceedances")

    b0 = np.mean(y)

    weights = np.array([i / (n - 1) for i in range(n)])

    b1 = np.mean(weights * y)

    xi = (4 * b1 - 3 * b0) / (2 * b1 - b0)
    beta = b0 * (1 - xi)

    return xi, beta