import numpy as np


def lmoment_estimator(exceedances):
  

    y = np.sort(np.asarray(exceedances))

    n = len(y)

    if n < 3:
        raise ValueError("Need at least 3 exceedances")

    b0 = np.mean(y)

    weights = np.array([i / (n - 1) for i in range(n)])

    b1 = np.mean(weights * y)

    l1 = b0
    l2 = 2 * b1 - b0

    tau = l2 / l1

    xi = (2 * tau - 1) / tau
    beta = l1 * (1 - xi)

    return xi, beta