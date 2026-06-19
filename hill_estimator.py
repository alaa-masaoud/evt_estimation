import numpy as np


def hill_estimator(data, k):
    

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    x = np.sort(x)

    n = len(x)

    if k <= 1 or k >= n:
        raise ValueError("k must satisfy 2 <= k < n")

    threshold = x[-k - 1]
    top_k = x[-k:]

    xi = np.mean(np.log(top_k) - np.log(threshold))

    return xi