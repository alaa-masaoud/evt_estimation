import numpy as np


def hill_for_k(data, k):
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)
    n = len(y)
    if k <= 0 or k >= n:
        return np.nan
    threshold = y[-k - 1]
    if threshold <= 0:
        return np.nan
    out = float(np.mean(np.log(y[-k:]) - np.log(threshold)))
    return out if np.isfinite(out) and out > 0 else np.nan


def hill_k_sweep(data, k_values):
    """
    Return gamma_hat and alpha_hat for a sweep of k values.

    Use this to eyeball a stable ("plateau") region of the Hill plot and
    pick k by inspection, rather than relying on an automatic selector.
    """
    rows = []
    for k in k_values:
        gamma = hill_for_k(data, int(k))
        alpha = 1.0 / gamma if np.isfinite(gamma) and gamma > 0 else np.nan
        rows.append({"k": int(k), "gamma_hat": gamma, "alpha_hat": alpha})
    return rows


def auto_k_selector(*args, **kwargs):
    """
    OPEN ITEM -- not implemented.

    Three attempts at an automatic bias-variance (AMSE-style) k selector
    were built and checked against a Monte Carlo oracle on simulated data
    (true gamma known):

      1. Bootstrap-variance-only scoring: picked k pinned near the top of
         whatever grid it was given, because it had no bias term to
         counterbalance variance (which keeps shrinking as k grows).
      2. Bootstrap-variance + a bootstrap-mean-vs-full-sample bias proxy:
         no better -- the bias proxy turned out to be dominated by
         resampling noise, not real bias; it wasn't even monotonic in k
         the way genuine Hill bias should be.
      3. A two-scale double-bootstrap reconstruction (in the spirit of
         Danielsson, de Haan, Peng & de Vries, 2001): substantially worse
         (picked k roughly 3.5x the oracle answer), most likely because it
         bootstrapped WITH replacement at a fixed subsample size, which
         doesn't preserve the scaling relationship the method depends on,
         and because the exact subsampling scheme / rate constants from
         the original paper weren't available to verify against.

    None of these are shipped because none of them survived being checked
    against ground truth. Use hill_k_sweep(data, k_values) instead and pick
    k by inspecting where the Hill plot is stable (the standard manual
    approach), or implement genuine Danielsson et al. (2001) double
    bootstrap with direct access to the original paper's algorithm rather
    than a from-memory reconstruction.
    """
    raise NotImplementedError(
        "No automatic k-selector here passed validation against a Monte "
        "Carlo oracle. Use hill_k_sweep() and pick k by inspection, or "
        "implement Danielsson et al. (2001) with the original paper in hand."
    )