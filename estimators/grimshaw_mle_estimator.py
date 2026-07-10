import numpy as np
from scipy.optimize import brentq


def grimshaw_mle_estimator(exceedances):
    """
    Numerically stable GPD MLE via Grimshaw's (1993) one-dimensional profile
    reduction. Returns (xi, beta).

    NOTE ON PROVENANCE: this file replaces an earlier "zhang_lme_estimator.py"
    that was mislabeled -- it ran a generic penalized 2-D optimizer, not
    Zhang (2007)'s actual likelihood-moment estimating equation. I was not
    able to retrieve Zhang's exact equations from a primary source (the
    journal is paywalled and my fetch attempts came back empty), so rather
    than guess at his specific formula again, this implements something I
    *can* verify directly: reparametrizing the GPD log-likelihood by
    theta = xi/beta reduces the two-parameter MLE to a single scalar
    equation h(theta) = 0, solvable by bracketed root-finding instead of a
    generic 2-D optimizer. This directly targets the same problem Zhang's
    paper cites as motivation (MLE's convergence issues) using a method
    I derived and checked algebraically, and confirmed numerically matches
    scipy.stats.genpareto.fit to ~5 decimal places on simulated GPD data.

    If you specifically need Zhang (2007)'s exact likelihood-moment
    equation (which generalizes this with a tuning constant r), you should
    verify it against the original paper -- Zhang, J. (2007), "Likelihood
    moment estimation for the generalized Pareto distribution",
    Aust. N. Z. J. Stat. 49(1), 69-77 -- rather than trust a reconstruction.

    Derivation summary:
      Let theta = xi/beta. The GPD log-likelihood in terms of (xi, theta) is
        l(xi, theta) = -n*log(xi) + n*log(theta) - (1+1/xi) * sum(log(1+theta*x_i))
      Setting dl/dxi = 0 gives:
        xi_hat(theta) = (1/n) * sum(log(1 + theta*x_i))                 =: u(theta)
      Setting dl/dbeta = 0 (the other original score equation, rewritten in
      terms of theta) and substituting u(theta) for xi gives:
        v(theta) * (u(theta) + 1) = 1,   where v(theta) = (1/n)*sum(1/(1+theta*x_i))
      so the MLE theta_hat is the root of:
        h(theta) = u(theta)*v(theta) + v(theta) - 1 = 0
      Note theta=0 is always a trivial root of h (corresponding to xi=0);
      the genuine MLE root is sought away from theta=0.
    """
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    n = len(x)
    if n < 5 or np.max(x) <= 0:
        raise ValueError("Need at least 5 nonnegative, nonconstant exceedances")

    mean_x = float(np.mean(x))

    def u(theta):
        return float(np.mean(np.log1p(theta * x)))

    def v(theta):
        return float(np.mean(1.0 / (1.0 + theta * x)))

    def h(theta):
        vv = v(theta)
        return u(theta) * vv + vv - 1.0

    # Search theta > 0 (corresponds to xi > 0, the case expected once
    # lower-tail Weibull-domain data has been reflected into the Frechet
    # domain by main.py's Y = 1/(X - lower_endpoint) transform). theta < 0
    # would need 1 + theta*x_i > 0 for all i (a finite-endpoint / xi<0 fit)
    # and is out of scope for this pipeline's use case.
    grid = np.geomspace(1e-6 / mean_x, 1e3 / mean_x, 400)
    vals = np.array([h(t) for t in grid])

    root = None
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            try:
                root = brentq(h, grid[i], grid[i + 1], xtol=1e-12, maxiter=200)
                break
            except Exception:
                continue

    if root is None or root <= 0:
        return np.nan, np.nan

    xi_hat = u(root)
    if not np.isfinite(xi_hat) or xi_hat <= 0:
        return np.nan, np.nan

    beta_hat = xi_hat / root
    if not np.isfinite(beta_hat) or beta_hat <= 0:
        return np.nan, np.nan

    return float(xi_hat), float(beta_hat)