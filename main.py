"""
main.py

Estimator-comparison simulation for lower-tail Weibull alpha.

Workflow:
1. Pick a distribution f(x).
2. Draw a fresh base sample of size BASE_SAMPLE_SIZE directly from f(x).
3. Repeatedly draw subsamples of SUBSAMPLE_SIZE from that base sample, without
   replacement, to build a percentile CI for alpha_hat (the inner resampling loop).
4. Check whether the true alpha falls inside that CI.
5. Repeat steps 2-4 across N_TRIALS independent trials (the outer Monte Carlo
   loop) so that "coverage" is an actual rate, not a single trial's 0/1 outcome.
6. Summarize bias, variance, MSE, mean CI length, and coverage rate per
   distribution x estimator, and export everything to Excel.

Expected estimator files in same folder:
    hill_estimator.py
    pickands_estimator.py
    gpd_mle_estimator.py
    lmoment_estimator.py
    pwm_estimator.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from hill_estimator import hill_estimator
from pickands_estimator import pickands_estimator
from gpd_mle_estimator import gpd_mle_estimator
from lmoment_estimator import lmoment_estimator
from pwm_estimator import pwm_estimator

# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------

RANDOM_SEED = 123
BASE_SAMPLE_SIZE = 10_000
SUBSAMPLE_SIZE = 1_000
N_TRIALS = 50           # outer Monte Carlo replications (fresh base sample each)
N_BOOTSTRAPS = 500      # inner resamples per trial, used to build that trial's CI
K_TAIL = 100            # upper order statistics used by every estimator (10% of 1,000)
LOWER_ENDPOINT = 0.0    # true lower endpoint of X, shared by all four distributions
OUTPUT_DIR = Path("outputs")

ESTIMATOR_NAMES = ["Hill", "Pickands", "GPD_MLE", "L_Moment", "PWM"]


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    true_alpha: float
    sampler: Callable[[np.random.Generator, int], np.ndarray]


# true_alpha = first shape parameter `a` of Beta(a,b): near the lower endpoint 0,
# a Beta(a,b) density behaves like C * x^(a-1), so F(x) ~ x^a and alpha = a for
# the minimum (lower-tail Weibull domain). Beta(2,2) and Beta(2,5) therefore
# share alpha = 2 -- they only differ in their *upper* tail (governed by b),
# which is irrelevant here since we're estimating the lower tail.
DISTRIBUTIONS: List[DistributionSpec] = [
    DistributionSpec("Uniform(0,1)", 1.0, lambda rng, n: rng.uniform(0.0, 1.0, size=n)),
    DistributionSpec("Beta(2,2)",    2.0, lambda rng, n: rng.beta(2.0, 2.0, size=n)),
    DistributionSpec("Beta(2,5)",    2.0, lambda rng, n: rng.beta(2.0, 5.0, size=n)),
    DistributionSpec("Beta(3,10)",   3.0, lambda rng, n: rng.beta(3.0, 10.0, size=n)),
]


# ---------------------------------------------------------------------
# Reflection transform: lower-tail Weibull domain -> upper-tail Frechet domain
# ---------------------------------------------------------------------

def to_frechet_domain(x: np.ndarray, lower_endpoint: float = LOWER_ENDPOINT) -> np.ndarray:
    """Y = 1 / (X - lower_endpoint). Small X near the endpoint -> large Y."""
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    return 1.0 / np.maximum(x - lower_endpoint, eps)


def top_k_exceedances(y_sorted: np.ndarray, k: int) -> np.ndarray:
    """Top-k values over the (k+1)-th largest, for POT-style estimators."""
    threshold = y_sorted[-k - 1]
    return y_sorted[-k:] - threshold


def safe_float(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Per-estimator wrappers (each estimator has a different call signature and
# return type -- this replaces the old one-size-fits-all call_estimator)
# ---------------------------------------------------------------------

def estimate_gammas(y: np.ndarray, k: int) -> Dict[str, float]:
    """Estimate the Frechet-domain tail index gamma (~= 1/alpha) via all five methods."""
    n = len(y)
    y_sorted = np.sort(y)
    gammas: Dict[str, float] = {}

    try:
        gammas["Hill"] = safe_float(hill_estimator(y, k))
    except Exception:
        gammas["Hill"] = np.nan

    try:
        gammas["Pickands"] = safe_float(pickands_estimator(y, k))
    except Exception:
        gammas["Pickands"] = np.nan

    try:
        q = 1.0 - k / n
        result = gpd_mle_estimator(y, threshold_quantile=q)
        gammas["GPD_MLE"] = safe_float(result["xi"])
    except Exception:
        gammas["GPD_MLE"] = np.nan

    try:
        exceed = top_k_exceedances(y_sorted, k)
        xi, _beta = lmoment_estimator(exceed)
        gammas["L_Moment"] = safe_float(xi)
    except Exception:
        gammas["L_Moment"] = np.nan

    try:
        exceed = top_k_exceedances(y_sorted, k)
        xi, _beta = pwm_estimator(exceed)
        gammas["PWM"] = safe_float(xi)
    except Exception:
        gammas["PWM"] = np.nan

    return gammas


def gamma_to_alpha(gamma_hat: float) -> float:
    """alpha = 1/gamma. Only valid for gamma > 0 (a genuine Frechet-domain estimate)."""
    if not math.isfinite(gamma_hat) or gamma_hat <= 0:
        return np.nan
    return 1.0 / gamma_hat


# ---------------------------------------------------------------------
# One outer trial: fresh base sample, N_BOOTSTRAPS subsample estimates per
# estimator, a percentile CI built from those, and a coverage check.
# ---------------------------------------------------------------------

def run_one_trial(rng: np.random.Generator, dist: DistributionSpec, trial_id: int
                   ) -> Tuple[List[dict], List[dict]]:
    base_sample = dist.sampler(rng, BASE_SAMPLE_SIZE)

    boot_records: List[dict] = []
    alpha_hats: Dict[str, List[float]] = {name: [] for name in ESTIMATOR_NAMES}

    for b in range(N_BOOTSTRAPS):
        subsample = rng.choice(base_sample, size=SUBSAMPLE_SIZE, replace=False)
        y = to_frechet_domain(subsample)
        gammas = estimate_gammas(y, K_TAIL)

        for name, gamma_hat in gammas.items():
            alpha_hat = gamma_to_alpha(gamma_hat)
            alpha_hats[name].append(alpha_hat)
            boot_records.append({
                "distribution": dist.name,
                "true_alpha": dist.true_alpha,
                "trial_id": trial_id,
                "bootstrap_id": b,
                "estimator": name,
                "gamma_hat": gamma_hat,
                "alpha_hat": alpha_hat,
            })

    trial_rows: List[dict] = []
    for name in ESTIMATOR_NAMES:
        arr = np.array(alpha_hats[name], dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            trial_rows.append({
                "distribution": dist.name, "true_alpha": dist.true_alpha,
                "trial_id": trial_id, "estimator": name,
                "point_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_length": np.nan, "covered": np.nan, "n_valid": 0,
            })
            continue
        ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
        trial_rows.append({
            "distribution": dist.name, "true_alpha": dist.true_alpha,
            "trial_id": trial_id, "estimator": name,
            "point_estimate": float(np.median(valid)),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "ci_length": float(ci_high - ci_low),
            "covered": float(ci_low <= dist.true_alpha <= ci_high),
            "n_valid": int(len(valid)),
        })

    return boot_records, trial_rows


def run_simulation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    all_boot_records: List[dict] = []
    all_trial_rows: List[dict] = []

    for dist in DISTRIBUTIONS:
        print(f"\nRunning {dist.name} | true alpha = {dist.true_alpha}")
        for t in range(N_TRIALS):
            boot_records, trial_rows = run_one_trial(rng, dist, t)
            all_boot_records.extend(boot_records)
            all_trial_rows.extend(trial_rows)
        print(f"Finished {dist.name}")

    bootstrap_df = pd.DataFrame.from_records(all_boot_records)
    trial_df = pd.DataFrame.from_records(all_trial_rows)
    return bootstrap_df, trial_df


# ---------------------------------------------------------------------
# Summary across trials: bias, variance, MSE, mean CI length, coverage rate
# ---------------------------------------------------------------------

def summarize_results(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dist_name, estimator), g in trial_df.groupby(["distribution", "estimator"]):
        true_alpha = float(g["true_alpha"].iloc[0])
        pts = g["point_estimate"].dropna().to_numpy(dtype=float)
        n_trials = len(g)
        n_valid = len(pts)
        failure_rate = 1.0 - n_valid / n_trials if n_trials else np.nan

        if n_valid == 0:
            rows.append(dict(
                distribution=dist_name, estimator=estimator, true_alpha=true_alpha,
                n_trials=n_trials, n_valid=0, failure_rate=failure_rate,
                mean_alpha_hat=np.nan, bias=np.nan, variance=np.nan, mse=np.nan,
                sd=np.nan, mean_ci_length=np.nan, coverage_rate=np.nan,
            ))
            continue

        mean_hat = float(np.mean(pts))
        bias = mean_hat - true_alpha
        variance = float(np.var(pts, ddof=1)) if n_valid > 1 else np.nan
        mse = float(np.mean((pts - true_alpha) ** 2))
        sd = float(np.std(pts, ddof=1)) if n_valid > 1 else np.nan
        mean_ci_length = float(g["ci_length"].dropna().mean())
        coverage_rate = float(g["covered"].dropna().mean())

        rows.append(dict(
            distribution=dist_name, estimator=estimator, true_alpha=true_alpha,
            n_trials=n_trials, n_valid=n_valid, failure_rate=failure_rate,
            mean_alpha_hat=mean_hat, bias=bias, variance=variance, mse=mse, sd=sd,
            mean_ci_length=mean_ci_length, coverage_rate=coverage_rate,
        ))

    return pd.DataFrame(rows).sort_values(["distribution", "mse"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    bootstrap_df, trial_df = run_simulation()
    summary_df = summarize_results(trial_df)
    print(summary_df[['distribution','estimator','mean_alpha_hat','bias','mse']])
    config_df = pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "base_sample_size": BASE_SAMPLE_SIZE,
        "subsample_size": SUBSAMPLE_SIZE,
        "n_trials": N_TRIALS,
        "n_bootstraps": N_BOOTSTRAPS,
        "k_tail": K_TAIL,
        "lower_endpoint": LOWER_ENDPOINT,
    }])

    excel_path = OUTPUT_DIR / "alpha_estimator_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        trial_df.to_excel(writer, sheet_name="TrialLevel", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

    bootstrap_path = OUTPUT_DIR / "alpha_estimates_bootstrap_raw.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    print("\nSaved files:")
    print(f"  {excel_path}  (Summary, TrialLevel, Config sheets)")
    print(f"  {bootstrap_path}  (every individual bootstrap draw, for auditing)")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()