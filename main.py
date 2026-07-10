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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from estimators.hill_estimator import hill_estimator
from estimators.pickands_estimator import pickands_estimator
from estimators.gpd_mle_estimator import gpd_mle_estimator
from estimators.lmoment_estimator import lmoment_estimator
from estimators.pwm_estimator import pwm_estimator

from estimators.moment_estimator import moment_estimator
from estimators.reduced_bias_hill_estimator import reduced_bias_hill_estimator
from estimators.uh_estimator import uh_exponential_regression_estimator
from estimators.zhang_lme_estimator import zhang_lme_estimator
from estimators.mps_gpd_estimator import mps_gpd_estimator
from estimators.qq_estimator import qq_estimator
from estimators.mixed_kernel_estimator import mixed_moment_kernel_estimator
from estimators.k_selection import hill_k_sweep
from estimators.grimshaw_mle_estimator import grimshaw_mle_estimator

# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------

RANDOM_SEED = 123
BASE_SAMPLE_SIZE = 10_000
SUBSAMPLE_SIZE = 1_000
N_TRIALS = 50
N_BOOTSTRAPS = 500
K_TAIL = 100
LOWER_ENDPOINT = 0.0
OUTPUT_DIR = Path("outputs")

# Threshold-sensitivity diagnostics. The simulation uses fixed K_TAIL;
# the k-sweep is exported separately rather than using the old variance-only selector.
K_SWEEP_VALUES = [25, 50, 75, 100, 125, 150, 200, 250]

ESTIMATOR_NAMES = [
    "Hill",
    "Pickands",
    "GPD_MLE",
    "Grimshaw_MLE",
    "L_Moment",
    "PWM",
    "Moment_DEdH",
    "ReducedBias_Hill",
    "UH_ExpRegression",
    "Zhang_LME",
    "GPD_MPS",
    "QQ_KR",
    "MixedKernel",
]


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    true_alpha: float
    sampler: Callable[[np.random.Generator, int], np.ndarray]


DISTRIBUTIONS: List[DistributionSpec] = [
    DistributionSpec("Uniform(0,1)", 1.0, lambda rng, n: rng.uniform(0.0, 1.0, size=n)),
    DistributionSpec("Beta(2,2)",    2.0, lambda rng, n: rng.beta(2.0, 2.0, size=n)),
    DistributionSpec("Beta(2,5)",    2.0, lambda rng, n: rng.beta(2.0, 5.0, size=n)),
    DistributionSpec("Beta(3,10)",   3.0, lambda rng, n: rng.beta(3.0, 10.0, size=n)),
    DistributionSpec("Power(alpha=2)",  2.0, lambda rng, n: rng.random(size=n) ** (1.0 / 2.0)),
    DistributionSpec("Power(alpha=3)",  3.0, lambda rng, n: rng.random(size=n) ** (1.0 / 3.0)),
    DistributionSpec("Power(alpha=4)",  4.0, lambda rng, n: rng.random(size=n) ** (1.0 / 4.0)),
    DistributionSpec("Power(alpha=5)",  5.0, lambda rng, n: rng.random(size=n) ** (1.0 / 5.0)),
    DistributionSpec("Power(alpha=10)", 10.0, lambda rng, n: rng.random(size=n) ** (1.0 / 10.0)),
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
# Per-estimator wrappers
# ---------------------------------------------------------------------

def estimate_gammas(y: np.ndarray, k: int) -> Dict[str, float]:
    """Estimate Frechet-domain tail index gamma (~= 1/alpha) via all methods."""
    n = len(y)
    y_sorted = np.sort(y)
    gammas: Dict[str, float] = {}

    def put(name: str, fn):
        try:
            gammas[name] = safe_float(fn())
        except Exception:
            gammas[name] = np.nan

    put("Hill", lambda: hill_estimator(y, k))
    put("Pickands", lambda: pickands_estimator(y, k))
    put("Moment_DEdH", lambda: moment_estimator(y, k))
    put("ReducedBias_Hill", lambda: reduced_bias_hill_estimator(y, k))
    put("UH_ExpRegression", lambda: uh_exponential_regression_estimator(y, k))
    put("QQ_KR", lambda: qq_estimator(y, k))
    put("MixedKernel", lambda: mixed_moment_kernel_estimator(y, k))

    try:
        q = 1.0 - k / n
        result = gpd_mle_estimator(y, threshold_quantile=q)
        gammas["GPD_MLE"] = safe_float(result["xi"])
    except Exception:
        gammas["GPD_MLE"] = np.nan

    # Grimshaw profile-likelihood GPD estimator on the same top-k exceedances
    # is evaluated below after exceedances are constructed.

    exceed = None
    try:
        exceed = top_k_exceedances(y_sorted, k)
    except Exception:
        pass

    if exceed is None:
        for name in ["L_Moment", "PWM", "Zhang_LME", "GPD_MPS", "Grimshaw_MLE"]:
            gammas[name] = np.nan
    else:
        put("L_Moment", lambda: lmoment_estimator(exceed)[0])
        put("PWM", lambda: pwm_estimator(exceed)[0])
        put("Zhang_LME", lambda: zhang_lme_estimator(exceed)[0])
        put("GPD_MPS", lambda: mps_gpd_estimator(exceed)[0])
        put("Grimshaw_MLE", lambda: grimshaw_mle_estimator(exceed)[0])

    return {name: gammas.get(name, np.nan) for name in ESTIMATOR_NAMES}


def gamma_to_alpha(gamma_hat: float) -> float:
    """alpha = 1/gamma. Only valid for gamma > 0."""
    if not math.isfinite(gamma_hat) or gamma_hat <= 0:
        return np.nan
    return 1.0 / gamma_hat


# ---------------------------------------------------------------------
# One outer trial
# ---------------------------------------------------------------------

def run_one_trial(
    rng: np.random.Generator,
    dist: DistributionSpec,
    trial_id: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    base_sample = dist.sampler(rng, BASE_SAMPLE_SIZE)

    boot_records: List[dict] = []
    k_diag_records: List[dict] = []
    alpha_hats: Dict[str, List[float]] = {name: [] for name in ESTIMATOR_NAMES}

    for b in range(N_BOOTSTRAPS):
        subsample = rng.choice(base_sample, size=SUBSAMPLE_SIZE, replace=False)
        y = to_frechet_domain(subsample)

        k_used = K_TAIL

        gammas = estimate_gammas(y, k_used)

        # Hill-plot-style k sweep on the first bootstrap of each trial.
        if b == 0:
            for row in hill_k_sweep(y, K_SWEEP_VALUES):
                k_diag_records.append({
                    "distribution": dist.name,
                    "true_alpha": dist.true_alpha,
                    "trial_id": trial_id,
                    **row,
                })

        for name, gamma_hat in gammas.items():
            alpha_hat = gamma_to_alpha(gamma_hat)
            alpha_hats[name].append(alpha_hat)
            boot_records.append({
                "distribution": dist.name,
                "true_alpha": dist.true_alpha,
                "trial_id": trial_id,
                "bootstrap_id": b,
                "estimator": name,
                "k_used": k_used,
                "gamma_hat": gamma_hat,
                "alpha_hat": alpha_hat,
            })

    trial_rows: List[dict] = []
    for name in ESTIMATOR_NAMES:
        arr = np.array(alpha_hats[name], dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            trial_rows.append({
                "distribution": dist.name,
                "true_alpha": dist.true_alpha,
                "trial_id": trial_id,
                "estimator": name,
                "point_estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "ci_length": np.nan,
                "covered": np.nan,
                "n_valid": 0,
            })
            continue

        ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
        trial_rows.append({
            "distribution": dist.name,
            "true_alpha": dist.true_alpha,
            "trial_id": trial_id,
            "estimator": name,
            "point_estimate": float(np.median(valid)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_length": float(ci_high - ci_low),
            "covered": float(ci_low <= dist.true_alpha <= ci_high),
            "n_valid": int(len(valid)),
        })

    return boot_records, trial_rows, k_diag_records


def run_simulation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    all_boot_records: List[dict] = []
    all_trial_rows: List[dict] = []
    all_k_diag_records: List[dict] = []

    for dist in DISTRIBUTIONS:
        print(f"\nRunning {dist.name} | true alpha = {dist.true_alpha}")
        for t in range(N_TRIALS):
            boot_records, trial_rows, k_diag_records = run_one_trial(rng, dist, t)
            all_boot_records.extend(boot_records)
            all_trial_rows.extend(trial_rows)
            all_k_diag_records.extend(k_diag_records)
        print(f"Finished {dist.name}")

    bootstrap_df = pd.DataFrame.from_records(all_boot_records)
    trial_df = pd.DataFrame.from_records(all_trial_rows)
    k_diag_df = pd.DataFrame.from_records(all_k_diag_records)
    return bootstrap_df, trial_df, k_diag_df


# ---------------------------------------------------------------------
# Summary across trials
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
                distribution=dist_name,
                estimator=estimator,
                true_alpha=true_alpha,
                n_trials=n_trials,
                n_valid=0,
                failure_rate=failure_rate,
                mean_alpha_hat=np.nan,
                bias=np.nan,
                variance=np.nan,
                mse=np.nan,
                sd=np.nan,
                mean_ci_length=np.nan,
                coverage_rate=np.nan,
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
            distribution=dist_name,
            estimator=estimator,
            true_alpha=true_alpha,
            n_trials=n_trials,
            n_valid=n_valid,
            failure_rate=failure_rate,
            mean_alpha_hat=mean_hat,
            bias=bias,
            variance=variance,
            mse=mse,
            sd=sd,
            mean_ci_length=mean_ci_length,
            coverage_rate=coverage_rate,
        ))

    return pd.DataFrame(rows).sort_values(["distribution", "mse"])


def summarize_k_diagnostics(k_diag_df: pd.DataFrame) -> pd.DataFrame:
    if k_diag_df.empty:
        return pd.DataFrame()
    rows = []
    for (dist_name, k), g in k_diag_df.groupby(["distribution", "k"]):
        true_alpha = float(g["true_alpha"].iloc[0])
        alpha = g["alpha_hat"].dropna().to_numpy(dtype=float)
        rows.append({
            "distribution": dist_name,
            "k": int(k),
            "true_alpha": true_alpha,
            "mean_alpha_hat": float(np.mean(alpha)) if len(alpha) else np.nan,
            "sd_alpha_hat": float(np.std(alpha, ddof=1)) if len(alpha) > 1 else np.nan,
            "bias": float(np.mean(alpha) - true_alpha) if len(alpha) else np.nan,
            "n_valid": int(len(alpha)),
        })
    return pd.DataFrame(rows).sort_values(["distribution", "k"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    bootstrap_df, trial_df, k_diag_df = run_simulation()
    summary_df = summarize_results(trial_df)
    k_summary_df = summarize_k_diagnostics(k_diag_df)

    print(summary_df[["distribution", "estimator", "mean_alpha_hat", "bias", "mse"]])

    config_df = pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "base_sample_size": BASE_SAMPLE_SIZE,
        "subsample_size": SUBSAMPLE_SIZE,
        "n_trials": N_TRIALS,
        "n_bootstraps": N_BOOTSTRAPS,
        "k_tail": K_TAIL,
        "k_sweep_values": ",".join(map(str, K_SWEEP_VALUES)),
        "lower_endpoint": LOWER_ENDPOINT,
    }])

    excel_path = OUTPUT_DIR / "alpha_estimator_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        trial_df.to_excel(writer, sheet_name="TrialLevel", index=False)
        k_summary_df.to_excel(writer, sheet_name="KDiagnostics", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

    bootstrap_path = OUTPUT_DIR / "alpha_estimates_bootstrap_raw.csv"
    k_diag_path = OUTPUT_DIR / "k_diagnostics_raw.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    k_diag_df.to_csv(k_diag_path, index=False)

    print("\nSaved files:")
    print(f"  {excel_path}  (Summary, TrialLevel, KDiagnostics, Config sheets)")
    print(f"  {bootstrap_path}  (every individual bootstrap draw, for auditing)")
    print(f"  {k_diag_path}  (Hill-plot-style k-sweep diagnostics)")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
