import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from hill_estimator import hill_estimator
from pickands_estimator import pickands_estimator
from gpd_mle_estimator import gpd_mle_estimator
from pwm_estimator import pwm_estimator
from lmoment_estimator import lmoment_estimator

CSV_FILE = "phi_distributions_size7_T50(in).csv"
df = pd.read_csv(CSV_FILE)

threshold_quantile = 0.90
results = []

for col in df.columns:
    losses = df[col].dropna().to_numpy(dtype=float)
    losses = losses[np.isfinite(losses)]

    nonpositive_count = np.sum(losses <= 0)
    losses = losses[losses > 0]

    if len(losses) < 20:
        continue

    n = len(losses)
    k = max(2, int(np.ceil((1 - threshold_quantile) * n)))

    try:
        xi_hill = hill_estimator(losses, k=k)
    except Exception:
        xi_hill = np.nan

    
    k_pickands = max(1, min(k, (n - 1) // 4))
    try:
        xi_pickands = pickands_estimator(losses, k=k_pickands)
    except Exception:
        xi_pickands = np.nan

    try:
        gpd = gpd_mle_estimator(losses, threshold_quantile=threshold_quantile)

        u = gpd["threshold"]
        xi = gpd["xi"]
        beta = gpd["beta"]
        exceedances = gpd["exceedances"]

        try:
            xi_pwm, beta_pwm = pwm_estimator(exceedances)
        except Exception:
            xi_pwm, beta_pwm = np.nan, np.nan

        try:
            xi_lmom, beta_lmom = lmoment_estimator(exceedances)
        except Exception:
            xi_lmom, beta_lmom = np.nan, np.nan

        if xi < 0:
            endpoint = u - beta / xi
        else:
            endpoint = np.inf

        results.append({
            "column": col,
            "n": n,
            "nonpositive_removed": nonpositive_count,
            "k_used": k,
            "k_pickands_used": k_pickands,
            "threshold": u,
            "exceedances": len(exceedances),
            "observed_max": losses.max(),
            "hill_xi": xi_hill,
            "pickands_xi": xi_pickands,
            "gpd_xi": xi,
            "gpd_beta": beta,
            "pwm_xi": xi_pwm,
            "pwm_beta": beta_pwm,
            "lmoment_xi": xi_lmom,
            "lmoment_beta": beta_lmom,
            "endpoint": endpoint,
            "gap": endpoint - losses.max() if np.isfinite(endpoint) else np.inf
        })

    except Exception as e:
        print(col, "failed:", e)

results_df = pd.DataFrame(results)
results_df.to_csv("csv/evt_results_by_distribution.csv", index=False)

print(results_df)

pooled_losses = df.to_numpy().flatten()
pooled_losses = pooled_losses[np.isfinite(pooled_losses)]
pooled_losses = pooled_losses[pooled_losses > 0]

gpd = gpd_mle_estimator(pooled_losses, threshold_quantile=threshold_quantile)

u = gpd["threshold"]
xi = gpd["xi"]
beta = gpd["beta"]
exceedances = gpd["exceedances"]


if xi < 0:
    upper_endpoint = u - beta / xi
else:
    upper_endpoint = np.inf

y = np.sort(exceedances)

empirical_survival = 1 - np.arange(1, len(y) + 1) / (len(y) + 1)

if xi < 0:
    y_max_plot = (-beta / xi) * 0.999
else:
    y_max_plot = max(y.max() * 3, y.max() + 3 * beta)

y_grid = np.linspace(0, y_max_plot, 500)
fitted_survival = genpareto.sf(y_grid, c=xi, loc=0, scale=beta)

plt.figure(figsize=(8, 6))
plt.plot(y, empirical_survival, marker="o", linestyle="", label="Empirical Tail")
plt.plot(y_grid, fitted_survival, linewidth=2, label="GPD Tail Fit")
plt.axvline(y.max(), linestyle="--", label="Largest Observed Exceedance")
plt.yscale("log")
plt.xlabel("Exceedance Above Threshold")
plt.ylabel("Survival Probability")
plt.title("Empirical Tail vs Extrapolated GPD Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("graphs/empirical_tail_vs_extrapolated_gpd_fit.png", dpi=300, bbox_inches="tight")
plt.close()

x_grid = u + y_grid
unconditional_tail_probability = np.mean(pooled_losses > u) * fitted_survival

plt.figure(figsize=(8, 6))
plt.plot(x_grid, unconditional_tail_probability, linewidth=2)
plt.axvline(pooled_losses.max(), linestyle="--", label="Observed Maximum")
if np.isfinite(upper_endpoint):
    plt.axvline(upper_endpoint, linestyle="-", label="Estimated Upper Endpoint")
plt.yscale("log")
plt.xlabel("Information Loss Level")
plt.ylabel("Estimated P(Loss > x)")
plt.title("Extrapolated Extreme Tail Probabilities")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("graphs/extrapolated_extreme_tail_probabilities.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(pooled_losses, bins=50, density=True, alpha=0.6)
plt.axvline(u, linestyle="--", label=f"Threshold u = {u:.3f}")
plt.xlabel("Information Loss")
plt.ylabel("Density")
plt.title("Pooled Information Loss Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("graphs/pooled_information_loss_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(results_df["column"], results_df["hill_xi"], marker="o", label="Hill xi")
plt.plot(results_df["column"], results_df["pickands_xi"], marker="o", label="Pickands xi")
plt.plot(results_df["column"], results_df["gpd_xi"], marker="o", label="GPD xi")
plt.plot(results_df["column"], results_df["pwm_xi"], marker="o", label="PWM xi")
plt.plot(results_df["column"], results_df["lmoment_xi"], marker="o", label="L-moment xi")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=90)
plt.xlabel("Distribution")
plt.ylabel("Tail Index xi")
plt.title("Tail Index Estimates by Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("graphs/tail_index_by_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

thresholds = [0.80, 0.85, 0.90, 0.925, 0.95]

stability_rows = []

for q in thresholds:
    try:
        gpd_q = gpd_mle_estimator(pooled_losses, threshold_quantile=q)

        u_q = gpd_q["threshold"]
        xi_q = gpd_q["xi"]
        beta_q = gpd_q["beta"]

        if xi_q < 0:
            endpoint_q = u_q - beta_q / xi_q
        else:
            endpoint_q = np.inf

        stability_rows.append({
            "threshold_quantile": q,
            "threshold": u_q,
            "exceedances": len(gpd_q["exceedances"]),
            "xi": xi_q,
            "beta": beta_q,
            "endpoint": endpoint_q,
            "gap": endpoint_q - pooled_losses.max() if np.isfinite(endpoint_q) else np.inf
        })

    except Exception:
        pass

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv("csv/threshold_stability_results.csv", index=False)

plt.figure(figsize=(8, 6))
plt.plot(stability_df["threshold_quantile"], stability_df["xi"], marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Threshold Quantile")
plt.ylabel("GPD xi")
plt.title("Threshold Stability Plot")
plt.grid(alpha=0.3)
plt.savefig("graphs/threshold_stability_xi.png", dpi=300, bbox_inches="tight")
plt.close()