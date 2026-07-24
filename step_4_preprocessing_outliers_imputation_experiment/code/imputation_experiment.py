"""
imputation_experiment.py — MICE robustness (multi-iteration).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

On reviewing the imputation
literature we found that single-draw validation is not accepted as evidence
of stability: a lone lucky/unlucky random mask says nothing about whether the
method's advantage is reproducible. Standard practice is to repeat the masking
under many independent draws and report the resulting distribution. This module
therefore repeats the experiment N_ITERATIONS times, each with an independent mask
(different seed, same 8% fraction, same predictor set), and reports the
distribution (mean +/- std, min/max) of R^2/RMSE/MAE across runs for both MICE
and the median baseline — establishing that the result generalises rather than
being a one-off draw.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

sns.set_theme(style="whitegrid", context="talk")
NAVY, TEAL, CORAL, GOLD = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a"


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _run_once(X: pd.DataFrame, target: str, mask_fraction: float,
              mice_max_iter: int, seed: int) -> dict[str, Any]:
    """One masked-reconstruction trial with a given random seed."""
    truth = X[target].to_numpy().copy()
    rng = np.random.default_rng(seed)
    n = len(X)
    n_mask = int(round(n * mask_fraction))
    mask_idx = rng.choice(n, size=n_mask, replace=False)
    mask_bool = np.zeros(n, dtype=bool)
    mask_bool[mask_idx] = True

    X_masked = X.copy()
    X_masked.loc[mask_bool, target] = np.nan
    true_masked = truth[mask_bool]

    mice = IterativeImputer(estimator=BayesianRidge(), max_iter=mice_max_iter,
                            random_state=seed, sample_posterior=False)
    mice_full = mice.fit_transform(X_masked)
    mice_pred = mice_full[:, X.columns.get_loc(target)][mask_bool]

    med = SimpleImputer(strategy="median")
    median_col = med.fit_transform(X_masked[[target]]).ravel()
    median_pred = median_col[mask_bool]

    return {"seed": seed, "n_masked": n_mask,
            "mice": _metrics(mice_pred, true_masked),
            "median": _metrics(median_pred, true_masked)}


def run_multi_iteration(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the masking experiment N_ITERATIONS times with independent seeds."""
    ie = cfg["imputation_experiment"]
    target = ie["target_feature"]
    predictors = ie["predictors"]
    n_iter = ie["n_iterations"]
    base_seed = cfg["seed"]

    data = df[df[target].notna()].copy()
    X = data[predictors].astype(float).reset_index(drop=True)

    rows = []
    for i in range(n_iter):
        seed = base_seed + i
        res = _run_once(X, target, ie["mask_fraction"], ie["mice_max_iter"], seed)
        rows.append({"run": i, "seed": seed,
                    "MICE_R2": res["mice"]["R2"], "MICE_RMSE": res["mice"]["RMSE"],
                    "MICE_MAE": res["mice"]["MAE"],
                    "Median_R2": res["median"]["R2"], "Median_RMSE": res["median"]["RMSE"],
                    "Median_MAE": res["median"]["MAE"]})

    runs = pd.DataFrame(rows)
    summary = {
        "n_iterations": n_iter, "n_total": len(X),
        "n_masked": int(round(len(X) * ie["mask_fraction"])),
        "target": target, "predictors": predictors,
        "MICE_R2_mean": float(runs["MICE_R2"].mean()), "MICE_R2_std": float(runs["MICE_R2"].std()),
        "MICE_R2_min": float(runs["MICE_R2"].min()), "MICE_R2_max": float(runs["MICE_R2"].max()),
        "MICE_RMSE_mean": float(runs["MICE_RMSE"].mean()), "MICE_RMSE_std": float(runs["MICE_RMSE"].std()),
        "Median_R2_mean": float(runs["Median_R2"].mean()), "Median_R2_std": float(runs["Median_R2"].std()),
        "Median_RMSE_mean": float(runs["Median_RMSE"].mean()), "Median_RMSE_std": float(runs["Median_RMSE"].std()),
    }
    return {"runs": runs, "summary": summary}


def plot_robustness(result: dict[str, Any], out_dir: Path) -> Path:
    """Boxplot of R^2 across all iterations, MICE vs median — the stability proof."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, summ = result["runs"], result["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: R^2 distribution across runs (the stability evidence).
    box_data = [runs["MICE_R2"], runs["Median_R2"]]
    bp = ax1.boxplot(box_data, labels=["MICE", "Median"], patch_artist=True,
                     widths=0.5)
    for patch, c in zip(bp["boxes"], [TEAL, CORAL]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, data in enumerate(box_data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(data))
        ax1.scatter(np.full(len(data), i) + jitter, data, s=18, color="black",
                   alpha=0.5, zorder=3)
    ax1.set_title(f"R² across {summ['n_iterations']} independent runs\n"
                 f"MICE {summ['MICE_R2_mean']:.3f} ± {summ['MICE_R2_std']:.3f}  vs  "
                 f"Median {summ['Median_R2_mean']:.3f} ± {summ['Median_R2_std']:.3f}",
                 fontsize=13)
    ax1.set_ylabel("R² (reconstruction of masked cells)")

    # Right: per-run R^2 trend line — visually shows no drift/instability.
    ax2.plot(runs["run"], runs["MICE_R2"], "o-", color=TEAL, label="MICE", lw=1.5)
    ax2.plot(runs["run"], runs["Median_R2"], "o-", color=CORAL, label="Median", lw=1.5)
    ax2.axhline(summ["MICE_R2_mean"], color=TEAL, ls="--", lw=1, alpha=0.6)
    ax2.set_title("R² per run (different random 8% mask each time)")
    ax2.set_xlabel("run index"); ax2.set_ylabel("R²")
    ax2.legend(fontsize=11)

    fig.suptitle(f"MICE Robustness: {summ['n_iterations']} independent masking trials "
                f"({int(summ['n_masked'])} cells masked each run, 8%)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "mice_robustness_multi_iteration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reconstruction_scatter(df: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> Path:
    """Reconstructed-vs-true scatter for one representative masking run.

    Complements the multi-iteration robustness view by showing *why* MICE wins:
    MICE points track the diagonal (perfect reconstruction), while the median
    baseline collapses to a flat horizontal line because it returns one value for
    every masked cell, ignoring the feature structure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ie = cfg["imputation_experiment"]
    target = ie["target_feature"]
    seed = cfg["seed"]

    data = df[df[target].notna()].copy()
    X = data[ie["predictors"]].astype(float).reset_index(drop=True)
    truth = X[target].to_numpy().copy()
    rng = np.random.default_rng(seed)
    n = len(X)
    n_mask = int(round(n * ie["mask_fraction"]))
    mask_bool = np.zeros(n, dtype=bool)
    mask_bool[rng.choice(n, size=n_mask, replace=False)] = True
    X_masked = X.copy()
    X_masked.loc[mask_bool, target] = np.nan
    true_masked = truth[mask_bool]

    mice = IterativeImputer(estimator=BayesianRidge(), max_iter=ie["mice_max_iter"],
                            random_state=seed, sample_posterior=False)
    mice_pred = mice.fit_transform(X_masked)[:, X.columns.get_loc(target)][mask_bool]
    median_pred = SimpleImputer(strategy="median").fit_transform(
        X_masked[[target]]).ravel()[mask_bool]

    mice_r2 = _metrics(mice_pred, true_masked)["R2"]
    median_r2 = _metrics(median_pred, true_masked)["R2"]

    lo = float(min(true_masked.min(), mice_pred.min()))
    hi = float(max(true_masked.max(), mice_pred.max()))
    pad = 0.05 * (hi - lo)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=NAVY,
            lw=1.5, zorder=1, label="perfect reconstruction")
    median_lbl = ("$R^2 \\approx 0$" if abs(median_r2) < 0.01
                  else f"$R^2 = {median_r2:.2f}$")
    ax.scatter(true_masked, median_pred, s=22, color=CORAL, alpha=0.55,
               edgecolor="none", zorder=2, label=f"Median fill  ({median_lbl})")
    ax.scatter(true_masked, mice_pred, s=22, color=TEAL, alpha=0.6,
               edgecolor="none", zorder=3, label=f"MICE  ($R^2 = {mice_r2:.2f}$)")
    ax.set_xlabel("True socioeconomic index value")
    ax.set_ylabel("Reconstructed value")
    ax.set_title(f"Reconstructing {int(round(ie['mask_fraction']*100))}% masked "
                 f"values ({n_mask} cells)", fontsize=15, color=NAVY)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    path = out_dir / "imputation_reconstruction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
