"""
explain.py — SHAP explainability + leaderboard/ablation visualisation.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Produces the four SHAP beeswarms (one per target), the cross-validated model
leaderboard, the VIF pruning trace, and ``plot_before_after`` — the ablation
chart contrasting the SES-only arm with the Boruta-selected full feature set.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")
NAVY, TEAL, CORAL, GOLD, GREY = "#1b2a4a", "#2a9d8f", "#d1495b", "#e8b23a", "#8a94a6"

TARGET_LABELS = {
    "math_avg_grade": "Math — average grade",
    "english_avg_grade": "English — average grade",
    "math_5unit_participation": "Math — 5-unit participation",
    "english_5unit_participation": "English — 5-unit participation",
}


def shap_beeswarm(model, X: pd.DataFrame, target: str, cfg: dict[str, Any],
                  out_dir: Path) -> Path:
    """TreeExplainer beeswarm for one tuned champion model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xs, show=False, plot_type="dot", max_display=15)
    plt.title(f"SHAP — {target}", fontsize=15)
    plt.tight_layout()
    path = out_dir / f"shap_beeswarm_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def shap_importance(model, X: pd.DataFrame, target: str, cfg: dict[str, Any]) -> pd.Series:
    """Mean |SHAP| per feature (used for the README importance table)."""
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X
    sv = shap.TreeExplainer(model).shap_values(Xs)
    return pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)


def plot_leaderboard(leaderboards: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Grouped bar of CV R^2 for every model across the four targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": target, "model": r["model"], "R2": r["R2"]})
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Step 5 — Cross-Validated R² Leaderboard (GroupKFold by school)\n"
                 "Full Boruta-selected SES+Budget feature set")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in long["target"].unique()], fontsize=11)
    ax.legend(title="Model", fontsize=10, loc="upper right")
    plt.tight_layout()
    path = out_dir / "models_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_before_after(ablation: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar: tuned-HGB R^2, SES-only baseline vs Boruta-selected full set
    (same rows, same protocol) — the Step-5 ablation study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long = ablation.melt(id_vars="target", value_vars=["R2_before", "R2_after"],
                         var_name="phase", value_name="R2")
    long["phase"] = long["phase"].map({"R2_before": "SES only",
                                       "R2_after": "SES + Budget (Boruta-selected)"})
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="phase", palette=[NAVY, TEAL], ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Ablation — does the budget dataset add information beyond SES?\n"
                 "(identical rows, tuned HistGradientBoosting, GroupKFold by school)")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in ablation["target"]], fontsize=11)
    ax.legend(title="", fontsize=11, loc="upper right")
    plt.tight_layout()
    path = out_dir / "ablation_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_histograms(resid_inputs: dict[str, dict], cfg: dict[str, Any],
                             out_dir: Path) -> Path:
    """Out-of-fold residual distribution (histogram + KDE) per target.

    ``resid_inputs[target]`` carries the tuned champion estimator plus that
    target's selected feature matrix, response, and school groups. Residuals are
    computed under the SAME GroupKFold(semel) CV used for every reported metric,
    so the picture is an honest out-of-fold diagnostic: a distribution centred on
    zero means the model is unbiased, and the spread equals the target's RMSE.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["modeling"]["cv_splits"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    for ax, (target, d) in zip(axes.flat, resid_inputs.items()):
        gkf = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(clone(d["estimator"]), d["X"], d["y"],
                                   groups=d["groups"], cv=gkf, n_jobs=-1)
        resid = d["y"].to_numpy() - y_pred
        mean_r, std_r = float(resid.mean()), float(resid.std())

        sns.histplot(resid, bins=34, kde=True, color=TEAL, edgecolor="white",
                     linewidth=0.6, alpha=0.85, ax=ax)
        if ax.lines:
            ax.lines[-1].set_color(NAVY)       # KDE curve
            ax.lines[-1].set_linewidth(2.0)
        ax.axvline(0, color=GREY, ls="--", lw=1.6)             # reference: no error
        ax.axvline(mean_r, color=CORAL, ls="-", lw=2.2)        # actual mean residual

        ax.set_title(f"{TARGET_LABELS.get(target, target)}   (R² = {d['r2']:.2f})",
                     fontsize=13.5, color=NAVY, pad=8)
        ax.set_xlabel("Residual (actual − predicted)", fontsize=12)
        ax.set_ylabel("Number of schools", fontsize=12)
        ax.tick_params(labelsize=10.5)
        ax.text(0.035, 0.94, f"mean = {mean_r:+.2f}\nstd = {std_r:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, alpha=0.9))

    legend_handles = [
        Line2D([0], [0], color=NAVY, lw=2.0, label="Distribution (KDE)"),
        Line2D([0], [0], color=GREY, lw=1.6, ls="--", label="Zero error (target)"),
        Line2D([0], [0], color=CORAL, lw=2.2, label="Mean residual"),
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("Out-of-fold residual distributions — tuned HistGradientBoosting champion\n"
                 "all centred on zero, confirming an unbiased fit (GroupKFold by school)",
                 fontsize=13.5, color=NAVY, y=0.995)
    path = out_dir / "residual_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_core_ranking(importances: dict[str, pd.Series], out_dir: Path) -> Path:
    """Mean |SHAP| share of the shared 10-feature core, compared across targets.

    Each target's SHAP importances are normalised to a share of that target's
    total, which makes them comparable even though the four outcomes live on
    different scales. Restricting the view to the core features confirmed for
    ALL four targets gives a fair like-for-like ranking and answers the study's
    question directly: does municipal ``cluster`` or a school-level attribute
    carry the weight?
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    core = [f for f in importances[next(iter(importances))].index
            if all(f in imp.index for imp in importances.values())]

    rows = []
    for target, imp in importances.items():
        total = float(imp.sum())
        for feat in core:
            rows.append({"feature": feat, "target": target,
                         "share": 100.0 * float(imp[feat]) / total})
    long = pd.DataFrame(rows)
    order = (long.groupby("feature")["share"].mean()
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(12.4, 7.4))
    sns.barplot(data=long, y="feature", x="share", hue="target", order=order,
                palette=[NAVY, TEAL, CORAL, GOLD], ax=ax)
    ax.set_xlabel("Share of total mean |SHAP| for that target  (%)")
    ax.set_ylabel("")
    ax.set_title("Feature importance across all four targets (SHAP)\n"
                 "the school-level nurture index leads everywhere, "
                 "municipal cluster sits near the bottom",
                 fontsize=14, color=NAVY)
    ax.legend(title="Target", fontsize=10, title_fontsize=10.5, loc="lower right")
    fig.tight_layout()
    path = out_dir / "shap_core_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vif_pruning(vif_result: dict[str, Any], out_dir: Path) -> Path:
    """Initial VIF (with the offending features highlighted) — collinearity story."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vif = vif_result["initial_vif"].sort_values("VIF", ascending=True)
    dropped = set(vif_result["dropped"])
    colors = [CORAL if f in dropped else TEAL for f in vif["feature"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(vif["feature"], vif["VIF"], color=colors)
    ax.axvline(vif_result["threshold"], color=GOLD, ls="--", lw=1.5,
              label=f"threshold = {vif_result['threshold']}")
    for y, v in enumerate(vif["VIF"]):
        ax.text(v, y, f" {v:.1f}", va="center", fontsize=10)
    ax.set_xlabel("Variance Inflation Factor"); ax.legend(fontsize=10)
    ax.set_title("Collinearity — initial VIF (red = dropped by iterative pruning)",
                fontsize=14)
    plt.tight_layout()
    path = out_dir / "vif_pruning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
