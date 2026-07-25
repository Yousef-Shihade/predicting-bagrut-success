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

# Human-readable axis labels. Two of these also correct literal translations of
# the Ministry's Hebrew field names that would otherwise mislead a reader:
#   * "שעות פרטניות" is the individual/small-group INSTRUCTION allocation given
#     to schools, not commercial private tutoring.
#   * "שירותי היקף" means ancillary/support services; "perimeter" was a literal
#     rendering of היקף and is not a meaningful English term here.
FEATURE_LABELS = {
    "nurture_quintile": "Nurture index (school, 5 = most disadvantaged)",
    "transport_per_student": "Transport budget line / student",
    "log_school_size": "School size (log)",
    "avg_class_size": "Average class size",
    "log_population": "Locality population (log)",
    "perimeter_per_student": "Support-services budget / student",
    "projects_per_student": "Projects budget / student",
    "purchases_per_student": "Purchases budget / student",
    "tuition_per_student": "Tuition budget / student",
    "cluster": "Municipal SES cluster",
    "private_hours_per_student": "Individual-instruction hours / student",
    "special_ed_share": "Special-education share",
}


def pretty(name: str) -> str:
    """Display label for a feature, falling back to the raw column name."""
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def shap_beeswarm(model, X: pd.DataFrame, target: str, cfg: dict[str, Any],
                  out_dir: Path) -> Path:
    """TreeExplainer beeswarm for the supplied fitted tree model.

    Callers are responsible for passing the model that should be explained —
    run_step5.py passes the final RandomForest refit (the family selected by
    nested_cv.py), so this figure matches the reported champion.
    """
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
    """Grouped bar of single-pass CV R^2 for every model across the four
    targets. Legacy: features are selected and models scored on the SAME
    full sample, so these numbers are optimistic. Superseded by
    ``plot_nested_leaderboard`` above, which is what the report reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": target, "model": r["model"], "R2": r["R2"]})
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Legacy single-pass R² leaderboard (NOT reported; GroupKFold by school)\n"
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
    """Grouped bar: legacy single-pass tuned-HGB R^2, SES-only baseline vs
    Boruta-selected full set (same rows, same protocol). Superseded by
    ``plot_nested_ablation`` above, which is what the report reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long = ablation.melt(id_vars="target", value_vars=["R2_before", "R2_after"],
                         var_name="phase", value_name="R2")
    long["phase"] = long["phase"].map({"R2_before": "SES only",
                                       "R2_after": "SES + Budget (Boruta-selected)"})
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="phase", palette=[NAVY, TEAL], ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Legacy ablation (single-pass, NOT reported) — does the budget dataset\n"
                 "add information beyond SES? (identical rows, tuned HGB, GroupKFold by school)")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in ablation["target"]], fontsize=11)
    ax.legend(title="", fontsize=11, loc="upper right")
    plt.tight_layout()
    path = out_dir / "ablation_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nested_leaderboard(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar of outer-fold R^2 (mean +/- sd) per model, per target.

    ``summary`` is the table written by run_nested_cv.py (leaderboard_nested.csv:
    columns target, model, R2_mean, R2_std, ...). This is the reproducible,
    leakage-free counterpart of the older ``plot_leaderboard`` above, whose
    numbers come from the single-pass tournament and are not the headline
    result.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = list(dict.fromkeys(summary["target"]))
    models = list(dict.fromkeys(summary["model"]))
    x = np.arange(len(targets))
    width = 0.8 / max(len(models), 1)
    palette = [NAVY, TEAL, CORAL, GOLD]

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("target").reindex(targets)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, sub["R2_mean"], width, yerr=sub["R2_std"],
                      capsize=3, color=palette[i % len(palette)], label=model)
        ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Nested outer-fold R² leaderboard (GroupKFold by school)\n"
                "Every family tuned inside the same inner folds — the headline comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=11)
    ax.set_ylabel("Outer-fold R²  (mean ± sd, higher = better)")
    ax.legend(title="Model", fontsize=10, loc="upper left")
    plt.tight_layout()
    path = out_dir / "nested_leaderboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nested_ablation(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar: municipal-only vs full-set outer-fold R^2, same rows.

    ``summary`` is the table written by run_nested_ablation.py
    (ablation_nested.csv: columns target, R2_municipal_mean, R2_full_mean, ...).
    Reproducible, leakage-free counterpart of the older ``plot_before_after``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    long = summary.melt(id_vars="target",
                        value_vars=["R2_municipal_mean", "R2_full_mean"],
                        var_name="phase", value_name="R2")
    long["phase"] = long["phase"].map({"R2_municipal_mean": "Municipal only",
                                       "R2_full_mean": "Municipal + school-level (Boruta-selected)"})
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="phase", palette=[NAVY, TEAL], ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Nested ablation — does the school-level data add information beyond SES?\n"
                "(identical rows, RandomForest, GroupKFold by school, same protocol as the leaderboard)")
    ax.set_xlabel(""); ax.set_ylabel("Outer-fold R²  (mean, higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in summary["target"]], fontsize=11)
    ax.legend(title="", fontsize=11, loc="upper right")
    plt.tight_layout()
    path = out_dir / "nested_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_histograms(resid_inputs: dict[str, dict], cfg: dict[str, Any],
                             out_dir: Path) -> Path:
    """Out-of-fold residual distribution (histogram + KDE) per target, for the
    LEGACY single-pass tuned-HGB model. NOT used by the report -- Figure 1
    comes from ``plot_nested_residuals`` below, built on nested_cv.py's
    out-of-fold predictions for the actually-selected family.

    ``resid_inputs[target]`` carries the legacy tuned HGB estimator plus that
    target's selected feature matrix, response, and school groups. Residuals
    are computed under GroupKFold(semel) CV, so the picture is an honest
    out-of-fold diagnostic for THIS model: a distribution centred on zero
    means the model is unbiased, and the spread equals the target's RMSE.
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
    fig.suptitle("Out-of-fold residual distributions — legacy tuned HistGradientBoosting\n"
                 "(single-pass path; NOT the reported model -- see residuals_nested.png)",
                 fontsize=13.5, color=NAVY, y=0.995)
    path = out_dir / "residual_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nested_residuals(oof: pd.DataFrame, out_dir: Path,
                          targets: list[str] | None = None) -> Path:
    """Residual distributions built from NESTED out-of-fold predictions.

    ``oof`` is the table written by run_nested_cv.py (columns: target, model,
    y_true, y_pred). Because those predictions come from outer folds whose rows
    were never seen during feature selection or tuning, this diagnostic reflects
    the same honest protocol as the reported leaderboard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = targets or list(dict.fromkeys(oof["target"]))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    for ax, target in zip(axes.flat, targets):
        sub = oof[oof["target"] == target]
        resid = sub["y_true"].to_numpy() - sub["y_pred"].to_numpy()
        mean_r, std_r = float(resid.mean()), float(resid.std())
        model = sub["model"].iloc[0]

        sns.histplot(resid, bins=34, kde=True, color=TEAL, edgecolor="white",
                     linewidth=0.6, alpha=0.85, ax=ax)
        if ax.lines:
            ax.lines[-1].set_color(NAVY)
            ax.lines[-1].set_linewidth(2.0)
        ax.axvline(0, color=GREY, ls="--", lw=1.6)
        ax.axvline(mean_r, color=CORAL, ls="-", lw=2.2)
        ax.set_title(f"{TARGET_LABELS.get(target, target)}   ({model})",
                     fontsize=13, color=NAVY, pad=6)
        ax.set_xlabel("Residual (actual − predicted)", fontsize=11.5)
        ax.set_ylabel("Number of schools", fontsize=11.5)
        ax.tick_params(labelsize=10.5)
        ax.text(0.035, 0.94, f"mean = {mean_r:+.3f}\nsd = {std_r:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, alpha=0.9))

    handles = [
        Line2D([0], [0], color=NAVY, lw=2.0, label="Distribution (KDE)"),
        Line2D([0], [0], color=GREY, lw=1.6, ls="--", label="Zero error"),
        Line2D([0], [0], color=CORAL, lw=2.2, label="Mean residual"),
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Nested out-of-fold residual distributions (selected model per target)",
                 fontsize=14, color=NAVY, y=0.995)
    path = out_dir / "residuals_nested.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
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

    # `cluster` is the study's comparison point, so it is always shown even when
    # it falls out of the shared core (Boruta rejects it for some targets). A
    # target where it was never selected simply contributes a zero-height bar,
    # which is itself the finding.
    focus = "cluster"
    if focus not in core and any(focus in imp.index for imp in importances.values()):
        core = core + [focus]

    rows = []
    for target, imp in importances.items():
        total = float(imp.sum())
        for feat in core:
            val = float(imp[feat]) if feat in imp.index else 0.0
            rows.append({"feature": feat, "target": target,
                         "share": 100.0 * val / total})
    long = pd.DataFrame(rows)
    order = (long.groupby("feature")["share"].mean()
             .sort_values(ascending=False).index.tolist())
    targets = list(importances.keys())

    # One small-multiple panel per target: 40 bars in a single grouped chart is
    # unreadable, whereas four 10-bar panels sharing an x-axis compare cleanly.
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
    xmax = long["share"].max() * 1.16
    for ax, target in zip(axes.flat, targets):
        sub = long[long["target"] == target].set_index("feature").reindex(order)
        colors = [NAVY if f == "nurture_quintile" else (CORAL if f == "cluster" else TEAL)
                  for f in order]
        ypos = np.arange(len(order))
        ax.barh(ypos, sub["share"].to_numpy(), color=colors, height=0.70, alpha=0.9)
        for i, v in enumerate(sub["share"].to_numpy()):
            label = "not selected" if v == 0 else f"{v:.1f}"
            ax.text(v + 0.8, i, label, va="center",
                    fontsize=10 if v == 0 else 12, color="#333333")
        ax.set_yticks(ypos)
        ax.set_yticklabels([pretty(f) for f in order], fontsize=11)
        ax.invert_yaxis()
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=15, color=NAVY, pad=6)
        ax.set_xlim(0, xmax)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Share of that target's total mean |SHAP|  (%)", fontsize=13)

    n_core = len(core) - (1 if focus in core and not all(
        focus in imp.index for imp in importances.values()) else 0)
    fig.suptitle(f"Feature importance per target (SHAP, {n_core}-feature shared core"
                 f" plus municipal cluster)\n"
                 "navy = school-level nurture index, red = municipal cluster",
                 fontsize=15, color=NAVY, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
