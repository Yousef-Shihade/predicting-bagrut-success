"""
ablation.py — SES-only baseline vs Boruta-selected full feature set.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

"Does the school-level budget data add information beyond municipal socioeconomic
status?" is the project's central question, and it deserves a directly controlled
answer rather than an inference drawn from two separately-reported model runs.

This module is the LEGACY single-pass ablation: for each target it tunes
HistGradientBoosting TWICE on IDENTICAL rows — once on municipal features only
(SES only: cluster, log_population, year, locality_form) and once on whatever
Boruta selected from the full SES+budget candidate space. Holding the rows, the
CV folds and the tuning protocol fixed means the R^2 delta is attributable ONLY
to the extra information, not to a different row set or a different protocol.
Superseded by run_nested_ablation.py, which reruns this same idea leakage-free
with RandomForest (the family nested_cv.py selects) -- that is the reported
ablation (Table 5 / ablation_nested.csv), not this module's output.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

import io_load as io
import modeling as ml


def build_ablation_xy(df: pd.DataFrame, target: str, cfg: dict[str, Any],
                      selected_numeric: list[str], selected_categorical: list[str]):
    """Build (X_before, X_after, y, groups) on IDENTICAL rows.

    The row mask requires the target AND every numeric column used by EITHER arm
    to be present, so both models are trained/evaluated on the same schools.
    """
    ab = cfg["ablation"]
    base_num, base_cat = ab["baseline_numeric"], ab["baseline_categorical"]
    group_col = cfg["features"]["group_col"]

    all_numeric = list(dict.fromkeys(base_num + selected_numeric))
    keep = df[target].notna()
    if all_numeric:
        keep = keep & df[all_numeric].notna().all(axis=1)

    sub = df[keep].reset_index(drop=True)
    X_before = io.encode_features(sub, base_num, base_cat)
    X_after = io.encode_features(sub, selected_numeric, selected_categorical)
    y = sub[target].reset_index(drop=True)
    groups = sub[group_col].reset_index(drop=True)
    return X_before, X_after, y, groups


def run_ablation_for_target(df: pd.DataFrame, target: str, cfg: dict[str, Any],
                            selected_numeric: list[str], selected_categorical: list[str]
                            ) -> dict[str, Any]:
    """Tune HGB on the SAME rows for the SES-only baseline and the full set."""
    X_before, X_after, y, groups = build_ablation_xy(
        df, target, cfg, selected_numeric, selected_categorical)

    before = ml.tune_champion(X_before, y, groups, cfg)
    after = ml.tune_champion(X_after, y, groups, cfg)

    bm, am = before["tuned_metrics"], after["tuned_metrics"]
    return {
        "target": target, "n_rows": len(y), "n_schools": int(groups.nunique()),
        "n_features_before": X_before.shape[1], "n_features_after": X_after.shape[1],
        "R2_before": bm["R2"], "R2_after": am["R2"], "dR2": am["R2"] - bm["R2"],
        "RMSE_before": bm["RMSE"], "RMSE_after": am["RMSE"],
        "MAE_before": bm["MAE"], "MAE_after": am["MAE"],
        "after_estimator": after["best_estimator"], "X_after": X_after, "y": y,
        "groups": groups,
    }
