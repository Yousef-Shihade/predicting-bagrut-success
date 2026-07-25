"""
nested_cv.py — leakage-free nested grouped cross-validation.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

WHY THIS MODULE EXISTS
----------------------
The earlier protocol selected features (Boruta) once on the FULL dataset and
tuned hyperparameters on the SAME GroupKFold folds it then reported scores from.
Both steps let information from the evaluation folds influence the model, so the
resulting metrics were optimistic. Grouping by ``semel`` prevents a school from
appearing in both train and test, but it does not prevent selection bias.

This module implements the standard remedy, NESTED grouped cross-validation:

    OUTER GroupKFold(semel)  -> untouched evaluation folds
      within each outer TRAINING fold only:
        * iterative VIF pruning        (fit on training rows only)
        * Boruta feature selection     (fit on training rows only)
        * RandomizedSearchCV tuning    (INNER GroupKFold on training rows only)
      -> score the resulting model once on the held-out OUTER fold

Every data-dependent decision is therefore made without ever seeing the fold it
is scored on. The outer-fold scores are the honest numbers to report.

FAIR MODEL COMPARISON
---------------------
Previously only HistGradientBoosting was tuned while the other families ran at
defaults, which makes "HGB wins" an artefact of unequal effort. Here EVERY
candidate family gets its own randomised search with a comparable budget inside
the same inner folds, so the comparison is like-for-like.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import feature_selection as fs

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  Candidate families + comparable search spaces (fair-competition protocol)
# --------------------------------------------------------------------------- #
def candidate_spaces(seed: int) -> dict[str, dict[str, Any]]:
    """name -> {estimator, param_dist}. Each family gets a real search space."""
    return {
        "Ridge": {
            "estimator": make_pipeline(StandardScaler(), Ridge(random_state=seed)),
            "param_dist": {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]},
        },
        "SGD (linear SVM)": {
            "estimator": make_pipeline(
                StandardScaler(),
                SGDRegressor(random_state=seed, max_iter=3000, tol=1e-4,
                             early_stopping=True)),
            "param_dist": {
                "sgdregressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "sgdregressor__penalty": ["l2", "l1", "elasticnet"],
                "sgdregressor__learning_rate": ["invscaling", "adaptive"],
            },
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=seed, n_jobs=-1),
            "param_dist": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 8, 14, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.3, 0.6, 1.0],
            },
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingRegressor(random_state=seed),
            "param_dist": {
                "learning_rate": [0.02, 0.05, 0.1, 0.2],
                "max_iter": [150, 250, 350, 500],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 40, 60],
                "l2_regularization": [0.0, 0.1, 1.0],
                "max_depth": [None, 3, 5],
            },
        },
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
    }


def _select_features_on_training_fold(X_tr: pd.DataFrame, y_tr: pd.Series,
                                      numeric_candidates: list[str],
                                      cfg: dict[str, Any]) -> list[str]:
    """VIF pruning + Boruta, fit ONLY on this outer training fold.

    Returns the encoded column names to keep. Falls back to all columns if
    Boruta confirms nothing (rare, but leaves the fold evaluable).
    """
    # 1. Iterative VIF on the numeric block present in this training fold.
    num_present = [c for c in numeric_candidates if c in X_tr.columns]
    keep_numeric = num_present
    if num_present:
        sub = X_tr[num_present].dropna()
        if len(sub) > len(num_present) + 1:
            vif_res = fs.iterative_vif_prune(sub, num_present,
                                             cfg["collinearity"]["vif_threshold"])
            keep_numeric = [c for c in vif_res["kept"] if c in X_tr.columns]

    # 2. Boruta over the VIF survivors plus every encoded categorical dummy.
    cat_cols = [c for c in X_tr.columns if c not in num_present]
    cand = keep_numeric + cat_cols
    if not cand:
        return list(X_tr.columns)
    bor = fs.run_boruta(X_tr[cand], y_tr, cfg)
    selected = [c for c in bor["selected"] if c in X_tr.columns]
    return selected if selected else cand


def nested_evaluate(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    numeric_candidates: list[str], cfg: dict[str, Any],
                    families: list[str] | None = None) -> dict[str, Any]:
    """Nested grouped CV for every candidate family on one target.

    Returns per-fold rows, a per-family summary (mean +/- sd over outer folds),
    and the pooled out-of-fold predictions for the best family.
    """
    seed = cfg["seed"]
    n_out = cfg["modeling"]["cv_splits"]
    n_in = cfg["modeling"].get("inner_cv_splits", 3)
    n_iter = cfg["modeling"]["tuning_iter"]
    spaces = candidate_spaces(seed)
    if families:
        spaces = {k: v for k, v in spaces.items() if k in families}

    outer = GroupKFold(n_splits=n_out)
    rows: list[dict[str, Any]] = []
    oof = {name: np.full(len(y), np.nan) for name in spaces}
    fold_features: list[list[str]] = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        g_tr = groups.iloc[tr]

        # --- all data-dependent selection happens on TRAINING ROWS ONLY ---
        feats = _select_features_on_training_fold(X_tr, y_tr, numeric_candidates, cfg)
        fold_features.append(feats)
        X_tr_s, X_te_s = X_tr[feats], X_te[feats]

        inner = GroupKFold(n_splits=min(n_in, g_tr.nunique()))
        for name, spec in spaces.items():
            search = RandomizedSearchCV(
                clone(spec["estimator"]), param_distributions=spec["param_dist"],
                n_iter=n_iter, scoring="r2", cv=inner, random_state=seed,
                n_jobs=-1, refit=True, error_score="raise")
            search.fit(X_tr_s, y_tr, groups=g_tr)
            pred = search.best_estimator_.predict(X_te_s)
            oof[name][te] = pred
            m = _metrics(y_te.to_numpy(), pred)
            rows.append({"fold": fold, "model": name, "n_features": len(feats),
                         "n_train": len(tr), "n_test": len(te), **m,
                         "best_params": str(search.best_params_)})

    per_fold = pd.DataFrame(rows)
    summary = (per_fold.groupby("model")
               .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                    n_features_mean=("n_features", "mean"))
               .sort_values("R2_mean", ascending=False).reset_index())
    best = summary.iloc[0]["model"]
    return {"per_fold": per_fold, "summary": summary, "best_model": best,
            "oof": oof, "fold_features": fold_features}
