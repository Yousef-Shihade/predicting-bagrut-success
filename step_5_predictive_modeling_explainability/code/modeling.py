"""
modeling.py — the modeling tournament, tuning, and CV evaluation.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

For each target we evaluate four models under GroupKFold(semel) CV:
  * Ridge            — regularised linear baseline (standardised)
  * SGDRegressor     — linear regression fit by stochastic gradient descent
                       (default squared-error loss, NOT an SVM; standardised)
  * RandomForest     — bagged tree ensemble (non-linear)
  * HistGradientBoosting — modern boosted trees

This module is the OLDER single-pass path (tournament on the full sample, then
``tune_champion`` tunes HistGradientBoosting only). It is kept for the
descriptive VIF/Boruta/SHAP artefacts, NOT for the headline predictive-
performance numbers — those come from nested_cv.py, where RandomForest is
selected on every target under a fair, equal-effort search across all four
families. See the Step 5 README for which files are the headline source.

GroupKFold keeps every year of a school in the same fold, so the leaderboard is
free of school-level leakage. Standardisation for the linear models happens
inside a Pipeline, i.e. re-fit on each training fold (no leakage).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def build_models(seed: int) -> dict[str, Any]:
    """name -> estimator (pipelines already include scaling where needed)."""
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed)),
        "SGDRegressor (linear)": make_pipeline(
            StandardScaler(),
            SGDRegressor(random_state=seed, max_iter=3000, tol=1e-4,
                         early_stopping=True)),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed,
                                              n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
    }


def _cv_metrics(model, X, y, groups, n_splits: int) -> dict[str, float]:
    cv = GroupKFold(n_splits=n_splits)
    res = cross_validate(model, X, y, groups=groups, cv=cv, scoring=_SCORING,
                         n_jobs=-1, return_train_score=False)
    return {
        "RMSE": float(-res["test_rmse"].mean()), "RMSE_std": float(res["test_rmse"].std()),
        "MAE": float(-res["test_mae"].mean()),
        "R2": float(res["test_r2"].mean()), "R2_std": float(res["test_r2"].std()),
    }


def run_tournament(X, y, groups, cfg) -> pd.DataFrame:
    """CV-evaluate all models for one target; return a leaderboard frame."""
    rows = []
    for name, model in build_models(cfg["seed"]).items():
        m = _cv_metrics(model, X, y, groups, cfg["modeling"]["cv_splits"])
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def tune_champion(X, y, groups, cfg) -> dict[str, Any]:
    """RandomizedSearchCV on HistGradientBoosting under GroupKFold.

    NOTE: this tunes HGB specifically for the legacy single-pass leaderboard
    (``leaderboard_tuned.csv``), which is NOT the headline result. It is not
    the model used for the report's SHAP figures — see ``tune_final_random_
    forest`` below, which refits the family nested_cv.py actually selected.
    """
    seed = cfg["seed"]
    cv = GroupKFold(n_splits=cfg["modeling"]["cv_splits"])
    param_dist = {
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_iter": [150, 250, 350, 500],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40, 60],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_depth": [None, 3, 5],
    }
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=seed),
        param_distributions=param_dist, n_iter=cfg["modeling"]["tuning_iter"],
        scoring="r2", cv=cv, random_state=seed, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)

    best = search.best_estimator_
    tuned = _cv_metrics(best, X, y, groups, cfg["modeling"]["cv_splits"])
    return {"best_estimator": best, "best_params": search.best_params_,
            "cv_best_r2": float(search.best_score_), "tuned_metrics": tuned}


def tune_final_random_forest(X, y, groups, cfg) -> dict[str, Any]:
    """RandomizedSearchCV on RandomForest under GroupKFold, on the descriptive
    (full-sample, full-data-Boruta-selected) feature matrix.

    RandomForest is the family nested_cv.py selects for every target. This
    refit gives SHAP a model that actually matches the reported champion,
    using the identical search space as the headline nested evaluation
    (``nested_cv.candidate_spaces``), so the descriptive fit and the reported
    predictive performance describe the same family.
    """
    import nested_cv as ncv
    from sklearn.base import clone

    seed = cfg["seed"]
    cv = GroupKFold(n_splits=cfg["modeling"]["cv_splits"])
    space = ncv.candidate_spaces(seed)["RandomForest"]
    search = RandomizedSearchCV(
        clone(space["estimator"]), param_distributions=space["param_dist"],
        n_iter=cfg["modeling"]["tuning_iter"], scoring="r2", cv=cv,
        random_state=seed, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)

    best = search.best_estimator_
    tuned = _cv_metrics(best, X, y, groups, cfg["modeling"]["cv_splits"])
    return {"best_estimator": best, "best_params": search.best_params_,
            "cv_best_r2": float(search.best_score_), "tuned_metrics": tuned}
