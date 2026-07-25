"""
io_load.py — Step 5 input loading & feature-matrix construction.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Loads Step 4's cleaned modeling table and builds, per target, the FULL candidate
predictor matrix X (SES + school-level budget features, one-hot encoded
categoricals), the target vector y, and the GroupKFold groups (`semel`).
`index_value` is deliberately NOT a modeling candidate (collinearity with
cluster) — it is loaded only so the VIF report can demonstrate why.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    return (ROOT / rel_path).resolve()


def apply_display_labels(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Translate Hebrew categorical VALUES to English (in place on a copy).

    Must run before one-hot encoding: matplotlib renders right-to-left Hebrew
    reversed, so any dummy column name derived from an untranslated Hebrew
    value (e.g. "sector_יהודי") is unreadable in SHAP plots, Boruta reports, and
    CSV outputs. Columns without a configured map are left untouched.
    """
    labels = cfg.get("display_labels", {})
    out = df.copy()
    for col, mapping in labels.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: mapping.get(v, v))
    return out


def load_cleaned(cfg: dict[str, Any] | None = None,
                 drop_outliers: bool = False) -> pd.DataFrame:
    """Load the Step 4 cleaned table and translate Hebrew categorical values to
    English for readable downstream artifacts.

    ``drop_outliers`` DEFAULTS TO FALSE: the primary analysis keeps every valid
    school. The Step 4 anomaly detector's feature space includes outcome
    variables (combined grade and both participation rates), so excluding the
    flagged rows would condition the modelling sample on the target and could
    silently remove legitimate extreme performers.

    Passing ``drop_outliers=True`` reproduces the exclusion as a sensitivity
    check. That comparison is run by ``run_outlier_sensitivity.py`` and the
    effect is negligible: mean outer-fold R2 differs by +0.001 across the four
    targets, well inside the fold-to-fold standard deviation.
    """
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["cleaned_in"]), encoding=cfg["io"]["encoding"])
    flag = cfg["features"].get("exclude_outlier_flag")
    if drop_outliers and flag and flag in df.columns:
        df = df[~df[flag].astype(bool)].reset_index(drop=True)
    df = apply_display_labels(df, cfg)
    return df


def encode_features(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Return an encoded predictor frame for an arbitrary numeric+categorical set.

    Generic over the feature LIST passed in (not tied to cfg['features']) so it
    can build both the full candidate matrix and the ablation baseline matrix
    from the same function.
    """
    X = df[numeric].copy()
    for c in categorical:
        col = df[c]
        # Numeric-coded categoricals (e.g. locality_form's CBS settlement code)
        # go through Int64 first so dummy labels read as "310" not "310.0";
        # string categoricals (sector, district, ...) are used as-is.
        if is_numeric_dtype(col):
            col = col.astype("Int64").astype("object")
        dummies = pd.get_dummies(col, prefix=c, dummy_na=False)
        X = pd.concat([X, dummies.astype(int)], axis=1)
    return X


def build_xy(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str],
            group_col: str):
    """Return (X, y, groups) for one target, dropping rows with a missing target
    or a missing NUMERIC candidate (categoricals encode NaN as an all-zero row,
    consistent throughout the pipeline)."""
    keep = df[target].notna()
    if numeric:
        keep = keep & df[numeric].notna().all(axis=1)
    X_all = encode_features(df, numeric, categorical)
    X = X_all[keep].reset_index(drop=True)
    y = df.loc[keep, target].reset_index(drop=True)
    groups = df.loc[keep, group_col].reset_index(drop=True)
    return X, y, groups


if __name__ == "__main__":
    cfg = load_config()
    df = load_cleaned(cfg)
    feats = cfg["features"]
    print(f"[io_load] cleaned (outliers excluded): {df.shape}")
    for t in cfg["targets"]:
        X, y, g = build_xy(df, t, feats["numeric"], feats["categorical"], feats["group_col"])
        print(f"  {t:30s} X={X.shape}  y={len(y)}  groups={g.nunique()}")
