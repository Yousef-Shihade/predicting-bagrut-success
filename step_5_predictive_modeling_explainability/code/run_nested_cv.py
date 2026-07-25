"""
run_nested_cv.py — leakage-free nested CV evaluation for all four targets.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Produces the HONEST headline numbers: every feature-selection and tuning
decision is made inside an outer training fold, and all four model families are
tuned with a comparable budget so the comparison is fair.

Outputs (models/):
    leaderboard_nested.csv      per-model mean +/- sd over outer folds, per target
    nested_per_fold.csv         every outer fold's score (full audit trail)
    nested_oof_predictions.csv  pooled out-of-fold predictions of the best family

Usage:
    python code/run_nested_cv.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import feature_selection as fs  # noqa: E402
import nested_cv  # noqa: E402
from io_load import build_xy, load_cleaned, load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    models_dir = resolve(cfg["paths"]["out_models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    feats = cfg["features"]

    # Primary analysis keeps EVERY valid school. Outlier exclusion is reported
    # separately as a sensitivity analysis, never as the headline sample,
    # because the Step 4 detector's feature space includes outcome variables.
    df = load_cleaned(cfg, drop_outliers=False)

    print("=" * 78)
    print("NESTED GROUPED CROSS-VALIDATION (leakage-free, fair tuning)")
    print("=" * 78)
    print(f"rows={len(df)}  outer folds={cfg['modeling']['cv_splits']}  "
          f"inner folds={cfg['modeling'].get('inner_cv_splits', 3)}  "
          f"search iters={cfg['modeling']['tuning_iter']}")

    # VIF candidate pool is defined here, but pruning itself happens per fold.
    numeric_candidates = [c for c in feats["numeric"]]

    all_summary, all_folds, oof_rows = [], [], []
    for target in cfg["targets"]:
        X, y, groups = build_xy(df, target, numeric_candidates,
                                feats["categorical"], feats["group_col"])
        print("\n" + "-" * 78)
        print(f"TARGET: {target}   n={len(y)}  schools={groups.nunique()}  "
              f"encoded cols={X.shape[1]}")

        res = nested_cv.nested_evaluate(X, y, groups, numeric_candidates, cfg)
        s = res["summary"].copy()
        s.insert(0, "target", target)
        all_summary.append(s)

        pf = res["per_fold"].copy()
        pf.insert(0, "target", target)
        all_folds.append(pf)

        for _, r in s.iterrows():
            print(f"    {r['model']:22s} R2={r['R2_mean']:+.3f} +/- {r['R2_std']:.3f}"
                  f"   RMSE={r['RMSE_mean']:.3f}   MAE={r['MAE_mean']:.3f}"
                  f"   feats~{r['n_features_mean']:.1f}")
        print(f"    -> best family (outer-fold mean R2): {res['best_model']}")

        best_oof = res["oof"][res["best_model"]]
        oof_rows.append(pd.DataFrame({"target": target, "model": res["best_model"],
                                      "y_true": y.to_numpy(), "y_pred": best_oof,
                                      "semel": groups.to_numpy()}))

    summary = pd.concat(all_summary, ignore_index=True)
    per_fold = pd.concat(all_folds, ignore_index=True)
    oof = pd.concat(oof_rows, ignore_index=True)

    enc = cfg["io"]["encoding"]
    summary.to_csv(models_dir / "leaderboard_nested.csv", index=False, encoding=enc)
    per_fold.to_csv(models_dir / "nested_per_fold.csv", index=False, encoding=enc)
    oof.to_csv(models_dir / "nested_oof_predictions.csv", index=False, encoding=enc)

    print("\n" + "=" * 78)
    print("HONEST LEADERBOARD (best family per target, outer-fold mean +/- sd)")
    print("=" * 78)
    for target in cfg["targets"]:
        sub = summary[summary["target"] == target].iloc[0]
        print(f"{target:32s}{sub['model']:22s}"
              f"R2={sub['R2_mean']:+.3f} +/- {sub['R2_std']:.3f}   "
              f"RMSE={sub['RMSE_mean']:.3f}   MAE={sub['MAE_mean']:.3f}")

    print("\n[SAVED] leaderboard_nested.csv, nested_per_fold.csv, "
          "nested_oof_predictions.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
