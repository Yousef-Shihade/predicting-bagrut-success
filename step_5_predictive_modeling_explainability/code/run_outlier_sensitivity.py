"""
run_outlier_sensitivity.py — does excluding the 49 consensus anomalies matter?

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Step 4 flags 49 school-years that BOTH Isolation Forest and the Local Outlier
Factor call anomalous. That detector's feature space includes outcome variables
(the combined grade and both participation rates), so dropping those rows would
condition the modelling sample on the target and could remove legitimate extreme
performers. The primary analysis therefore keeps every valid school.

This script quantifies what that choice costs or buys: it re-runs the nested
evaluation for the selected model family with the anomalies IN and OUT, and
reports both. Reporting the exclusion as a sensitivity check, rather than
silently applying it, is the honest treatment.

Output (models/):
    outlier_sensitivity.csv

Usage:
    python code/run_outlier_sensitivity.py
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

import pandas as pd  # noqa: E402

import nested_cv  # noqa: E402
from io_load import build_xy, load_cleaned, load_config, resolve  # noqa: E402

FAMILY = "RandomForest"   # the family selected on every target


def main() -> None:
    cfg = load_config()
    models_dir = resolve(cfg["paths"]["out_models"])
    feats = cfg["features"]
    numeric_candidates = list(feats["numeric"])

    print("=" * 78)
    print("OUTLIER SENSITIVITY — consensus anomalies retained vs excluded")
    print(f"model family: {FAMILY}")
    print("=" * 78)

    rows = []
    for drop in (False, True):
        arm = "excluded" if drop else "retained (primary)"
        df = load_cleaned(cfg, drop_outliers=drop)
        print(f"\n--- anomalies {arm}: {len(df)} school-year rows ---")
        for target in cfg["targets"]:
            X, y, groups = build_xy(df, target, numeric_candidates,
                                    feats["categorical"], feats["group_col"])
            res = nested_cv.nested_evaluate(X, y, groups, numeric_candidates,
                                            cfg, families=[FAMILY])
            s = res["summary"].iloc[0]
            rows.append({"arm": arm, "target": target, "n_rows": len(y),
                         "R2_mean": s["R2_mean"], "R2_std": s["R2_std"],
                         "RMSE_mean": s["RMSE_mean"], "MAE_mean": s["MAE_mean"]})
            print(f"    {target:32s} n={len(y):5d}  "
                  f"R2={s['R2_mean']:+.3f} +/- {s['R2_std']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(models_dir / "outlier_sensitivity.csv", index=False,
               encoding=cfg["io"]["encoding"])

    print("\n" + "=" * 78)
    print("DIFFERENCE (excluded minus retained)")
    print("=" * 78)
    piv = out.pivot(index="target", columns="arm", values="R2_mean")
    for target in cfg["targets"]:
        a, b = piv.loc[target, "retained (primary)"], piv.loc[target, "excluded"]
        print(f"{target:32s} retained={a:+.3f}  excluded={b:+.3f}  diff={b - a:+.3f}")
    print("\n[SAVED] outlier_sensitivity.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
