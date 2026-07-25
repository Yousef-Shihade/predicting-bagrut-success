"""
run_nested_ablation.py — the study's central experiment, re-run leakage-free.

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Question: how much predictive information do the school-level institutional
features add beyond municipal socioeconomics alone?

Protocol (matches run_nested_cv.py so the two are comparable):
    * Both arms are evaluated on IDENTICAL rows. Rows are fixed by the
      complete-case rule over the full candidate set, then the municipal arm
      simply uses a column subset, so no row can differ between arms.
    * Outer GroupKFold(semel) provides the evaluation folds.
    * Inside each outer TRAINING fold only: VIF pruning, Boruta selection (full
      arm), and hyperparameter search. The municipal arm is a fixed, small,
      pre-specified set, so it receives tuning but no selection.
    * The reported delta is therefore a difference between two honestly
      evaluated models rather than between two optimistically fitted ones.

Interpretation note: a positive delta demonstrates additional out-of-sample
PREDICTIVE information, not a causal effect of school resources.

Usage:
    python code/run_nested_ablation.py
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
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold, RandomizedSearchCV  # noqa: E402

import explain  # noqa: E402
import nested_cv  # noqa: E402
from io_load import build_xy, load_cleaned, load_config, resolve  # noqa: E402

ARM_FAMILY = "RandomForest"   # the family that wins every target under nested CV


def _cols_for_municipal(X: pd.DataFrame, cfg: dict) -> list[str]:
    """Encoded columns belonging to the municipal-only baseline arm."""
    ab = cfg["ablation"]
    keep = [c for c in ab["baseline_numeric"] if c in X.columns]
    for cat in ab["baseline_categorical"]:
        keep += [c for c in X.columns if c.startswith(cat + "_")]
    return keep


def main() -> None:
    cfg = load_config()
    models_dir = resolve(cfg["paths"]["out_models"])
    feats = cfg["features"]
    numeric_candidates = list(feats["numeric"])
    seed = cfg["seed"]
    n_out = cfg["modeling"]["cv_splits"]
    n_in = cfg["modeling"].get("inner_cv_splits", 3)
    n_iter = cfg["modeling"]["tuning_iter"]
    space = nested_cv.candidate_spaces(seed)[ARM_FAMILY]

    df = load_cleaned(cfg, drop_outliers=False)   # primary analysis keeps all schools

    print("=" * 78)
    print("NESTED ABLATION — municipal-only vs full school-level feature set")
    print(f"model family: {ARM_FAMILY} | outer folds: {n_out} | inner folds: {n_in}")
    print("=" * 78)

    rows, fold_rows = [], []
    for target in cfg["targets"]:
        X, y, groups = build_xy(df, target, numeric_candidates,
                                feats["categorical"], feats["group_col"])
        muni_cols = _cols_for_municipal(X, cfg)
        print(f"\n{target}:  n={len(y)} identical rows, schools={groups.nunique()}, "
              f"municipal cols={len(muni_cols)}, full candidate cols={X.shape[1]}")

        outer = GroupKFold(n_splits=n_out)
        per_arm = {"municipal": [], "full": []}
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            g_tr = groups.iloc[tr]
            inner = GroupKFold(n_splits=min(n_in, g_tr.nunique()))

            for arm in ("municipal", "full"):
                if arm == "municipal":
                    cols = muni_cols
                else:
                    # Selection happens on training rows only.
                    cols = nested_cv._select_features_on_training_fold(
                        X_tr, y_tr, numeric_candidates, cfg)
                search = RandomizedSearchCV(
                    clone(space["estimator"]), param_distributions=space["param_dist"],
                    n_iter=n_iter, scoring="r2", cv=inner, random_state=seed,
                    n_jobs=-1, refit=True)
                search.fit(X_tr[cols], y_tr, groups=g_tr)
                pred = search.best_estimator_.predict(X_te[cols])
                m = nested_cv._metrics(y_te.to_numpy(), pred)
                per_arm[arm].append(m["R2"])
                fold_rows.append({"target": target, "fold": fold, "arm": arm,
                                  "n_features": len(cols), **m})

        mu = np.array(per_arm["municipal"])
        fu = np.array(per_arm["full"])
        d = fu - mu
        rows.append({
            "target": target, "n_rows": len(y), "n_schools": int(groups.nunique()),
            "R2_municipal_mean": mu.mean(), "R2_municipal_std": mu.std(ddof=1),
            "R2_full_mean": fu.mean(), "R2_full_std": fu.std(ddof=1),
            "dR2_mean": d.mean(), "dR2_std": d.std(ddof=1),
            "dR2_min": d.min(), "dR2_max": d.max(),
            "folds_improved": int((d > 0).sum()), "n_folds": len(d),
        })
        print(f"    municipal R2={mu.mean():+.3f} +/- {mu.std(ddof=1):.3f}   "
              f"full R2={fu.mean():+.3f} +/- {fu.std(ddof=1):.3f}   "
              f"dR2={d.mean():+.3f} +/- {d.std(ddof=1):.3f}  "
              f"(improved in {int((d > 0).sum())}/{len(d)} folds)")

    summary = pd.DataFrame(rows)
    enc = cfg["io"]["encoding"]
    summary.to_csv(models_dir / "ablation_nested.csv", index=False, encoding=enc)
    pd.DataFrame(fold_rows).to_csv(models_dir / "ablation_nested_per_fold.csv",
                                   index=False, encoding=enc)

    graphs_dir = resolve(cfg["paths"]["out_graphs"])
    explain.plot_nested_ablation(summary, graphs_dir)

    print("\n" + "=" * 78)
    print(f"mean dR2 across the four targets: {summary['dR2_mean'].mean():+.4f}")
    print(f"[SAVED] ablation_nested.csv, ablation_nested_per_fold.csv, "
          f"nested_ablation.png -> {graphs_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
