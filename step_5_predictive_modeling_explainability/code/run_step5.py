"""
run_step5.py — Step 5 orchestrator (modeling, ablation, Boruta, SHAP).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Pipeline, following the redesign that merges all three datasets from the start:
    load -> iterative VIF collinearity pruning (full 15-feature numeric set)
    -> per target: Boruta feature selection (SES+budget candidates)
                -> model tournament (GroupKFold CV) -> tune champion (HGB)
                -> SHAP -> ablation (SES-only vs Boruta-selected, same rows)
    -> serialize models -> leaderboards -> comprehensive console report

Usage:
    python code/run_step5.py
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

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

import ablation as ab  # noqa: E402
import explain  # noqa: E402
import feature_selection as fs  # noqa: E402
import modeling  # noqa: E402
from io_load import build_xy, encode_features, load_cleaned, load_config, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def _map_selected_to_original(selected: list[str], numeric_candidates: list[str],
                              categorical_candidates: list[str]) -> tuple[list[str], list[str]]:
    """Map Boruta's selected ENCODED column names back to original feature names.

    A numeric candidate keeps its name unchanged; for a categorical, if ANY of
    its one-hot dummy levels was selected, the WHOLE categorical is carried into
    the ablation "after" arm (a fair, interpretable unit — "sector was selected",
    not "only the Arab-sector dummy was selected").
    """
    sel_numeric = [c for c in selected if c in numeric_candidates]
    sel_categorical = []
    for cat in categorical_candidates:
        prefix = cat + "_"
        if any(s.startswith(prefix) for s in selected):
            sel_categorical.append(cat)
    return sel_numeric, sel_categorical


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])
    models_dir = resolve(cfg["paths"]["out_models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    targets = cfg["targets"]
    feats = cfg["features"]

    print(_hr())
    print("STEP 5 — PREDICTIVE MODELING, ABLATION & EXPLAINABILITY")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    df = load_cleaned(cfg)
    print(f"\n[DATA] modeling rows {len(df)} (Step-4 consensus outliers excluded)")
    print(f"    candidate numeric ({len(feats['numeric'])}): {feats['numeric']}")
    print(f"    candidate categorical ({len(feats['categorical'])}): {feats['categorical']}")

    # ---------------- Collinearity: iterative VIF pruning -------------------- #
    print("\n" + _hr("-"))
    print("COLLINEARITY — ITERATIVE VIF PRUNING")
    vif_candidates = feats["numeric"] + cfg["collinearity"]["vif_features_extra"]
    vif_result = fs.iterative_vif_prune(df, vif_candidates, cfg["collinearity"]["vif_threshold"])
    print(f"    candidates checked ({len(vif_candidates)}): {vif_candidates}")
    print(f"    threshold: {vif_result['threshold']}")
    if not vif_result["history"].empty:
        for _, r in vif_result["history"].iterrows():
            print(f"    step {int(r['step'])}: dropped '{r['dropped_feature']}' "
                  f"(VIF={r['VIF_at_drop']:.2f}) -> {int(r['remaining_after'])} remain")
    else:
        print("    no features exceeded the threshold")
    print(f"    KEPT ({len(vif_result['kept'])}): {vif_result['kept']}")
    print(f"    DROPPED ({len(vif_result['dropped'])}): {vif_result['dropped']}")
    numeric_candidates = [c for c in vif_result["kept"] if c != "index_value"]
    vif_result["initial_vif"].to_csv(resolve(cfg["paths"]["vif_out"]), index=False,
                                     encoding=cfg["io"]["encoding"])
    vif_plot = explain.plot_vif_pruning(vif_result, graphs)

    # ---------------- per-target: Boruta -> tournament -> tune -> SHAP -> ablation
    leaderboards: dict[str, pd.DataFrame] = {}
    tuned_store: dict[str, dict] = {}
    boruta_rows = []
    ablation_rows = []
    resid_inputs: dict[str, dict] = {}
    shap_importances: dict[str, pd.Series] = {}
    plots: list[Path] = []

    for target in targets:
        print("\n" + _hr("-"))
        print(f"TARGET: {target}")
        X, y, groups = build_xy(df, target, numeric_candidates, feats["categorical"],
                                feats["group_col"])
        print(f"    n={len(y)}  schools(groups)={groups.nunique()}  candidate columns={X.shape[1]}")

        # ----- Boruta feature selection (full SES+budget candidate space) ----- #
        bor = fs.run_boruta(X, y, cfg)
        print(f"    Boruta confirmed ({len(bor['confirmed'])}): {bor['confirmed']}")
        print(f"    Boruta tentative ({len(bor['tentative'])}): {bor['tentative']}")
        Xsel = X[bor["selected"]]
        print(f"    -> using {len(bor['selected'])} selected column(s) for models")
        for c in bor["selected"]:
            boruta_rows.append({"target": target, "selected_feature": c,
                               "rank": bor["ranking"].get(c)})

        # ----- Tournament (CV) ----- #
        lb = modeling.run_tournament(Xsel, y, groups, cfg)
        leaderboards[target] = lb
        print("    tournament (GroupKFold CV):")
        for _, r in lb.iterrows():
            print(f"      {r['model']:22s} R2={r['R2']:+.3f}  RMSE={r['RMSE']:.3f}  MAE={r['MAE']:.3f}")

        # ----- Tune champion (HGB) ----- #
        tuned = modeling.tune_champion(Xsel, y, groups, cfg)
        tuned_store[target] = {**tuned, "features": list(Xsel.columns)}
        tm = tuned["tuned_metrics"]
        print(f"    TUNED HistGradientBoosting -> R2={tm['R2']:+.3f} RMSE={tm['RMSE']:.3f} MAE={tm['MAE']:.3f}")

        model_path = models_dir / f"{target}_hgb.joblib"
        joblib.dump({"model": tuned["best_estimator"], "features": list(Xsel.columns),
                    "target": target, "cv_metrics": tm}, model_path)

        # Keep the champion + its data for the out-of-fold residual diagnostic.
        resid_inputs[target] = {"estimator": tuned["best_estimator"], "X": Xsel,
                                "y": y, "groups": groups, "r2": tm["R2"]}

        # ----- SHAP ----- #
        plots.append(explain.shap_beeswarm(tuned["best_estimator"], Xsel, target, cfg, graphs))
        shap_importances[target] = explain.shap_importance(tuned["best_estimator"], Xsel,
                                                           target, cfg)
        top3 = ", ".join(f"{f} ({v:.3f})" for f, v in shap_importances[target].head(3).items())
        print(f"    SHAP top-3 (mean |SHAP|): {top3}")

        # ----- Ablation: SES-only vs Boruta-selected full set, SAME rows ----- #
        sel_num, sel_cat = _map_selected_to_original(bor["selected"], numeric_candidates,
                                                      feats["categorical"])
        arow = ab.run_ablation_for_target(df, target, cfg, sel_num, sel_cat)
        print(f"    ABLATION  SES-only R2={arow['R2_before']:+.3f}  ->  "
              f"SES+Budget R2={arow['R2_after']:+.3f}  (dR2={arow['dR2']:+.3f}, "
              f"n={arow['n_rows']} identical rows)")
        ablation_rows.append({k: v for k, v in arow.items()
                             if k not in ("after_estimator", "X_after", "y", "groups")})

    # ---------------- leaderboard artefacts ----------------------------------- #
    plots.append(explain.plot_leaderboard(leaderboards, graphs))
    rows = []
    for t, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": t, **r.to_dict()})
    pd.DataFrame(rows).to_csv(resolve(cfg["paths"]["leaderboard_out"]), index=False,
                              encoding=cfg["io"]["encoding"])
    pd.DataFrame(boruta_rows).to_csv(resolve(cfg["paths"]["boruta_out"]), index=False,
                                     encoding=cfg["io"]["encoding"])

    # The headline numbers quoted in the READMEs are the *tuned* champion's, not
    # the untuned tournament's — persist them so they are auditable from a CSV
    # rather than only from inside the .joblib files.
    tuned_rows = [{"target": t,
                   "model": "HistGradientBoosting (tuned)",
                   "n_features": len(s["features"]),
                   **{k: s["tuned_metrics"][k] for k in ("R2", "RMSE", "MAE")}}
                  for t, s in tuned_store.items()]
    pd.DataFrame(tuned_rows).sort_values("R2", ascending=False).to_csv(
        resolve(cfg["paths"]["tuned_out"]), index=False,
        encoding=cfg["io"]["encoding"])

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(resolve(cfg["paths"]["ablation_out"]), index=False,
                       encoding=cfg["io"]["encoding"])
    plots.append(explain.plot_before_after(ablation_df, graphs))

    # Out-of-fold residual diagnostic for the tuned champions (report §6).
    plots.append(explain.plot_residual_histograms(resid_inputs, cfg, graphs))

    # Cross-target SHAP ranking of the shared core features (report §7).
    plots.append(explain.plot_shap_core_ranking(shap_importances, graphs))

    # ---------------- final report --------------------------------------------- #
    print("\n" + _hr())
    print("FINAL CROSS-VALIDATED LEADERBOARD (champion = tuned HistGradientBoosting,")
    print("full Boruta-selected SES+Budget feature set)")
    print(_hr())
    print(f"{'target':32s}{'best model':24s}{'R2':>8}{'RMSE':>9}{'MAE':>9}")
    for t in targets:
        tm = tuned_store[t]["tuned_metrics"]
        print(f"{t:32s}{'HGB (tuned)':24s}{tm['R2']:+8.3f}{tm['RMSE']:9.3f}{tm['MAE']:9.3f}")

    print("\n" + _hr())
    print("ABLATION SUMMARY — SES-only vs SES+Budget (Boruta-selected)")
    print(_hr())
    with pd.option_context("display.width", 130, "display.max_columns", None):
        print(ablation_df[["target", "n_rows", "R2_before", "R2_after", "dR2"]].to_string(index=False))
    print(f"\n  mean dR2 across 4 targets: {ablation_df['dR2'].mean():+.4f}")

    print("\n[ARTEFACTS]")
    print(f"    serialized models : {len(targets)} -> {models_dir.name}/*.joblib")
    print(f"    VIF report        : {Path(cfg['paths']['vif_out']).name}")
    print(f"    Boruta report     : {Path(cfg['paths']['boruta_out']).name}")
    print(f"    Ablation report   : {Path(cfg['paths']['ablation_out']).name}")
    print(f"    leaderboard csv   : {Path(cfg['paths']['leaderboard_out']).name} (untuned tournament)")
    print(f"    tuned champion csv: {Path(cfg['paths']['tuned_out']).name} (headline numbers)")
    print("    graphs:")
    for p in [vif_plot] + plots:
        print(f"      - {Path(p).name:38s} ({Path(p).stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 5 COMPLETE ✔")
    print(_hr())


if __name__ == "__main__":
    main()
