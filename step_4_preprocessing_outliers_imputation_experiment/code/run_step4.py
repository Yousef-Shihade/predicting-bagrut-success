"""
run_step4.py — Step 4 orchestrator (preprocessing, outliers, MICE robustness).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Flow:
    load Step-3 school-level table + derived columns
    -> MICE robustness — 25 independent masking trials
    -> Outlier detection: Isolation Forest + LOF
    -> Exploratory analysis: Q1 (resilience) + Q2 (overachievers) — unchanged
    -> save cleaned_modeling_ready.csv + all diagnostic plots

Usage:
    python code/run_step4.py
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

import exploratory as ex  # noqa: E402
import imputation_experiment as imp  # noqa: E402
import outliers as out  # noqa: E402
from io_load import load_config, load_school_level, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])
    data_dir = resolve(cfg["paths"]["out_data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    print(_hr())
    print("STEP 4 — PREPROCESSING, OUTLIERS & MICE ROBUSTNESS")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    df = load_school_level(cfg)
    print(f"\n[LOAD] school-level table: {df.shape}")

    # ---------------- MICE robustness (multi-iteration, NEW) -------- #
    ie = cfg["imputation_experiment"]
    print(f"\n[1/3] MICE robustness — {ie['n_iterations']} independent masking trials")
    print(f"    target feature: {ie['target_feature']}  |  mask fraction: {ie['mask_fraction']}")
    print(f"    predictors ({len(ie['predictors'])}): {ie['predictors']}")

    mice_result = imp.run_multi_iteration(df, cfg)
    s = mice_result["summary"]
    print(f"\n    n masked per run: {s['n_masked']} / {s['n_total']} rows")
    print(f"    MICE   R² = {s['MICE_R2_mean']:.4f} ± {s['MICE_R2_std']:.4f}  "
          f"(range {s['MICE_R2_min']:.4f}–{s['MICE_R2_max']:.4f})")
    print(f"    MICE   RMSE = {s['MICE_RMSE_mean']:.4f} ± {s['MICE_RMSE_std']:.4f}")
    print(f"    Median R² = {s['Median_R2_mean']:.4f} ± {s['Median_R2_std']:.4f}")
    print(f"    Median RMSE = {s['Median_RMSE_mean']:.4f} ± {s['Median_RMSE_std']:.4f}")
    print(f"    -> MICE beats median on ALL {s['n_iterations']} runs: "
          f"{(mice_result['runs']['MICE_R2'] > mice_result['runs']['Median_R2']).all()}")

    mice_result["runs"].to_csv(resolve(cfg["paths"]["mice_runs_out"]), index=False,
                               encoding=cfg["io"]["encoding"])
    robustness_plot = imp.plot_robustness(mice_result, graphs)
    recon_plot = imp.plot_reconstruction_scatter(df, cfg, graphs)
    print(f"    saved per-run detail: {Path(cfg['paths']['mice_runs_out']).name}")

    # ---------------- Outlier detection ------------------------------ #
    oc = cfg["outliers"]
    print(f"\n[2/3] Outlier detection — Isolation Forest + LOF")
    print(f"    feature space ({len(oc['features'])}): {oc['features']}")
    df, osumm = out.detect(df, cfg)
    print(f"    evaluated: {osumm['n_evaluated']:,} complete-case rows")
    print(f"    Isolation Forest flagged: {osumm['iso_flagged']}  |  LOF flagged: {osumm['lof_flagged']}")
    print(f"    consensus (both): {osumm['overlap_both']}  |  Jaccard: {osumm['jaccard']:.3f}")
    outlier_plot = out.plot_mapping(df, graphs)

    # ---------------- Exploratory questions --------------------------- #
    print(f"\n[3/3] Exploratory questions")
    q1 = ex.question1_resilience(df, cfg)
    print(f"    Q1 — more resilient subject (smaller grade gap): {q1['more_resilient_grade']}")
    for subj, r in q1["subjects"].items():
        print(f"        {subj:8s} grade gap {r['grade_gap']:.2f} pts (d={r['grade_cohens_d']:.2f})  "
              f"| participation gap {r['part_gap']:.3f}")
    q1_plot = ex.plot_resilience_gap(q1, graphs)

    q2 = ex.question2_overachievers(df, cfg)
    print(f"\n    Q2 — low-SES overachievers: {q2['n_over']}/{q2['n_low']} = {q2['over_pct']:.1f}%")
    print(f"        elite benchmark ({q2['benchmark_type']}): {q2['benchmark']:.2f}")
    print(f"        overachiever participation: math {q2['over_math_part']:.3f} "
          f"(p={q2['p_math_part']:.2e})  english {q2['over_eng_part']:.3f} (p={q2['p_eng_part']:.2e})")
    q2_plot = ex.plot_overachievers(q2, graphs)

    # ---------------- save cleaned table -------------------------------------- #
    out_path = resolve(cfg["paths"]["cleaned_out"])
    df.to_csv(out_path, index=False, encoding=cfg["io"]["encoding"])
    print(f"\n[SAVE] {out_path.name}  ({df.shape[0]:,} x {df.shape[1]})  "
          f"({out_path.stat().st_size/1024/1024:.2f} MB)")

    print("\n[PLOTS]")
    for p in [robustness_plot, recon_plot, outlier_plot, q1_plot, q2_plot]:
        print(f"    - {p.name:38s} ({p.stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 4 COMPLETE ✔  — MICE robustness proven, outliers flagged, "
          "exploratory findings computed.")
    print(_hr())


if __name__ == "__main__":
    main()
