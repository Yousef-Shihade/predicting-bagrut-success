"""
clean_text.py — Hebrew string standardisation for the three datasets (Step 1).

Project: Predicting Bagrut Success from Municipal Socioeconomics and
         School-Level Institutional Resources
Authors: Yousef Shihade & Shada Esawi

Why this module exists
----------------------
The three datasets are joined on two DIFFERENT keys, and each needs its own kind
of standardisation:

* **Bagrut ↔ CBS** must be joined on the *locality name*, because the ``semel``
  columns are incompatible (in Bagrut ``semel`` is a **school** code; in CBS it is
  a **locality** code — zero overlap). Locality names differ across the sources by
  whitespace padding, parenthetical qualifiers such as (יישוב)/(מוסד), and Hebrew
  spelling variants — above all the "yod-doubling" convention (קרית -> קריית).
  We therefore build a normalised key (``city_norm`` / ``locality_norm``).

* **Bagrut ↔ Budget** is joined on ``semel``, which IS a school code in both, so
  it needs no fuzzy text work — only type coercion and de-duplication. Its Hebrew
  *categorical* columns (district, sector, supervision, ...) still get whitespace
  trimming so their category labels group correctly.

The actual matching happens in Step 2; here we only standardise and cache.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from io_load import load_bagrut, load_budget, load_config, load_ses, resolve

# Gershayim / geresh / quotation marks that appear inside Hebrew names.
_QUOTES = "\"'״׳‘’“”`"
_PAREN_RE = re.compile(r"\([^)]*\)")          # (...) including the brackets
_LEFTOVER_PAREN_RE = re.compile(r"[()\[\]{}]")
_WS_RE = re.compile(r"\s+")


def normalize_hebrew(
    value: Any,
    *,
    strip_parentheses: bool = True,
    yod_prefix_fixes: dict[str, str] | None = None,
    spelling_map: dict[str, str] | None = None,
) -> str | float:
    """Return a normalised locality string (or ``pd.NA``).

    Order matters: structural cleaning (whitespace, parentheses, quotes) runs
    first so the spelling rules always see a canonical string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA

    text = str(value)

    # 1. Remove parenthetical qualifiers, then any stray bracket characters.
    if strip_parentheses:
        text = _PAREN_RE.sub(" ", text)
        text = _LEFTOVER_PAREN_RE.sub(" ", text)

    # 2. Drop Hebrew/ASCII quote marks.
    text = text.translate({ord(ch): " " for ch in _QUOTES})

    # 3. Collapse whitespace and trim.
    text = _WS_RE.sub(" ", text).strip()

    # 4. Generic yod-prefix fixes (word-initial only, e.g. קרית -> קריית).
    for src, dst in (yod_prefix_fixes or {}).items():
        text = re.sub(rf"(^|\s){re.escape(src)}(?=\s|$)", rf"\1{dst}", text)

    # 5. Documented whole-string spelling variants (Bagrut -> CBS spelling).
    if spelling_map and text in spelling_map:
        text = spelling_map[text]

    return text if text else pd.NA


def _rules_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_cleaning", {})
    return {
        "strip_parentheses": tc.get("strip_parentheses", True),
        "yod_prefix_fixes": tc.get("yod_prefix_fixes", {}) or {},
        "spelling_map": tc.get("spelling_map", {}) or {},
    }


def clean_string_column(s: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Vectorised application of :func:`normalize_hebrew`."""
    return s.map(lambda v: normalize_hebrew(v, **_rules_from_cfg(cfg)))


def strip_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Light strip+collapse on listed text columns (no spelling normalisation)."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (out[col].astype("string")
                        .str.replace(_WS_RE, " ", regex=True).str.strip())
    return out


# --------------------------------------------------------------------------- #
# Dataset-level cleaning                                                       #
# --------------------------------------------------------------------------- #
def clean_bagrut(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Strip padded text columns and add the normalised ``city_norm`` join key."""
    out = strip_text_columns(df, cfg["bagrut"]["text_columns"])
    out["city_norm"] = clean_string_column(out[cfg["bagrut"]["city_column"]], cfg)
    return out


def clean_ses(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add the normalised ``locality_norm`` join key to the CBS table."""
    name_col = cfg["ses"]["name_column"]
    out = df.copy()
    out[name_col] = (out[name_col].astype("string")
                     .str.replace(_WS_RE, " ", regex=True).str.strip())
    out["locality_norm"] = clean_string_column(out[name_col], cfg)
    return out


def _parse_nurture_quintile(value: Any) -> float:
    """Extract the UPPER-SCHOOL nurture quintile (1-5) from the raw CBS-style string.

    The Ministry encodes several education stages in one cell, e.g.
        "חטיבה ביניים 5  חטיבה עליונה 5"  -> upper-school quintile 5
        "חטיבה עליונה 1"                  -> 1
    Bagrut is an UPPER-school (חטיבה עליונה) outcome, so we take that stage's
    quintile; if it is absent we fall back to any digit present in the string.

    DIRECTION: this is the Ministry's *nurture* (טיפוח) index, so a HIGHER
    quintile means MORE educational disadvantage, i.e. a greater need for
    nurturing resources. Verified against the data: mean combined grade falls
    monotonically 85.3 -> 74.1 from quintile 1 to 5, and mean municipal cluster
    falls 6.5 -> 2.8 over the same range. Quintile 1 = least disadvantaged.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    text = _WS_RE.sub(" ", str(value)).strip()
    m = re.search(r"חטיבה עליונה\s*(\d)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d)", text)
    return float(m.group(1)) if m else np.nan


def clean_budget(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardise the budget table and key it on ``semel``.

    * Coerce the configured numeric columns (Excel yields mixed str/float).
    * Trim the Hebrew categorical labels so categories group correctly.
    * Parse the upper-school nurture quintile out of its composite string.
    * De-duplicate on ``semel`` (defensive: this extract is already 1 row/school).
    """
    b = cfg["budget"]
    out = df.copy()

    for col in b["numeric_columns"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = strip_text_columns(out, b["text_columns"])

    if "nurture_quintile_raw" in out.columns:
        out["nurture_quintile"] = out["nurture_quintile_raw"].map(_parse_nurture_quintile)

    # Key integrity: drop rows without a school code, then de-duplicate.
    out = out[out["semel"].notna()].copy()
    out["semel"] = out["semel"].astype(int)
    n_before = len(out)
    out = out.drop_duplicates(subset=["semel"], keep="first").reset_index(drop=True)

    report = {
        "rows": len(out),
        "duplicate_semel_dropped": n_before - len(out),
        "unique_semel": int(out["semel"].nunique()),
        "nurture_parsed_pct": (round(100 * out["nurture_quintile"].notna().mean(), 2)
                               if "nurture_quintile" in out.columns else None),
    }
    return out, report


# --------------------------------------------------------------------------- #
# Orchestrated run                                                             #
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load -> clean all THREE datasets and cache them to ``data/``."""
    cfg = cfg or load_config()

    bag_clean = clean_bagrut(load_bagrut(cfg), cfg)
    ses_clean = clean_ses(load_ses(cfg), cfg)
    bud_raw, bud_ingest = load_budget(cfg)
    bud_clean, bud_report = clean_budget(bud_raw, cfg)

    out_dir = resolve(cfg["paths"]["out_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bagrut": out_dir / "bagrut_clean.csv",
        "ses": out_dir / "ses_clean.csv",
        "budget": out_dir / "budget_clean.csv",
    }
    # utf-8-sig so the cached CSVs open cleanly in Excel with Hebrew intact.
    bag_clean.to_csv(paths["bagrut"], index=False, encoding="utf-8-sig")
    ses_clean.to_csv(paths["ses"], index=False, encoding="utf-8-sig")
    bud_clean.to_csv(paths["budget"], index=False, encoding="utf-8-sig")

    return {"bagrut": bag_clean, "ses": ses_clean, "budget": bud_clean,
            "paths": paths, "budget_ingest": bud_ingest, "budget_report": bud_report}


if __name__ == "__main__":
    res = run()
    print(f"[clean_text] bagrut {res['bagrut'].shape} -> {res['paths']['bagrut'].name}")
    print(f"[clean_text] ses    {res['ses'].shape} -> {res['paths']['ses'].name}")
    print(f"[clean_text] budget {res['budget'].shape} -> {res['paths']['budget'].name}")
