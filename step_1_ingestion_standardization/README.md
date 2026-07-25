# Step 1 — Ingestion & Standardization

**Project:** Predicting Bagrut Success from Municipal Socioeconomics and School-Level Institutional Resources
**Authors:** Yousef Shihade & Shada Esawi

> This step ingests **all three sources** — the Bagrut exam records, the CBS
> municipal socioeconomic index, and the Ministry-of-Education budget report —
> each of which arrives with its own encoding, header and Hebrew-text problems.
> Treating the budget file as a **first-class third source from the outset** is
> what lifts the analysis from a municipality-only design into a school-level
> feature space.

---

## 1. Directory structure

```
step_1_ingestion_standardization/
├── README.md
├── config.yaml            # paths, 3 dataset schemas, Hebrew rules, display labels
├── code/
│   ├── io_load.py         # 3 loaders (BOM, CBS header offset, openpyxl colour patch)
│   ├── clean_text.py      # Hebrew normalisation + per-dataset cleaning
│   ├── visualize.py       # 5 diagnostic plots
│   └── run_step1.py       # orchestrator + verification summary
├── data/                  # bagrut_clean.csv · ses_clean.csv · budget_clean.csv
└── graphs/                # 5 PNGs
```

Run: `python code/run_step1.py` (paths anchored to the step folder; CWD-independent).

---

## 2. The three ingestion challenges (and how each was solved)

| # | Dataset | Problem | Solution |
|---|---|---|---|
| 1 | **Bagrut** `israel_bagrut_averages.csv` | UTF-8 **BOM** corrupts the first header (`grade` → `﻿grade`); all text columns padded with trailing spaces | read with `utf-8-sig`; strip/collapse whitespace on `subject`/`city`/`school` |
| 2 | **CBS SES** `downloadFile.xlsx` | Real header sits on **row index 10**; `..` marks unranked localities; Hebrew headers with trailing qualifiers | `header=10`, drop empty rows, **prefix**-match the rename map, `..` → `NaN` |
| 3 | **Budget** `school_budget.xlsx` | Malformed `styles.xml` → `ValueError: Colors must be aRGB hex values`; **row 2 is a grand-totals row**; headers carry stray double spaces | **monkeypatch openpyxl's `RGB` descriptor** with a lenient validator *before* opening; drop the `סה"כ` totals row; whitespace-normalise headers and match **exactly** |

**Why exact (not prefix) matching for the budget file:** `סה"כ תקציב שכר ותשלומים`
would otherwise also capture its sibling `... - ללא קורונה`, silently importing the
wrong column.

---

## 3. Hebrew standardisation — why the two joins differ

The three datasets are linked by **two different keys**:

- **Bagrut ↔ CBS** must join on the **locality name**, because `semel` is a
  *school* code in Bagrut but a *locality* code in CBS — **zero overlap**. Names
  differ by whitespace, parenthetical qualifiers `(יישוב)/(מוסד)`, and Hebrew
  spelling variants (yod-doubling: `קרית` → `קריית`). We build a normalised
  `city_norm` / `locality_norm` key.
- **Bagrut ↔ Budget** joins on **`semel`**, a school code in *both* — a clean key
  join needing no fuzzy work, only type coercion and de-duplication.

`normalize_hebrew()` applies, in order: strip parentheses → drop gershayim/quotes
→ collapse whitespace → yod-prefix fixes → documented spelling map.

---

## 4. Verified results

### Dataset scale

| Dataset | Rows | Key cardinality |
|---|--:|---|
| Bagrut exam records | **69,638** | 1,063 schools · 315 cities · 2013–2016 |
| CBS socioeconomic index | **1,208** | 1,199 localities · cluster present 97.9 % |
| Budget institutions | **4,718** | 4,718 unique `semel` (0 duplicates) |

Budget ingestion: **25 / 25 requested columns resolved**, 1 totals row dropped,
nurture quintile parsed for **85.5 %** of institutions.

### Join feasibility (executed in Step 2)

| Join | Key | Result |
|---|---|---|
| Bagrut ↔ CBS | normalised locality name | 291/315 cities match exactly = **92.8 % of records** → fuzzy stages close the rest in Step 2 |
| Bagrut ↔ Budget | **`semel`** | **1,048 / 1,063 schools = 98.6 %** → clean key join, no fuzzy needed |

### 🔑 New school-level attributes unlocked by Dataset 3

These directly address the narrow feature space that limited the two-dataset
design, where every predictor was a municipality-level aggregate:

| Attribute | Coverage | Distinct | Note |
|---|--:|--:|---|
| `district` | **100 %** | 6 | absent from the CBS extract entirely — only the budget file supplies it |
| `sector` | **100 %** | 5 | Jewish / Arab / Bedouin / Druze / Circassian |
| `supervision` | **100 %** | 3 | State / State-religious / Haredi |
| `legal_status` | **100 %** | 4 | Official / Recognised / Culturally-unique / Exempt |
| `education_stage` | **100 %** | 7 | structural control |
| `nurture_quintile` | 85.5 % | 5 | **school-level** disadvantage index (vs the *municipal* cluster) |
| `avg_class_size` | 91.1 % | — | pedagogically meaningful resourcing proxy |

`nurture_quintile` matters conceptually: it is a **school-level** socioeconomic
measure, so the project can now contrast *school* disadvantage against *municipal*
disadvantage — the core of this project's research question.

> ⚠️ **Direction — read this before interpreting any model.** This is the
> Ministry's *nurture* (טיפוח) index, so **HIGHER = MORE disadvantaged**. It is
> not a "quality" score. Verified against the outcomes: mean combined grade falls
> monotonically **85.3 → 74.1** from quintile 1 to quintile 5, so quintile 1 =
> *least* disadvantaged. The same note is in the `clean_text.py` docstring.
> Getting this backwards flips the sign of the headline finding in Step 5.
>
> The 85.5 % coverage above is over all **4,718 institutions in the raw budget
> file**. In the merged school-year modelling table the field is missing for
> **11.9 %** of rows — a different denominator, not a conflicting number.

### Columns excluded up front (verified all-zero, documented not hidden)
- `תקציב גפ"ן` (Gefen budget) — grand total = 0 across the entire workbook
- `שעות הדרכה` (guidance hours) — 0 % non-zero among our schools

### ⚠️ Data-provenance limitation — the budget file is a single-year snapshot

The budget workbook's `שנת לימודים` (school year) column carries one value for
every row: **`תשעה`** — the Hebrew calendar year **תשע"ה** (gematria
ת+ש+ע+ה = 400+300+70+5 = 775 → Hebrew year 5775), which spans **September 2014
– September 2015**, i.e. the **2014/15 Israeli school year**.

The Bagrut data, by contrast, spans **four** school years (2013–2016). Because
`semel` is a stable school identifier, we join this **one** budget snapshot onto
**every** year of that school's Bagrut records (Step 2) — meaning a school's
2013 and 2016 rows share the identical 2014/15 budget figures. This is a
deliberate, documented trade-off (the Ministry did not provide a multi-year
budget history), not an error: it lets us use the richer feature set at all, at
the cost of assuming each school's resourcing was reasonably stable across
2013–2016. **This is a stated limitation of the study.**

---

## 5. Plots (`graphs/`)

| File | Shows |
|---|---|
| `three_dataset_overview.png` | scale, join-key cardinality, columns carried forward |
| `budget_column_coverage.png` | usable-data audit of every budget column (teal ≥60 % · gold partial · red unusable) |
| `budget_school_profile.png` | the new school-level categoricals (sector, supervision, district, nurture quintile) |
| `ses_cluster_frequency.png` | CBS cluster distribution |
| `bagrut_target_missingness.png` | the 21 % grade suppression + proof it is **not** missing at random |

> **Plot labels are English.** matplotlib renders right-to-left Hebrew reversed
> (`יהודי` → `ידוהי`), so `config.yaml → display_labels` maps every Hebrew
> category to English **for plots only**; the stored data keeps the Hebrew values.

---

## 6. Key methodological decision — the target stays un-imputed

`bagrut_target_missingness.png` shows **21.4 %** of `grade` values are missing, and
that missingness is **not random**: suppressed cells have a median of **7 takers**
versus **28** where the grade is present. This is **privacy suppression of small
cohorts**. We therefore **never impute the target** — Step 3 aggregates over
observed cells only.

---

## 7. Step 1 verification checklist

- [x] Bagrut read with `utf-8-sig`; padded text columns trimmed; `city_norm` built.
- [x] CBS read at `header=10`; `..` → NaN; numeric columns coerced; `locality_norm` built.
- [x] Budget read despite the styles.xml colour error; totals row dropped; 25/25 columns resolved.
- [x] Budget keyed on `semel` (4,718 unique, 0 duplicates); nurture quintile parsed.
- [x] All-zero budget columns identified and excluded **with documentation**.
- [x] Budget file's single-year snapshot (2014/15) decoded and documented as a
      limitation (broadcast statically across the 2013–2016 Bagrut years).
- [x] Join feasibility confirmed for **both** keys (92.8 % name, 98.6 % semel).
- [x] 5/5 plots saved; Hebrew labels mapped to English for readability.
- [x] Three cached CSVs written (`utf-8-sig`).

**Status: Step 1 complete ✔**
