# ============================================================
# 09_merge_ex3.py
# EXERCISE 3 — Master Dataset Construction (Merge Only)
#
# PURPOSE:
#   Load all Exercise 3 raw data sources and merge them into
#   one clean master CSV (ex3_monthly.csv).
#
#   SCOPE — this script does ONLY the merge:
#     No energy_Rt, no demand_dummy  → script 10
#     No VAR shocks (eps_*)          → script 11
#     No Fama-MacBeth                → script 12
#
# INPUTS:
#   [A] Data/cleaned/ex2_monthly.csv
#         R_t, rho_t, rho_sb, macro controls from Ex. 1–2
#   [B] Data/raw/Exercise_3/stock_total_return.csv       s1–s5
#   [C] Data/raw/Exercise_3/reit_total_return.csv        re1–re3
#   [D] Data/raw/Exercise_3/intl_total_return.csv        in1–in3
#   [E] Data/raw/Exercise_3/corporate_total_return.csv   cp1–cp4
#   [F] Data/raw/Exercise_3/treasury_approx_returns.csv  dgs1–dgs20
#   [G] Data/raw/Exercise_3/commodity_total_return.csv   cm1–cm5
#   [H] Data/raw/Exercise_3/currency_carry_returns.csv   fx1–fx7
#   [I] Data/raw/Exercise_3/CPILFESL/CPIENGSL/CPIFABSL   pi MoM annualized
#   [J] Data/raw/Exercise_3/shiller_pd_ratio.csv         log(P/D)
#
# PORTFOLIO INVENTORY (35 total — no agency bonds):
#   Stocks      s1–s5   Ken French 5 Industries
#   REITs       re1–re3 FTSE NAREIT (Bloomberg)
#   Intl stocks in1–in3 MSCI N.America / Europe / Far East
#   Corp bonds  cp1–cp4 Bloomberg IG (1-3Y, 3-5Y, Broad, >10Y)
#   Treasuries  dgs1–dgs20 (7 maturities, FRED duration approx.)
#   Commodities cm1–cm5 GSCI equal-weighted sectors
#   FX carry    fx1–fx7 Verdelhan LRV (ends May 2021)
#
# ANALYSIS SAMPLE: May 2001 – June 2023  (266 months)
#   Start: May 2001 = first available month of LMPRP news indices
#          (Lumbanraja, Mouabbi, Passari & Rousset Planat 2025).
#          energy_Rt and demand_dummy have NaN for the first ~22 months
#          (rolling 24m window initialisation, min_periods=12).
#          These rows are excluded automatically by dropna() in script 12.
#          The FM effective sample = 244 months (after dropping NaN dummy).
#   End:   June 2023 = LMPRP news indices end (binding constraint).
#   FX:    available Jan 1999 – May 2021; NaN for 43 months after.
#
# OUTPUT:
#   Data/cleaned/ex3_monthly.csv   — 242 rows × ~52 columns
#   output/figures/Fig20_data_overview.png
#
# HOW TO RUN:
#   & C:/ProgramData/anaconda3/python.exe 09_merge_ex3.py
# ============================================================

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3 = os.path.join(BASE, "Data", "raw", "Exercise_3")
CLEAN   = os.path.join(BASE, "Data", "cleaned")

OUTPUT_CSV = os.path.join(CLEAN, "ex3_monthly.csv")

# ── SAMPLE BOUNDS ─────────────────────────────────────────────
SAMPLE_START = "2001-05"
SAMPLE_END   = "2023-06"
FULL_PERIOD  = pd.period_range("1999-01", "2024-12", freq="M")

print("=" * 65)
print("09_merge_ex3.py")
print("EXERCISE 3 — Master Dataset (Merge Only)")
print("=" * 65)
print(f"\n  Sample : {SAMPLE_START} -> {SAMPLE_END}  (266 months)")
print(f"  energy_Rt + demand_dummy : built in script 10")
print(f"  VAR shocks               : built in script 11")


# ============================================================
# HELPER — load CSV with monthly PeriodIndex
# ============================================================

def load_period_csv(path, date_col=None):
    df = pd.read_csv(path)
    if date_col is None:
        date_col = df.columns[0]
    try:
        df["period"] = pd.PeriodIndex(df[date_col], freq="M")
    except Exception:
        df["period"] = (pd.to_datetime(df[date_col], errors="coerce")
                          .dt.to_period("M"))
    return df.drop(columns=[date_col]).set_index("period")


# ============================================================
# [A] EXERCISE 2 BASE — R_t, macro controls
# ============================================================

print(f"\n{'─'*65}")
print("[A] ex2_monthly.csv  — R_t, rho_t, rho_sb, macro controls")
print(f"{'─'*65}")

ex2 = load_period_csv(os.path.join(CLEAN, "ex2_monthly.csv"),
                      date_col="date")

# Keep regime indicators + macro controls only.
# Drop sp500_ret, bond_ret, spxt_level, y10_decimal — those are
# superseded by the return series loaded below.
EX2_KEEP = [
    "R_t", "rho_t", "rho_sb",
    "inflation_yoy", "ugap", "policy_rate",
    "vix", "dollar_index", "term_spread", "indpro_growth",
]
EX2_KEEP = [c for c in EX2_KEEP if c in ex2.columns]
ex2 = ex2[EX2_KEEP]

print(f"  Range  : {ex2.index[0]} -> {ex2.index[-1]}  ({len(ex2)} obs)")
print(f"  Columns: {EX2_KEEP}")


# ============================================================
# [B–H] ASSET RETURNS
# ============================================================

print(f"\n{'─'*65}")
print("[B–H] Asset return files  (34 portfolios total)")
print(f"{'─'*65}\n")

ASSET_FILES = {
    "stock": (
        "stock_total_return.csv",
        ["ret_s1", "ret_s2", "ret_s3", "ret_s4", "ret_s5"],
        "Ken French 5 Industries",
    ),
    "reit": (
        "reit_total_return.csv",
        ["ret_re1", "ret_re2", "ret_re3"],
        "FTSE NAREIT (Bloomberg)",
    ),
    "intl": (
        "intl_total_return.csv",
        ["ret_in1", "ret_in2", "ret_in3"],
        "MSCI (Bloomberg)",
    ),
    "corp": (
        "corporate_total_return.csv",
        ["ret_cp1", "ret_cp2", "ret_cp3", "ret_cp4"],
        "Bloomberg IG Corporate",
    ),
    "treas": (
        "treasury_approx_returns.csv",
        ["ret_dgs1", "ret_dgs2", "ret_dgs3", "ret_dgs5",
         "ret_dgs7", "ret_dgs10", "ret_dgs20", "ret_dgs30"],
        "FRED duration approx.",
    ),
    "comm": (
        "commodity_total_return.csv",
        ["ret_cm1", "ret_cm2", "ret_cm3", "ret_cm4", "ret_cm5"],
        "GSCI equal-weighted",
    ),
    "fx": (
        "currency_carry_returns.csv",
        ["ret_fx1", "ret_fx2", "ret_fx3", "ret_fx4",
         "ret_fx5", "ret_fx6", "ret_fx7"],
        "Verdelhan LRV (ends May 2021)",
    ),
}

asset_dfs = {}
for key, (fname, cols, source) in ASSET_FILES.items():
    fpath = os.path.join(RAW_EX3, fname)
    if not os.path.exists(fpath):
        print(f"  !! MISSING  {fname}")
        continue
    df = load_period_csv(fpath)
    cols_ok   = [c for c in cols if c in df.columns]
    cols_miss = [c for c in cols if c not in df.columns]
    df = df[cols_ok]
    asset_dfs[key] = df
    n  = df.dropna().shape[0]
    f  = df.dropna().index[0]  if n > 0 else "N/A"
    l  = df.dropna().index[-1] if n > 0 else "N/A"
    miss_note = f"  [cols missing: {cols_miss}]" if cols_miss else ""
    print(f"  OK  {key:<6}  {fname:<42}  "
          f"{len(cols_ok):2d} cols  |  {n:3d} complete  "
          f"({f} -> {l}){miss_note}")


# ============================================================
# [I] INFLATION FACTORS — MoM annualized from raw FRED CPI levels
#
#
#   pi_t = (CPI_t / CPI_{t-1} - 1) × 12 × 100  [MoM annualized]
#
#   CHANGED from YoY: consistent with the VAR in script 11 which
#   uses MoM annualized inflation. The pi_core/food/energy columns
#   stored in ex3_monthly.csv now match the VAR shock scale.
#
# ============================================================

print(f"\n{'─'*65}")
print("[I] Inflation — MoM annualized from raw FRED CPI levels")
print(f"{'─'*65}\n")
print("  Formula: pi_t = (CPI_t / CPI_{t-1} - 1) x 12 x 100  [exact MoM annualized]\n")

def load_cpi_mom(fname, cpi_col, out_col):
    path = os.path.join(RAW_EX3, fname)
    df   = pd.read_csv(path)
    df["period"] = pd.PeriodIndex(df["observation_date"], freq="M")
    df   = df.set_index("period")[[cpi_col]]
    # CHANGED: MoM annualized (was YoY - consistent with VAR in script 11)
    df[out_col] = (df[cpi_col] / df[cpi_col].shift(1) - 1) * 12 * 100
    return df[[out_col]]

pi_core   = load_cpi_mom("CPILFESL.csv", "CPILFESL", "pi_core")
pi_energy = load_cpi_mom("CPIENGSL.csv", "CPIENGSL", "pi_energy")
pi_food   = load_cpi_mom("CPIFABSL.csv", "CPIFABSL", "pi_food")

for name, df in [("pi_core", pi_core),
                 ("pi_energy", pi_energy),
                 ("pi_food",   pi_food)]:
    s = df.iloc[:, 0].dropna()
    print(f"  {name:<12}  {str(s.index[0])} -> {str(s.index[-1])}  "
          f"mean={s.mean():+.2f}%  std={s.std():.2f}%  "
          f"[{s.min():.1f}%, {s.max():.1f}%]")


# ============================================================
# [J] SHILLER LOG(P/D)
# ============================================================

print(f"\n{'─'*65}")
print("[J] Shiller log(P/D)")
print(f"{'─'*65}")

shiller = load_period_csv(
    os.path.join(RAW_EX3, "shiller_pd_ratio.csv"),
    date_col="observation_date"
)[["pd_ratio"]]   # drop 'extrapolated' flag — not needed downstream

s_pd = shiller["pd_ratio"].dropna()
print(f"\n  Range : {str(s_pd.index[0])} -> {str(s_pd.index[-1])}")
print(f"  Stats : mean={s_pd.mean():.4f}  "
      f"[{s_pd.min():.4f}, {s_pd.max():.4f}]")


# ============================================================
# MERGE ON FULL PERIOD
# ============================================================

print(f"\n{'─'*65}")
print("MERGING on full period 1999-01 – 2024-12")
print(f"{'─'*65}")

master = pd.DataFrame(index=FULL_PERIOD)
master.index.name = "period"

master = master.join(ex2, how="left")
n_after_a = len(master.columns)

for key, df in asset_dfs.items():
    master = master.join(df, how="left")
n_after_bh = len(master.columns)

master = master.join(pi_core,   how="left")
master = master.join(pi_energy, how="left")
master = master.join(pi_food,   how="left")

master = master.join(shiller,   how="left")

print(f"\n  [A] macro + R_t       : {n_after_a} cols")
print(f"  [B-H] + asset returns : {n_after_bh} cols")
print(f"  [I-J] + pi + pd_ratio : {len(master.columns)} cols")
print(f"\n  Full range : {master.index[0]} -> {master.index[-1]}"
      f"  ({len(master)} obs)")


# ============================================================
# RESTRICT TO ANALYSIS SAMPLE
# ============================================================

print(f"\n{'─'*65}")
print(f"RESTRICTING to {SAMPLE_START} – {SAMPLE_END}")
print(f"{'─'*65}")

sample = master.loc[SAMPLE_START:SAMPLE_END].copy()

print(f"\n  Shape  : {sample.shape[0]} months × {sample.shape[1]} columns")
print(f"  Range  : {sample.index[0]} -> {sample.index[-1]}")

fx_cols = [c for c in sample.columns if c.startswith("ret_fx")]
fx_ok   = sample.loc[:"2021-05", fx_cols].dropna().shape[0]
print(f"\n  FX ({len(fx_cols)} cols): {fx_ok} complete months "
      f"({SAMPLE_START}–2021-05) | NaN after May 2021 (documented)")

print(f"\n  Missing value summary (columns with any NaN):")
miss = sample.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) == 0:
    print("    None.")
else:
    for col, n in miss.items():
        print(f"    {col:<28} {n:3d} missing")


# ============================================================
# SAVE
# ============================================================

print(f"\n{'─'*65}")
print("SAVING ex3_monthly.csv")
print(f"{'─'*65}")

out = sample.copy()
out.index = out.index.astype(str)
out.index.name = "date"
out.reset_index().to_csv(OUTPUT_CSV, index=False)

print(f"\n  File : {OUTPUT_CSV}")
print(f"  Rows : {len(out)}")
print(f"  Cols : {len(out.columns)}")

# Column inventory by group
groups = {
    "Macro + R_t"          : [c for c in sample.columns if c in EX2_KEEP],
    "Stocks   (s1–s5)"     : [c for c in sample.columns if c.startswith("ret_s")],
    "REITs    (re1–re3)"   : [c for c in sample.columns if c.startswith("ret_re")],
    "Intl     (in1–in3)"   : [c for c in sample.columns if c.startswith("ret_in")],
    "Corp     (cp1–cp4)"   : [c for c in sample.columns if c.startswith("ret_cp")],
    "Treasury (dgs1–20)"   : [c for c in sample.columns if c.startswith("ret_dgs")],
    "Commodity(cm1–cm5)"   : [c for c in sample.columns if c.startswith("ret_cm")],
    "FX carry (fx1–fx7)"   : [c for c in sample.columns if c.startswith("ret_fx")],
    "Inflation (pi_*)"     : [c for c in sample.columns if c.startswith("pi_")],
    "Valuation"            : (["pd_ratio"] if "pd_ratio" in sample.columns else []),
}
print(f"\n  Column inventory:")
n_ret = 0
for g, cols in groups.items():
    if cols:
        # count all return portfolios (any group whose cols start with ret_)
        if cols and cols[0].startswith("ret_"):
            n_ret += len(cols)
        print(f"    {g:<24} ({len(cols):2d})  {cols}")
print(f"\n  Total return portfolios : {n_ret}  (35 expected)")
print(f"\n  PENDING — to be added by subsequent scripts:")
print(f"    script 10 : energy_Rt, demand_dummy")
print(f"    script 11 : eps_core, eps_energy, eps_energy_D, eps_energy_S")


print(f"\n{'='*65}")
print("FINAL VERIFICATION")
print(f"{'='*65}\n")

ret_cols = [c for c in sample.columns if c.startswith("ret_")]

checks = [
    ("Rows = 266",
     len(sample) == 266,
     f"{len(sample)} rows"),
    ("Starts 2001-05",
     str(sample.index[0]) == "2001-05",
     str(sample.index[0])),
    ("Ends 2023-06",
     str(sample.index[-1]) == "2023-06",
     str(sample.index[-1])),
    ("35 return columns",
     len(ret_cols) == 35,
     f"{len(ret_cols)} found"),
    ("R_t NaN ≤ 25 (first 23m expected)",
     sample["R_t"].isna().sum() <= 25,
     f"{sample['R_t'].isna().sum()} missing (Rₜ needs 24m LMPRP rolling → NaN 2001-05 to 2003-03 normal)"),
    ("pi_core complete",
     sample["pi_core"].isna().sum() == 0,
     f"{sample['pi_core'].isna().sum()} missing"),
    ("pi_energy complete",
     sample["pi_energy"].isna().sum() == 0,
     f"{sample['pi_energy'].isna().sum()} missing"),
    ("pi_food complete",
     sample["pi_food"].isna().sum() == 0,
     f"{sample['pi_food'].isna().sum()} missing"),
    ("pd_ratio complete",
     sample["pd_ratio"].isna().sum() == 0,
     f"{sample['pd_ratio'].isna().sum()} missing"),
    ("Stocks s1-s5 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_s")]].isna().sum().sum() == 0,
     "OK"),
    ("REITs re1-re3 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_re")]].isna().sum().sum() == 0,
     "OK"),
    ("Intl in1-in3 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_in")]].isna().sum().sum() == 0,
     "OK"),
    ("Corp cp1-cp4 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_cp")]].isna().sum().sum() == 0,
     "OK"),
    ("Treasury dgs1-20 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_dgs")]].isna().sum().sum() == 0,
     "OK"),
    ("Commodity cm1-5 complete",
     sample[[c for c in sample.columns
             if c.startswith("ret_cm")]].isna().sum().sum() == 0,
     "OK"),
    ("FX complete to 2021-05",
     sample.loc[:"2021-05", "ret_fx1"].isna().sum() == 0,
     f"{sample.loc[:'2021-05', 'ret_fx1'].isna().sum()} missing"),
    ("R_t in [0, 1] (non-NaN)",
     sample["R_t"].dropna().between(0, 1).all(),
     f"[{sample['R_t'].dropna().min():.3f}, {sample['R_t'].dropna().max():.3f}]"),
]

all_ok = True
for label, passed, detail in checks:
    status = "OK  " if passed else "FAIL"
    if not passed:
        all_ok = False
    print(f"  {status}  {label:<30}  {detail}")

print()
if all_ok:
    print("  All checks passed.")
    print(f"\n  ex3_monthly.csv ready — {len(out)} rows × {len(out.columns)} cols")
    print(f"\n  Pipeline:")
    print(f"    10_build_energy_Rt.py        → energy_Rt, demand_dummy")
    print(f"    11_var_inflation_shocks.py   → eps_core, eps_energy, eps_energy_D/S")
    print(f"    12_fama_macbeth_ex3.py       → FM betas + prices of risk")
else:
    print("  Some checks FAILED — review above before proceeding.")

print(f"\n{'='*65}")
print("09_merge_ex3.py -- DONE")
print(f"{'='*65}")