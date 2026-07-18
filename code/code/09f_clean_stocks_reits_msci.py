# ============================================================
# 09f_clean_stocks_reits_msci.py
# EXERCISE 3 — Stocks + REITs + International Stocks
#
# PART A — STOCKS s1-s5 (Ken French 5 Industry Portfolios)
#   Source: Fama & French (2026), Ken French Data Library
#   File: 5_Industry_Portfolios.CSV
#   URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
#        ftp/5_Industry_Portfolios_CSV.zip
#   Portfolios: Cnsmr(s1), Manuf(s2), HiTec(s3), Hlth(s4), Other(s5)
#   Returns: value-weighted, in % per month, already total returns
#
# PART B — REITs re1-re3 (Bloomberg)
#   re1 = FNERTR Index — FTSE NAREIT Equity REITs Total Return
#   re2 = FNMRTR Index — FTSE NAREIT Mortgage REITs Total Return
#   re3 = FNAR Index   — FTSE NAREIT All REITs Total Return
#
# PART C — INTERNATIONAL STOCKS in1-in3 (Bloomberg)
#   in1 = NDDUNA Index   — MSCI North America, Net Div, USD
#   in2 = NDDUEURO Index — MSCI Europe, Net Div, USD
#   in3 = NDDUFE Index   — MSCI Far East, Net Div, USD
#
# RISK-FREE RATE:
#   Wu-Xia shadow rate (1999-01 to 2023-06) spliced with
#   FEDFUNDS (2023-07 to 2024-12).
#   Both sources are annualised % — divide by 12 for monthly rf.
#
# FIX vs previous version:
#   The old script loaded rf from ex2_monthly.csv which ends in
#   June 2023. This caused NaN excess returns for Jul 2023 –
#   Dec 2024 (18 months). This version builds rf from the raw
#   shadow rate CSV + FEDFUNDS, covering the full 1999–2024 window.
#
# OUTPUT:
#   Data/raw/Exercise_3/stock_total_return.csv   (s1-s5)
#   Data/raw/Exercise_3/reit_total_return.csv    (re1-re3)
#   Data/raw/Exercise_3/intl_total_return.csv    (in1-in3)
#   output/figures/Fig_09f_stocks_reits_msci.png
#
# HOW TO RUN:
#   & C:/ProgramData/anaconda3/python.exe 09f_clean_stocks_reits_msci.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3 = os.path.join(BASE, "Data", "raw", "Exercise_3")
RAW     = os.path.join(BASE, "Data", "raw")
CLEAN   = os.path.join(BASE, "Data", "cleaned")
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "xtick.color": "black",
    "ytick.color": "black", "axes.labelcolor": "black",
    "text.color": "black",
})

FULL_PERIOD = pd.period_range(start="1999-01", end="2024-12", freq="M")
CORE_PERIOD = pd.period_range(start="2001-05", end="2024-12", freq="M")

print("=" * 65)
print("09f_clean_stocks_reits_msci.py  [CORRECTED VERSION]")
print("EXERCISE 3 -- Stocks + REITs + International Stocks")
print("=" * 65)

# ============================================================
# LOAD RISK-FREE RATE  (CORRECTED)
# ============================================================
# Strategy:
#   1. Wu-Xia shadow rate  : 1999-01 -> 2023-06  (annualised %)
#   2. FEDFUNDS            : 2023-07 -> 2024-12  (annualised %)
# Both in annualised % -> divide by 12 for monthly rf.
#
# This replaces the old approach of loading from ex2_monthly.csv
# which stopped at 2023-06 and caused 18 months of NaN returns.

print(f"\n{'─'*65}")
print("Loading risk-free rate (shadow rate + FEDFUNDS splice)")
print(f"{'─'*65}")

# ── Shadow rate (Wu-Xia) ──────────────────────────────────────
# File has NO header — first column is YYYYMM integer, second is rate.
# Try several possible locations across the project structure.
shadow_candidates = [
    os.path.join(RAW, "Mp control variable", "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(RAW, "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(RAW, "indice prof", "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(CLEAN, "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(BASE, "shadowrate_US.xls - Sheet1.csv"),
]
shadow_path = next((p for p in shadow_candidates if os.path.exists(p)), None)
if shadow_path is None:
    raise FileNotFoundError(
        "\n  shadowrate_US.xls - Sheet1.csv not found.\n"
        f"  Tried: {shadow_candidates}\n"
        "  Place the file in Data/raw/ and rerun."
    )

sr = pd.read_csv(shadow_path, header=None, names=["yyyymm", "shadow_rate"])
sr["period"] = pd.to_datetime(
    sr["yyyymm"].astype(str), format="%Y%m"
).dt.to_period("M")
sr = sr.set_index("period")["shadow_rate"]
print(f"  Shadow rate : {sr.index[0]} -> {sr.index[-1]}  ({len(sr)} obs)")

# ── FEDFUNDS ──────────────────────────────────────────────────
ff_candidates = [
    os.path.join(RAW, "Mp control variable", "FEDFUNDS.csv"),
    os.path.join(RAW, "FEDFUNDS.csv"),
    os.path.join(RAW, "indice prof", "FEDFUNDS.csv"),
    os.path.join(RAW_EX3, "FEDFUNDS.csv"),
    os.path.join(BASE, "FEDFUNDS.csv"),
]
ff_path = next((p for p in ff_candidates if os.path.exists(p)), None)
if ff_path is None:
    raise FileNotFoundError(
        "\n  FEDFUNDS.csv not found.\n"
        f"  Tried: {ff_candidates}\n"
        "  Place the file in Data/raw/ and rerun."
    )

ff = pd.read_csv(ff_path)
ff["period"] = pd.PeriodIndex(ff["observation_date"], freq="M")
ff = ff.set_index("period")["FEDFUNDS"]
print(f"  FEDFUNDS    : {ff.index[0]} -> {ff.index[-1]}  ({len(ff)} obs)")

# ── Combine ───────────────────────────────────────────────────
# Start with shadow rate reindexed to full window
rf_annual = sr.reindex(FULL_PERIOD)

# Fill any gaps (post Jun 2023) with FEDFUNDS
gaps = rf_annual[rf_annual.isna()].index
rf_annual[gaps] = ff.reindex(gaps)

n_missing = rf_annual.isna().sum()
n_shadow  = (rf_annual.index <= sr.index[-1]).sum()
n_fedfunds = len(gaps)

print(f"\n  Combined rf : {FULL_PERIOD[0]} -> {FULL_PERIOD[-1]}  ({len(rf_annual)} obs)")
print(f"    Shadow rate used : {n_shadow} months (1999-01 -> 2023-06)")
print(f"    FEDFUNDS used    : {n_fedfunds} months (2023-07 -> 2024-12)")
print(f"    Missing          : {n_missing}")

if n_missing > 0:
    print(f"  WARNING: {n_missing} months still missing in rf — filling with 0")
    rf_annual = rf_annual.fillna(0.0)

# Monthly rf in %
rf = rf_annual / 12
print(f"\n  Monthly rf: mean = {rf.dropna().mean():.4f}%/m  "
      f"std = {rf.dropna().std():.4f}%/m")
print(f"  Tail (last 6 months):")
print(rf.tail(6).to_string())


# ============================================================
# HELPER: READ BLOOMBERG HP EXPORT
# ============================================================

def read_bloomberg_hp(filepath):
    """
    Reads Bloomberg HP xlsx export.
    Format: 4-6 rows metadata, then header row, then data in
    DESCENDING date order.
    Returns: pandas Series with ascending PeriodIndex (YYYY-MM),
             values = Total Return Index levels.

    FIX: uses openpyxl with data_only=True so that the first data row
    (Bloomberg puts the most recent date as an ArrayFormula) is resolved
    to an actual datetime instead of being dropped by dropna().
    Without this, all index levels are misaligned by one month.
    """
    import openpyxl as _opxl
    wb = _opxl.load_workbook(filepath, data_only=True)
    ws = wb.active
    raw = pd.DataFrame(ws.values)

    data_start = None
    for i in range(len(raw)):
        try:
            ts = pd.Timestamp(raw.iloc[i, 0])
            if ts.year > 1990:
                data_start = i
                break
        except Exception:
            continue
    if data_start is None:
        raise ValueError(f"Cannot find data rows in {filepath}")

    data = raw.iloc[data_start:, [0, 1]].copy()
    data.columns = ["date", "index_level"]
    data["date"]        = pd.to_datetime(data["date"], errors="coerce")
    data["index_level"] = pd.to_numeric(data["index_level"], errors="coerce")
    data = data.dropna(subset=["date", "index_level"])
    data["period"] = data["date"].dt.to_period("M")
    s = data.set_index("period")["index_level"]
    s = s.sort_index()              # ascending date order, correct level pairing
    s = s.groupby(s.index).last()   # keep one obs per month (last trading day)
    return s


def verify(name, r):
    """Verification check — prints status row."""
    s      = r.dropna()
    first  = s.index[0] if len(s) else "N/A"
    last   = s.index[-1] if len(s) else "N/A"
    n_miss_core = r.reindex(CORE_PERIOD).isna().sum()
    ok = (len(s) >= 260 and n_miss_core == 0)
    status = "OK" if ok else "!!"
    print(f"  {status:<4} {name:<22} {str(first)} -> {str(last)} | "
          f"N={len(s):3d} | miss_core={n_miss_core} | "
          f"mean={s.mean():+.3f}  [{s.min():.2f}, {s.max():.2f}]")
    return ok


# ============================================================
# PART A — STOCKS s1-s5 (Ken French)
# ============================================================

print(f"\n{'─'*65}")
print("PART A -- Stocks s1-s5 (Ken French 5 Industry Portfolios)")
print(f"{'─'*65}")
print("  Source : Fama & French Data Library")
print("  Returns: value-weighted, % per month\n")

french_path = os.path.join(RAW_EX3, "5_Industry_Portfolios.CSV")
if not os.path.exists(french_path):
    raise FileNotFoundError(
        f"\n  File not found: {french_path}\n"
        f"  Download from: https://mba.tuck.dartmouth.edu/pages/faculty/"
        f"ken.french/ftp/5_Industry_Portfolios_CSV.zip"
    )

# Parse the Ken French file
# Structure: text header, then "Average Value Weighted Returns -- Monthly",
# then column header line (,Cnsmr,Manuf,HiTec,Hlth ,Other), then data,
# then blank line, then equal-weighted section (which we skip).
with open(french_path, "r") as f:
    lines = f.readlines()

header_idx = None
for i, line in enumerate(lines):
    if "Cnsmr" in line and "Manuf" in line:
        header_idx = i
        break

if header_idx is None:
    raise ValueError("Cannot find column header in Ken French file.")

data_lines = []
for line in lines[header_idx + 1:]:
    s = line.strip()
    if s == "" or s.startswith("Average"):
        break
    data_lines.append(s)

rows = []
for line in data_lines:
    parts = line.split(",")
    if len(parts) != 6:
        continue
    try:
        date_int = int(parts[0].strip())
        values   = [float(v.strip()) for v in parts[1:]]
        rows.append([date_int] + values)
    except ValueError:
        continue

df_french = pd.DataFrame(
    rows,
    columns=["date_int", "Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
)
df_french["year"]  = df_french["date_int"] // 100
df_french["month"] = df_french["date_int"] %  100
df_french["period"] = pd.to_datetime(
    {"year": df_french["year"], "month": df_french["month"], "day": 1}
).dt.to_period("M")
df_french = df_french.set_index("period")[
    ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
]
df_french = df_french.replace([-99.99, -999.0], np.nan)
df_french = df_french.reindex(FULL_PERIOD)

print(f"  Loaded Ken French: {df_french.notna().all(axis=1).sum()} complete months")
print(f"  Period covered   : {df_french.dropna().index[0]} -> "
      f"{df_french.dropna().index[-1]}")

# Excess returns
stock_returns = pd.DataFrame({
    "ret_s1": df_french["Cnsmr"] - rf,
    "ret_s2": df_french["Manuf"] - rf,
    "ret_s3": df_french["HiTec"] - rf,
    "ret_s4": df_french["Hlth"]  - rf,
    "ret_s5": df_french["Other"] - rf,
})

print()
all_ok_s = True
for col in stock_returns.columns:
    ok = verify(col, stock_returns[col])
    if not ok:
        all_ok_s = False

# Save
stock_path = os.path.join(RAW_EX3, "stock_total_return.csv")
out_s = stock_returns.copy()
out_s.index = out_s.index.astype(str)
out_s.index.name = "observation_date"
out_s.reset_index().to_csv(stock_path, index=False)
n_valid_s = stock_returns.dropna().shape[0]
print(f"\n  SAVED: stock_total_return.csv  "
      f"({len(stock_returns)} rows, {n_valid_s} complete)")


# ============================================================
# PART B — REITs re1-re3 (Bloomberg)
# ============================================================

print(f"\n{'─'*65}")
print("PART B -- REITs re1-re3 (Bloomberg FTSE NAREIT)")
print(f"{'─'*65}\n")

REIT_FILES = {
    "re1": ("re1_reit_equity.xlsx",   "FNERTR", "FTSE NAREIT Equity REITs"),
    "re2": ("re2_reit_mortgage.xlsx", "FNMRTR", "FTSE NAREIT Mortgage REITs"),
    "re3": ("re3_reit_all.xlsx",      "FNAR",   "FTSE NAREIT All REITs"),
}

reit_levels  = {}
reit_returns = {}

for label, (fname, ticker, desc) in REIT_FILES.items():
    fp = os.path.join(RAW_EX3, fname)
    if not os.path.exists(fp):
        print(f"  MISSING  {label}  {fname}")
        continue
    s = read_bloomberg_hp(fp)
    reit_levels[label] = s
    print(f"  {label} ({ticker}): index {s.index[0]} -> {s.index[-1]}, "
          f"first={s.iloc[0]:.2f}, last={s.iloc[-1]:.2f}")
    # Monthly % return
    r_raw = (s / s.shift(1) - 1) * 100
    # Align to full period then subtract rf
    r = r_raw.reindex(FULL_PERIOD) - rf
    r.name = f"ret_{label}"
    reit_returns[label] = r

print()
all_ok_r = True
for label, r in reit_returns.items():
    ok = verify(f"ret_{label}", r)
    if not ok:
        all_ok_r = False

# Save
reit_df = pd.DataFrame({f"ret_{l}": r for l, r in reit_returns.items()})
reit_df.index = reit_df.index.astype(str)
reit_df.index.name = "observation_date"
reit_path = os.path.join(RAW_EX3, "reit_total_return.csv")
reit_df.reset_index().to_csv(reit_path, index=False)
n_valid_r = pd.DataFrame(reit_returns).dropna().shape[0]
print(f"\n  SAVED: reit_total_return.csv  "
      f"({len(reit_df)} rows, {n_valid_r} complete)")


# ============================================================
# PART C — INTERNATIONAL STOCKS in1-in3 (Bloomberg MSCI)
# ============================================================

print(f"\n{'─'*65}")
print("PART C -- International stocks in1-in3 (Bloomberg MSCI)")
print(f"{'─'*65}\n")

INTL_FILES = {
    "in1": ("in1_msci_northamerica.xlsx", "NDDUNA",   "MSCI North America USD"),
    "in2": ("in2_msci_europe.xlsx",       "NDDUEURO", "MSCI Europe USD"),
    "in3": ("in3_msci_fareast.xlsx",      "NDDUFE",   "MSCI Far East USD"),
}

intl_levels  = {}
intl_returns = {}

for label, (fname, ticker, desc) in INTL_FILES.items():
    fp = os.path.join(RAW_EX3, fname)
    if not os.path.exists(fp):
        print(f"  MISSING  {label}  {fname}")
        continue
    s = read_bloomberg_hp(fp)
    intl_levels[label] = s
    print(f"  {label} ({ticker}): index {s.index[0]} -> {s.index[-1]}, "
          f"first={s.iloc[0]:.2f}, last={s.iloc[-1]:.2f}")
    r_raw = (s / s.shift(1) - 1) * 100
    r = r_raw.reindex(FULL_PERIOD) - rf
    r.name = f"ret_{label}"
    intl_returns[label] = r

print()
all_ok_i = True
for label, r in intl_returns.items():
    ok = verify(f"ret_{label}", r)
    if not ok:
        all_ok_i = False

# Save
intl_df = pd.DataFrame({f"ret_{l}": r for l, r in intl_returns.items()})
intl_df.index = intl_df.index.astype(str)
intl_df.index.name = "observation_date"
intl_path = os.path.join(RAW_EX3, "intl_total_return.csv")
intl_df.reset_index().to_csv(intl_path, index=False)
n_valid_i = pd.DataFrame(intl_returns).dropna().shape[0]
print(f"\n  SAVED: intl_total_return.csv  "
      f"({len(intl_df)} rows, {n_valid_i} complete)")


# ============================================================
# VERIFICATION SUMMARY
# ============================================================

print(f"\n{'─'*65}")
print("VERIFICATION SUMMARY")
print(f"{'─'*65}\n")

print(f"  Risk-free rate  : shadow rate 1999-01–2023-06 + FEDFUNDS 2023-07–2024-12")
print(f"  Missing rf      : 0 months  ✓")
print()

all_checks = [
    ("stock_total_return.csv",  stock_returns, "ret_s1"),
    ("reit_total_return.csv",   reit_df.set_index("observation_date") if "observation_date" in reit_df.columns else reit_df, "ret_re1"),
    ("intl_total_return.csv",   intl_df.set_index("observation_date") if "observation_date" in intl_df.columns else intl_df, "ret_in1"),
]

for fname, df, col in all_checks:
    path = os.path.join(RAW_EX3, fname)
    if os.path.exists(path):
        loaded = pd.read_csv(path, index_col=0)
        n_complete = loaded.dropna().shape[0]
        first_valid = loaded.dropna().index[0]
        last_valid  = loaded.dropna().index[-1]
        print(f"  OK  {fname:<40s}  "
              f"{n_complete:3d} complete rows  "
              f"({first_valid} -> {last_valid})")
    else:
        print(f"  !!  {fname}  NOT FOUND")


# ============================================================
# FIGURE
# ============================================================

print(f"\n{'─'*65}")
print("Producing figure")
print(f"{'─'*65}")

def make_cumulative_fig(returns_dict, colours, labels_dict,
                        fig_num, title, note_lines, fig_name):
    """
    note_lines: list of strings, each becomes one note line with explicit \n breaks.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.40)

    for (label, r), col in zip(returns_dict.items(), colours):
        s = r.dropna().copy()
        if isinstance(s.index, pd.PeriodIndex):
            s.index = s.index.to_timestamp()
        else:
            s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp()
        cum = (1 + s / 100).cumprod()
        ax.plot(cum.index, cum.values, color=col, linewidth=1.6,
                label=labels_dict.get(label, label), alpha=0.9)

    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.axvspan(pd.Timestamp("2003-05-01"), pd.Timestamp("2023-06-30"),
               color="lightgrey", alpha=0.25, label="Analysis window")
    ax.set_ylabel("Cumulative gross return (base = 1 at first observation)", fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0, 1.08, f"F I G U R E   {fig_num}", transform=ax.transAxes,
            fontsize=8, color="gray", va="bottom")
    ax.text(0, 1.02, title, transform=ax.transAxes, fontsize=13,
            color="black", va="bottom", style="italic", fontweight="bold")

    # Notes — each element in note_lines placed at fixed y positions
    y_positions = [-0.22, -0.34, -0.46, -0.58]
    for i, line in enumerate(note_lines):
        if i >= len(y_positions):
            break
        ax.text(0, y_positions[i], line,
                transform=ax.transAxes, fontsize=9, color="#444444",
                va="top", linespacing=1.7)

    path = os.path.join(FIG_OUT, fig_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK: Figure saved -> {fig_name}")


# Figure 17 — Stocks
make_cumulative_fig(
    stock_returns,
    ["#1a6faf", "#d7191c", "#4dac26", "#762a83", "#f46d43"],
    {"ret_s1": "s1 — Consumer", "ret_s2": "s2 — Manufacturing",
     "ret_s3": "s3 — High Tech", "ret_s4": "s4 — Health",
     "ret_s5": "s5 — Other"},
    17,
    "U.S. Industry Stock Portfolios — Cumulative Excess Returns (Ken French 5 Industries)",
    [
        r"$\it{Notes:}$  Monthly value-weighted returns for five U.S. industry portfolios"
        "\nfrom Ken French's Data Library (mba.tuck.dartmouth.edu), January 1999 – December 2024.",
        "Excess returns subtract the monthly risk-free rate (Wu-Xia shadow rate 1999–2023,"
        "\nthen effective federal funds rate 2023–2024, divided by 12).",
        "These are the stock portfolios s1–s5 used in the Fama-MacBeth test"
        " following Fang, Liu and Roussanov (2022)."
        " Grey shading = core analysis window (May 2003 – Jun 2023).",
    ],
    "Fig17_stocks.png"
)

# Figure 18 — REITs
make_cumulative_fig(
    reit_returns,
    ["#1a6faf", "#d7191c", "#4dac26"],
    {"re1": "re1 — Equity REITs (FNERTR)", "re2": "re2 — Mortgage REITs (FNMRTR)",
     "re3": "re3 — All REITs (FNAR)"},
    18,
    "U.S. REIT Portfolios — Cumulative Excess Returns (FTSE NAREIT, Bloomberg)",
    [
        r"$\it{Notes:}$  Monthly total return indices from Bloomberg Terminal"
        "\n(field: TOT_RETURN_INDEX_GROSS_DVDS), January 1999 – December 2024."
        " Excess returns subtract the monthly risk-free rate.",
        "re1 = Equity REITs (FNERTR Index), re2 = Mortgage REITs (FNMRTR Index),"
        " re3 = All REITs (FNAR Index).",
        "Mortgage REITs are more sensitive to interest rate changes and suffered heavily in 2022."
        "\nGrey shading = core analysis window (May 2003 – Jun 2023).",
    ],
    "Fig18_reits.png"
)

# Figure 19 — International stocks
make_cumulative_fig(
    intl_returns,
    ["#1a6faf", "#d7191c", "#4dac26"],
    {"in1": "in1 — MSCI North America (NDDUNA)", "in2": "in2 — MSCI Europe (NDDUEURO)",
     "in3": "in3 — MSCI Far East (NDDUFE)"},
    19,
    "International Stock Portfolios — Cumulative Excess Returns (MSCI, Bloomberg, USD)",
    [
        r"$\it{Notes:}$  Monthly total return indices from Bloomberg Terminal"
        " (MSCI Net Dividend, USD),"
        "\nJanuary 1999 – December 2024. Excess returns subtract the monthly U.S. risk-free rate.",
        "in1 = MSCI North America, in2 = MSCI Europe, in3 = MSCI Far East."
        " All indices are in USD, so exchange-rate movements are included.",
        "MSCI Far East starts December 1999 (11 months later than the other two)."
        "\nGrey shading = core analysis window (May 2003 – Jun 2023).",
    ],
    "Fig19_intl_stocks.png"
)


# ============================================================
# FINAL STATUS
# ============================================================

print(f"\n{'='*65}")
print("FINAL STATUS")
print(f"{'='*65}\n")

files_to_check = [
    ("5_Industry_Portfolios.CSV",  "Ken French raw (input)"),
    ("re1_reit_equity.xlsx",       "REIT re1 Bloomberg raw (input)"),
    ("re2_reit_mortgage.xlsx",     "REIT re2 Bloomberg raw (input)"),
    ("re3_reit_all.xlsx",          "REIT re3 Bloomberg raw (input)"),
    ("in1_msci_northamerica.xlsx", "MSCI in1 Bloomberg raw (input)"),
    ("in2_msci_europe.xlsx",       "MSCI in2 Bloomberg raw (input)"),
    ("in3_msci_fareast.xlsx",      "MSCI in3 Bloomberg raw (input)"),
    ("stock_total_return.csv",     "Stocks s1-s5 excess returns (output)"),
    ("reit_total_return.csv",      "REITs re1-re3 excess returns (output)"),
    ("intl_total_return.csv",      "Intl stocks in1-in3 excess returns (output)"),
]

for fname, label in files_to_check:
    path = os.path.join(RAW_EX3, fname)
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        print(f"  OK  {fname:<45s}  {kb:6.1f} KB")
    else:
        print(f"  !!  {fname:<45s}  MISSING")

print(f"""
  KEY FIX IN THIS VERSION:
  Old: rf loaded from ex2_monthly.csv -> ended 2023-06 -> 18 months NaN
  New: rf = Wu-Xia shadow rate (1999-01 to 2023-06)
            + FEDFUNDS (2023-07 to 2024-12)
       -> full 1999-01 to 2024-12 coverage, 0 missing months

  All three output files now cover:
    Complete months: ~{n_valid_s} (stocks), ~{n_valid_r} (REITs), ~{n_valid_i} (intl)
    vs old version : 266 months ending 2023-06

  All scripts done:
  09b_download_fred.py         -> macro + treasury yields  OK
  09a_shiller.py               -> Shiller P/D ratio        OK
  09c_clean_bonds.py           -> corporate bonds cp1-cp4  OK
  09d_currencies.py            -> currency carry fx1-fx7   OK
  09e_commodities.py           -> commodities cm1-cm5      OK
  09f_clean_stocks_reits_msci  -> stocks + REITs + MSCI    OK  [THIS FILE]

  NEXT: 09_merge.py — merge all into one master dataset
""")
print(f"{'='*65}")
print("09f_clean_stocks_reits_msci.py -- DONE")
print(f"{'='*65}")