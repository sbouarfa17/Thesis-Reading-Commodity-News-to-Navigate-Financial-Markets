# ============================================================
# 09c_clean_bonds.py
# EXERCISE 3 — Corporate Bonds: Bloomberg xlsx -> Clean CSV
#
# SOURCE:
#   Bloomberg Terminal — Total Return Index (Gross Dividends)
#   Field RT116, Monthly, Jan 1999 – Dec 2024
#
# PORTFOLIOS:
#   cp1 = LT01TRUU Index  — Bloomberg 1-3Y US Corp IG
#   cp2 = LT03TRUU Index  — Bloomberg 3-5Y US Corp IG
#   cp3 = LUACTRUU Index  — Bloomberg US Corp IG Broad
#   cp4 = LD07TRUU Index  — Bloomberg Long US Corp (>10Y)
#
# NOTE ON AGENCY BONDS:
#   Maturity-sorted ICE BofA Agency Total Return indices are
#   not freely accessible (FRED restricts to 2023+, Bloomberg
#   licence at Dauphine does not include them). We use 4
#   Bloomberg Corporate IG portfolios spanning short to long
#   maturities. This gives equivalent beta dispersion across
#   maturities for the Fama-MacBeth test. Documented in thesis.
#
# WHAT THIS SCRIPT DOES:
#   Step 1 — Reads each Bloomberg xlsx (format: header rows + data
#             in descending date order)
#   Step 2 — Converts Total Return Index levels to monthly % returns
#             r_t = (Index_t / Index_{t-1} - 1) x 100
#   Step 3 — Aligns all series to Jan 1999 – Dec 2024
#   Step 4 — Verification + figure + saves corporate_total_return.csv
#
# INPUT  : Data/raw/Exercise_3/cp1_corp_1_3Y.xlsx
#          Data/raw/Exercise_3/cp2_corp_3_5Y.xlsx
#          Data/raw/Exercise_3/cp3_corp_broad_IG.xlsx
#          Data/raw/Exercise_3/cp4_corp_long.xlsx
# OUTPUT : Data/raw/Exercise_3/corporate_total_return.csv
#          output/figures/Fig_09c_bonds.png
#
# HOW TO RUN:
#   & C:/ProgramData/anaconda3/python.exe 09c_clean_bonds.py
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
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

OUTPUT_CSV = os.path.join(RAW_EX3, "corporate_total_return.csv")
OUTPUT_FIG = os.path.join(FIG_OUT,  "Fig14_corporate_bonds.png")

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
CORE_PERIOD = pd.period_range(start="2001-05", end="2024-06", freq="M")

print("=" * 65)
print("09c_clean_bonds.py")
print("EXERCISE 3 -- Corporate Bonds: Bloomberg xlsx -> CSV")
print("=" * 65)

# ============================================================
# INPUT FILE DEFINITIONS
# ============================================================

BOND_FILES = {
    "cp1": ("cp1_corp_1_3Y.xlsx",    "LT01TRUU", "Bloomberg 1-3Y Corp IG  [cp1]"),
    "cp2": ("cp2_corp_3_5Y.xlsx",    "LT03TRUU", "Bloomberg 3-5Y Corp IG  [cp2]"),
    "cp3": ("cp3_corp_broad_IG.xlsx","LUACTRUU", "Bloomberg US Corp Broad  [cp3]"),
    "cp4": ("cp4_corp_long.xlsx",    "LD07TRUU", "Bloomberg Long Corp >10Y [cp4]"),
}

# ============================================================
# HELPER: READ BLOOMBERG HP EXPORT
# ============================================================

def read_bloomberg_hp(filepath):
    """
    Reads a Bloomberg HP export xlsx file.

    Bloomberg HP format:
      Row 0 : Security  | [ticker]   | NaN
      Row 1 : Start Date| [date]     | NaN
      Row 2 : End Date  | [date]     | NaN
      Row 3 : Period    | M          | NaN
      Row 4 : NaN       | NaN        | NaN
      Row 5 : Date      | TOT_RETURN_INDEX_GROSS_DVDS | CHG_PCT_1D
      Row 6+: data rows in DESCENDING order (newest first)

    Returns: pandas Series, PeriodIndex (YYYY-MM), ascending,
             values = Total Return Index level.
    """
    raw = pd.read_excel(filepath, sheet_name=0,
                        engine="openpyxl", header=None)

    # Find first data row (first row where col 0 is a datetime)
    data_start = None
    for i in range(len(raw)):
        val = raw.iloc[i, 0]
        try:
            pd.Timestamp(val)
            if isinstance(val, (pd.Timestamp,)) or hasattr(val, 'year'):
                data_start = i
                break
        except Exception:
            continue

    if data_start is None:
        raise ValueError(f"Cannot find data rows in {filepath}")

    # Extract date + index level
    data = raw.iloc[data_start:, [0, 1]].copy()
    data.columns = ["date", "index_level"]
    data["date"]        = pd.to_datetime(data["date"], errors="coerce")
    data["index_level"] = pd.to_numeric(data["index_level"], errors="coerce")
    data = data.dropna(subset=["date", "index_level"])

    # Convert to monthly PeriodIndex
    data["period"] = data["date"].dt.to_period("M")
    s = data.set_index("period")["index_level"]

    # Sort ascending + keep one obs per month (last trading day)
    s = s.sort_index()
    s = s.groupby(s.index).last()

    return s


# ============================================================
# STEP 1 — READ ALL FILES
# ============================================================

print(f"\n{'─'*65}")
print("STEP 1 -- Reading Bloomberg xlsx files")
print(f"{'─'*65}\n")

index_levels = {}
failed       = []

for label, (fname, ticker, description) in BOND_FILES.items():
    filepath = os.path.join(RAW_EX3, fname)

    if not os.path.exists(filepath):
        print(f"  MISSING  {label}  {fname}")
        failed.append(label)
        continue

    try:
        s = read_bloomberg_hp(filepath)
        index_levels[label] = s
        print(f"  OK  {label}  {ticker:<12s} | "
              f"{str(s.index[0])} -> {str(s.index[-1])} | "
              f"{len(s)} obs | "
              f"first={s.iloc[0]:.2f}  last={s.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  FAIL  {label}  {fname} | {e}")
        failed.append(label)

if failed:
    print(f"\n  WARNING: {len(failed)} files missing: {failed}")


# ============================================================
# STEP 2 — INDEX LEVELS -> MONTHLY % RETURNS
# ============================================================
# r_t = (Index_t / Index_{t-1} - 1) x 100
# The Total Return Index already includes coupon reinvestment.
# This formula gives the monthly total return in percent.

print(f"\n{'─'*65}")
print("STEP 2 -- Index levels -> monthly % returns")
print(f"{'─'*65}")
print("  r_t = (Index_t / Index_t-1 - 1) x 100\n")

monthly_returns = {}

for label, idx in index_levels.items():
    r = (idx / idx.shift(1) - 1) * 100
    r.name = f"ret_{label}"
    monthly_returns[label] = r
    s = r.dropna()
    print(f"  {label}  {BOND_FILES[label][1]:<12s} | "
          f"mean={s.mean():+.3f}%/m  std={s.std():.3f}%/m  "
          f"[{s.min():.3f}%, {s.max():.3f}%]")


# ============================================================
# STEP 3 — ALIGN TO FULL PERIOD (1999-01 to 2024-12)
# ============================================================

print(f"\n{'─'*65}")
print("STEP 3 -- Aligning to Jan 1999 – Dec 2024")
print(f"{'─'*65}\n")

aligned = {}
for label, r in monthly_returns.items():
    r_al = r.reindex(FULL_PERIOD)
    aligned[label] = r_al
    n_miss = r_al.isna().sum()
    first  = r_al.first_valid_index()
    print(f"  {label}  first valid: {first}  |  total missing: {n_miss}")


# ============================================================
# STEP 4 — SAVE CSV
# ============================================================

print(f"\n{'─'*65}")
print("STEP 4 -- Saving corporate_total_return.csv")
print(f"{'─'*65}")

df_out = pd.DataFrame({f"ret_{lbl}": s for lbl, s in aligned.items()})
df_out.index.name = "observation_date"
df_out.index = df_out.index.astype(str)
df_out.reset_index().to_csv(OUTPUT_CSV, index=False)

print(f"\n  File    : {OUTPUT_CSV}")
print(f"  Rows    : {len(df_out)}")
print(f"  Columns : {list(df_out.columns)}")
print(f"  Units   : % per month (total return, gross dividends)")
print(f"\n  First 3 rows:")
print(df_out.head(3).round(4).to_string())
print(f"\n  Last 3 rows:")
print(df_out.tail(3).round(4).to_string())


# ============================================================
# STEP 5 — VERIFICATION
# ============================================================

print(f"\n{'─'*65}")
print("STEP 5 -- Verification checks")
print(f"{'─'*65}\n")

checks_pass = True
for label, r in aligned.items():
    idx    = pd.PeriodIndex(r.index, freq="M")
    s_core = r.reindex(CORE_PERIOD)
    n_miss = s_core.isna().sum()
    clean  = r.dropna()
    ok     = (str(idx.min()) <= "1999-01" and
              str(idx.max()) >= "2024-12" and n_miss == 0)
    status = "OK" if ok else "!!"
    if not ok:
        checks_pass = False
    print(f"  {status:<4} ret_{label:<5} "
          f"{str(idx.min())} -> {str(idx.max())} | "
          f"N={len(r):3d} | miss_core={n_miss:3d} | "
          f"mean={clean.mean():+.3f}  [{clean.min():.3f}, {clean.max():.3f}]")

print()
if checks_pass:
    print("  All checks passed.")
else:
    print("  Some checks failed -- review above.")


# ============================================================
# STEP 6 — FIGURE
# ============================================================

print(f"\n{'─'*65}")
print("STEP 6 -- Visual check figure")
print(f"{'─'*65}")

# ── Figure 14 — Corporate bond total return indices ──────────
colours = ["#1a6faf", "#d7191c", "#4dac26", "#762a83"]
labels_fig = {
    "cp1": "cp1 — LT01TRUU (1–3Y)",
    "cp2": "cp2 — LT03TRUU (3–5Y)",
    "cp3": "cp3 — LUACTRUU (Broad IG)",
    "cp4": "cp4 — LD07TRUU (Long >10Y)",
}

fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.32)

for (label, idx), col in zip(index_levels.items(), colours):
    s = idx.copy()
    s.index = s.index.to_timestamp()
    s_norm = s / s.dropna().iloc[0] * 100
    ax.plot(s_norm.index, s_norm.values, color=col, linewidth=1.6,
            label=labels_fig[label], alpha=0.9)

ax.axvspan(pd.Timestamp("2003-05-01"), pd.Timestamp("2023-06-30"),
           color="lightgrey", alpha=0.25, label="Analysis window")
ax.set_ylabel("Total Return Index (base = 100 at Jan 1999)", fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(labelsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0, 1.08, "F I G U R E   1 4", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
        "Corporate Bond Total Return Indices by Maturity (Bloomberg, rebased to 100 in Jan 1999)",
        transform=ax.transAxes, fontsize=13, color="black",
        va="bottom", style="italic", fontweight="bold")

ax.text(0, -0.22,
    r"$\it{Notes:}$  Monthly total return indices from Bloomberg Terminal"
    " (field: TOT_RETURN_INDEX_GROSS_DVDS), January 1999 – December 2024."
    " All series are rebased to 100 in January 1999 for comparability.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.31,
    "Longer-maturity bonds (cp4, >10Y) grew faster over this period but are also more volatile."
    " These four portfolios replace the agency bond portfolios (a1–a4) used in"
    " Fang, Liu and Roussanov (2022), which were not accessible at the Dauphine Bloomberg terminal.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.40,
    "Grey shading = core analysis window (May 2003 – Jun 2023).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")

plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Figure saved -> Fig14_corporate_bonds.png")


# ============================================================
# FINAL STATUS
# ============================================================

print(f"\n{'='*65}")
print("FINAL STATUS")
print(f"{'='*65}\n")

for fname, label in [
    ("cp1_corp_1_3Y.xlsx",    "cp1 raw Bloomberg xlsx"),
    ("cp2_corp_3_5Y.xlsx",    "cp2 raw Bloomberg xlsx"),
    ("cp3_corp_broad_IG.xlsx","cp3 raw Bloomberg xlsx"),
    ("cp4_corp_long.xlsx",    "cp4 raw Bloomberg xlsx"),
    ("corporate_total_return.csv", "Corporate monthly returns cp1-cp4"),
]:
    path = os.path.join(RAW_EX3, fname)
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        print(f"  OK  {fname:<40s}  {kb:6.1f} KB")
    else:
        print(f"  !!  {fname:<40s}  MISSING")

print(f"\n  Scripts done so far:")
print(f"  09_download_fred.py    -> macro + treasury yields  ✓")
print(f"  09a_shiller.py         -> Shiller P/D ratio        ✓")
print(f"  09b_ken_french.py      -> stocks s1-s5             (later)")
print(f"  09c_clean_bonds.py     -> corporate bonds cp1-cp4  ✓")
print(f"\n  Next: 09d — REITs + MSCI + currencies")
print(f"\n{'='*65}")
print("09c_clean_bonds.py -- DONE")
print(f"{'='*65}")