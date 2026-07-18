# ============================================================
# 09d_clean_currencies.py
# EXERCISE 3 — Currency Carry Portfolios: LRV xls -> Clean CSV
#
# SOURCE:
#   Lustig, H., Roussanov, N., and Verdelhan, A. (2011).
#   "Common Risk Factors in Currency Markets."
#   Review of Financial Studies, 24(11), 3731-3777.
#   Data file: CurrencyPortfolios-3.xls
#   Available at: https://web.mit.edu/adrienv/www/Research.html
#
# PORTFOLIOS (FLR 2022 notation):
#   fx1 = RX  — Dollar carry (equal-weighted average of all portfolios)
#   fx2 = P1  — Lowest interest rate currencies (short carry)
#   fx3 = P2
#   fx4 = P3
#   fx5 = P4
#   fx6 = P5
#   fx7 = P6  — Highest interest rate currencies (long carry)
#
# CONSTRUCTION (from LRV 2011):
#   Each month, all available currencies are sorted into 6 portfolios
#   based on their forward discount vs USD (= interest rate differential).
#   Portfolio 1 = lowest interest rate (appreciate vs USD on average).
#   Portfolio 6 = highest interest rate (depreciate vs USD on average).
#   Returns are excess returns in USD (= carry trade returns).
#   RX = equal-weighted average across all portfolios = dollar risk factor.
#
# NOTE ON SAMPLE:
#   The LRV data file covers January 1999 – May 2021 (269 months).
#   The last 43 months (June 2021 – December 2024) are not available
#   because the authors have not updated the file since 2021.
#   This is documented in the thesis. Beta estimation uses the
#   available 1999-2021 period for currency portfolios.
#
# INPUT  : Data/raw/Exercise_3/CurrencyPortfolios-3.xls
# OUTPUT : Data/raw/Exercise_3/currency_carry_returns.csv
#          output/figures/Fig_09e_currencies.png
#
# HOW TO RUN:
#   & C:/ProgramData/anaconda3/python.exe 09e_clean_currencies.py
# ============================================================

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── Auto-install xlrd if missing ──────────────────────────────
try:
    import xlrd
except ModuleNotFoundError:
    print("  xlrd not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "xlrd==1.2.0", "--quiet"])
    import xlrd

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3 = os.path.join(BASE, "Data", "raw", "Exercise_3")
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

INPUT_XLS  = os.path.join(RAW_EX3, "CurrencyPortfolios-3.xls")
OUTPUT_CSV = os.path.join(RAW_EX3, "currency_carry_returns.csv")
OUTPUT_FIG = os.path.join(FIG_OUT,  "Fig15_currencies.png")

# ── STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "xtick.color": "black",
    "ytick.color": "black", "axes.labelcolor": "black",
    "text.color": "black",
})

CORE_PERIOD = pd.period_range(start="2001-05", end="2021-05", freq="M")

print("=" * 65)
print("09e_clean_currencies.py")
print("EXERCISE 3 -- Currency Carry Portfolios (LRV 2011)")
print("=" * 65)

# ============================================================
# STEP 1 — READ RAW LRV FILE
# ============================================================
# We use the "All currencies" sheet which has 6 portfolios.
# FLR (2022) use this sheet (all available currencies, not just developed).
# The sheet structure:
#   Col 0  : Date (Excel serial number)
#   Col 1-6: Portfolio1 to Portfolio6 (carry-sorted, decimal returns)
#   Col 9  : RX = Mean (equal-weighted average = dollar carry factor)
#   Col 10 : HML = P6 - P1 (carry factor = long P6, short P1)
# Returns are in DECIMAL (e.g. 0.01 = 1%) -> we convert to %

print(f"\n{'─'*65}")
print("STEP 1 -- Reading CurrencyPortfolios-3.xls")
print(f"{'─'*65}")
print("  Sheet   : 'All currencies'")
print("  Source  : Lustig, Roussanov & Verdelhan (2011)")
print("  URL     : https://web.mit.edu/adrienv/www/Research.html\n")

if not os.path.exists(INPUT_XLS):
    raise FileNotFoundError(
        f"\n  File not found: {INPUT_XLS}\n"
        f"  Download from: https://web.mit.edu/adrienv/www/Research.html\n"
        f"  Save as CurrencyPortfolios-3.xls in Data/raw/Exercise_3/"
    )

wb = xlrd.open_workbook(INPUT_XLS)
ws = wb.sheet_by_name("All currencies")

print(f"  Shape   : {ws.nrows} rows x {ws.ncols} cols")
print(f"  Headers : {[ws.cell_value(0, j) for j in range(ws.ncols) if ws.cell_value(0,j) != '']}")

# ============================================================
# STEP 2 — PARSE DATA
# ============================================================
# Extract: Date, P1-P6 (carry portfolios), RX (dollar carry)
# Convert Excel serial dates to proper dates
# Convert decimal returns to percent

print(f"\n{'─'*65}")
print("STEP 2 -- Parsing data")
print(f"{'─'*65}")
print("  Converting decimal returns to % (multiply by 100)")
print("  Converting Excel serial dates to calendar dates\n")

rows = []
for i in range(1, ws.nrows):
    serial = ws.cell_value(i, 0)
    if not serial:
        continue
    # Convert Excel serial date -> Python datetime -> Period
    date = xlrd.xldate_as_datetime(serial, wb.datemode)
    period = pd.Period(date, freq="M")

    # Extract portfolio returns (decimal) + RX
    p1 = ws.cell_value(i, 1)   # lowest interest rate currencies
    p2 = ws.cell_value(i, 2)
    p3 = ws.cell_value(i, 3)
    p4 = ws.cell_value(i, 4)
    p5 = ws.cell_value(i, 5)
    p6 = ws.cell_value(i, 6)   # highest interest rate currencies
    rx = ws.cell_value(i, 9)   # equal-weighted average (dollar carry)

    rows.append([period, p1, p2, p3, p4, p5, p6, rx])

df = pd.DataFrame(rows, columns=[
    "date", "P1", "P2", "P3", "P4", "P5", "P6", "RX"
])
df = df.set_index("date")

# Convert decimal -> percent
df = df * 100

print(f"  Raw data: {df.index[0]} -> {df.index[-1]}")
print(f"  Rows    : {len(df)}")
print(f"\n  First 3 rows (% per month):")
print(df.head(3).round(4).to_string())
print(f"\n  Last 3 rows:")
print(df.tail(3).round(4).to_string())

# ============================================================
# STEP 3 — RENAME TO FLR NOTATION
# ============================================================
# FLR (2022) notation:
#   fx1 = RX  (dollar carry factor)
#   fx2 = P1  (lowest interest rate = short carry)
#   fx3 = P2
#   fx4 = P3
#   fx5 = P4
#   fx6 = P5
#   fx7 = P6  (highest interest rate = long carry)

print(f"\n{'─'*65}")
print("STEP 3 -- Renaming to FLR notation (fx1-fx7)")
print(f"{'─'*65}\n")

df_out = pd.DataFrame({
    "ret_fx1": df["RX"],   # dollar carry = equal-weighted mean
    "ret_fx2": df["P1"],   # lowest interest rate
    "ret_fx3": df["P2"],
    "ret_fx4": df["P3"],
    "ret_fx5": df["P4"],
    "ret_fx6": df["P5"],
    "ret_fx7": df["P6"],   # highest interest rate
})

for col in df_out.columns:
    s = df_out[col].dropna()
    print(f"  {col} | mean={s.mean():+.3f}%/m  std={s.std():.3f}%/m  "
          f"[{s.min():.3f}%, {s.max():.3f}%]")

# ============================================================
# STEP 4 — SAVE CSV
# ============================================================

print(f"\n{'─'*65}")
print("STEP 4 -- Saving currency_carry_returns.csv")
print(f"{'─'*65}")

df_out.index = df_out.index.astype(str)
df_out.index.name = "observation_date"
df_out.reset_index().to_csv(OUTPUT_CSV, index=False)

print(f"\n  File    : {OUTPUT_CSV}")
print(f"  Rows    : {len(df_out)}")
print(f"  Columns : {list(df_out.columns)}")
print(f"  Period  : 1999-01 -> 2021-05")
print(f"  Units   : % per month (excess returns vs USD)")
print(f"\n  NOTE: Data ends May 2021 (last update by Verdelhan).")
print(f"  The 43-month gap (Jun 2021 – Dec 2024) is documented")
print(f"  in the thesis. Beta estimation uses 1999-2021 for fx.")

# ============================================================
# STEP 5 — VERIFICATION
# ============================================================

print(f"\n{'─'*65}")
print("STEP 5 -- Verification checks")
print(f"{'─'*65}\n")

checks = [
    ("Rows = 269",       len(df_out) == 269,   f"{len(df_out)} rows"),
    ("Starts 1999-01",   df_out.index[0] == "1999-01",  df_out.index[0]),
    ("Ends 2021-05",     df_out.index[-1] == "2021-05", df_out.index[-1]),
    ("No NaN",           df_out.isna().sum().sum() == 0,
     f"{df_out.isna().sum().sum()} missing values"),
    ("Returns plausible",df_out.abs().max().max() < 30,
     f"max abs = {df_out.abs().max().max():.2f}%"),
    ("fx7 > fx2 on avg", df_out["ret_fx7"].mean() > df_out["ret_fx2"].mean(),
     f"fx7={df_out['ret_fx7'].mean():.3f}% > fx2={df_out['ret_fx2'].mean():.3f}%"),
]

all_ok = True
for label, passed, detail in checks:
    status = "OK  " if passed else "FAIL"
    if not passed:
        all_ok = False
    print(f"  {status}  {label:<30s}  {detail}")

print()
if all_ok:
    print("  All checks passed.")
    print("  NOTE: fx7 > fx2 confirms carry premium — high interest")
    print("  rate currencies earn more than low interest rate ones.")
else:
    print("  Some checks failed.")

# ============================================================
# STEP 6 — FIGURE
# ============================================================

print(f"\n{'─'*65}")
print("STEP 6 -- Visual check figure")
print(f"{'─'*65}")

# ── Figure 15 — Currency carry cumulative returns ─────────────
colours = ["#1a6faf", "#4393c3", "#74add1", "#fdae61", "#f46d43",
           "#d73027", "#a50026"]
labels  = ["fx1 — RX (dollar carry)", "fx2 — P1 (low interest rate)",
           "fx3 — P2", "fx4 — P3", "fx5 — P4",
           "fx6 — P5", "fx7 — P6 (high interest rate)"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.40)

for col, col_label, col_colour in zip(df_out.columns, labels, colours):
    s = df_out[col].copy()
    s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp()
    cum = (1 + s / 100).cumprod()
    ax.plot(cum.index, cum.values, color=col_colour,
            linewidth=1.4, label=col_label, alpha=0.9)

ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Cumulative gross return (base = 1 in Jan 1999)", fontsize=12)
ax.legend(fontsize=9, frameon=False, ncol=2)
ax.tick_params(labelsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0, 1.08, "F I G U R E   1 5", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
        "Currency Carry Portfolios — Cumulative Returns (Lustig, Roussanov & Verdelhan 2011)",
        transform=ax.transAxes, fontsize=13, color="black",
        va="bottom", style="italic", fontweight="bold")

ax.text(0, -0.22,
    r"$\it{Notes:}$  Monthly excess returns in USD from Lustig, Roussanov and Verdelhan (2011),"
    "\navailable at web.mit.edu/adrienv/www/Research.html."
    " Each month, currencies are sorted into 6 portfolios"
    "\nby forward discount (their interest rate differential vs the USD).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top", linespacing=1.7)
ax.text(0, -0.40,
    "fx2 holds low interest rate currencies (short carry); fx7 holds high interest rate currencies"
    "\n(long carry); fx1 (RX) is the equal-weighted average across all portfolios (the dollar risk factor).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top", linespacing=1.7)
ax.text(0, -0.52,
    "Data ends May 2021 — 43 months (Jun 2021 – Dec 2024) are missing."
    " Currency betas are estimated over the available 1999–2021 period.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")

plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Figure saved -> Fig15_currencies.png")

# ============================================================
# FINAL STATUS
# ============================================================

print(f"\n{'='*65}")
print("FINAL STATUS")
print(f"{'='*65}\n")

for fname, label in [
    ("CurrencyPortfolios-3.xls",     "LRV raw file"),
    ("currency_carry_returns.csv",   "Currency monthly returns fx1-fx7"),
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
print(f"  09e_clean_currencies.py-> currency carry fx1-fx7   ✓")
print(f"\n  Next: 09d — REITs + MSCI international")
print(f"\n{'='*65}")
print("09e_clean_currencies.py -- DONE")
print(f"{'='*65}")