# ============================================================
# 09a_shiller_xlsx.py
# Reads ie_data.xlsx (Shiller saved as xlsx from Excel)
# Computes log(P/D) and saves shiller_pd_ratio.csv
#
# Run AFTER saving ie_data.xls as ie_data.xlsx in Excel.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE        = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3     = os.path.join(BASE, "Data", "raw", "Exercise_3")
FIG_OUT     = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

XLSX_FILE   = os.path.join(RAW_EX3, "ie_data.xlsx")
OUTPUT_CSV  = os.path.join(RAW_EX3, "shiller_pd_ratio.csv")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

print("=" * 60)
print("09a_shiller_xlsx.py")
print("Shiller P/D ratio from ie_data.xlsx")
print("=" * 60)

# ── Check file exists ─────────────────────────────────────────
if not os.path.exists(XLSX_FILE):
    print(f"\n  FILE NOT FOUND: {XLSX_FILE}")
    print("\n  Steps to create it:")
    print("  1. Open ie_data.xls in Excel")
    print("  2. File -> Save As")
    print("  3. Type: Classeur Excel (.xlsx)")
    print("  4. Save as ie_data.xlsx in the same folder")
    raise SystemExit(1)

print(f"\n  Reading: ie_data.xlsx")
print(f"  Engine : openpyxl (standard for .xlsx)")

# ── Read the file ─────────────────────────────────────────────
# Shiller's sheet is named "Data"
# First 7 rows are title text, row 8 is the column header
raw = pd.read_excel(
    XLSX_FILE,
    sheet_name="Data",
    header=7,           # row 8 (0-indexed: 7) = column names
    engine="openpyxl",  # no xlrd needed
)

print(f"  Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
print(f"  Columns  : {raw.columns.tolist()[:8]}")

# ── Extract Date, P (price), D (monthly dividend) ────────────
df = raw[["Date", "P", "D"]].copy()
df.columns = ["date_raw", "price", "dividend_monthly"]

# Drop non-numeric rows (text notes at bottom of Shiller file)
df = df[pd.to_numeric(df["date_raw"], errors="coerce").notna()].copy()
df["date_raw"]         = pd.to_numeric(df["date_raw"])
df["price"]            = pd.to_numeric(df["price"],            errors="coerce")
df["dividend_monthly"] = pd.to_numeric(df["dividend_monthly"], errors="coerce")
df = df.dropna(subset=["date_raw", "price", "dividend_monthly"])

# ── Parse Shiller date format: YYYY.MM ───────────────────────
# e.g. 1999.01 = January 1999, 1999.10 = October 1999
df["year"]  = df["date_raw"].astype(int)
df["month"] = ((df["date_raw"] % 1) * 100).round().astype(int)
df["month"] = df["month"].replace(0, 1)  # edge case

df["observation_date"] = pd.to_datetime(
    {"year": df["year"], "month": df["month"], "day": 1}
).dt.to_period("M")

df = df.set_index("observation_date")[["price", "dividend_monthly"]].sort_index()

print(f"\n  Shiller raw data: {df.index[0]} -> {df.index[-1]}")
print(f"  Total rows      : {len(df)}")

# ── Compute log(P/D) ─────────────────────────────────────────
# Annual dividend = monthly dividend x 12
# (Shiller's D is monthly, not annual)
# log(P/D) = ln(Price / Annual_Dividend)
df["dividend_annual"] = df["dividend_monthly"] * 12
df["pd_ratio"]        = np.log(df["price"] / df["dividend_annual"])

# ── Filter to 1999-01 -> 2024-12 ─────────────────────────────
df         = df.loc["1999-01":"2024-12"]
pd_ratio   = df["pd_ratio"].dropna()
last_real  = pd_ratio.index[-1]

# Print worked example for transparency
ex = df.loc["1999-01"]
print(f"\n  Example (1999-01):")
print(f"    Price           = {ex['price']:.4f}")
print(f"    Monthly div     = {ex['dividend_monthly']:.4f}")
print(f"    Annual div      = {ex['dividend_annual']:.4f}  (x 12)")
print(f"    log(P/D)        = ln({ex['price']:.4f} / {ex['dividend_annual']:.4f})")
print(f"                    = {ex['pd_ratio']:.6f}")

# ── Extend to 2024-12 if data ends earlier ───────────────────
full_idx = pd.period_range(start="1999-01", end="2024-12", freq="M")
missing  = full_idx[~full_idx.isin(pd_ratio.index)]

if len(missing) > 0:
    fill_val  = pd_ratio.iloc[-6:].mean()
    extension = pd.Series(fill_val, index=missing, name="pd_ratio")
    pd_ratio  = pd.concat([pd_ratio, extension]).sort_index()
    print(f"\n  Shiller ends at {last_real}.")
    print(f"  Extending {len(missing)} months with flat value = {fill_val:.4f}")
    print(f"  (mean of last 6 available months — documented in thesis)")
else:
    print(f"\n  Data covers full window — no extension needed")

# ── Summary statistics ────────────────────────────────────────
s = pd_ratio.dropna()
print(f"\n  Summary statistics (full series):")
print(f"    N     = {len(s)}")
print(f"    Mean  = {s.mean():.4f}")
print(f"    Std   = {s.std():.4f}")
print(f"    Min   = {s.min():.4f}  at {s.idxmin()}")
print(f"    Max   = {s.max():.4f}  at {s.idxmax()}")

# ── Save CSV ──────────────────────────────────────────────────
out = pd_ratio.reset_index()
out.columns = ["observation_date", "pd_ratio"]
out["observation_date"] = out["observation_date"].astype(str)
out["extrapolated"]     = out["observation_date"] > str(last_real)
out.to_csv(OUTPUT_CSV, index=False)

print(f"\n  SAVED: shiller_pd_ratio.csv")
print(f"  Rows  : {len(out)}")
print(f"  Cols  : observation_date | pd_ratio | extrapolated")
print(f"\n  First 3 rows:")
print(out.head(3).to_string(index=False))
print(f"\n  Last 3 rows:")
print(out.tail(3).to_string(index=False))

# ── Figure 11 — Shiller log(P/D) ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

s_plot = pd_ratio.copy()
s_plot.index = s_plot.index.to_timestamp()

real_s = s_plot[s_plot.index < pd.Timestamp("2023-07-01")]
ext_s  = s_plot[s_plot.index >= pd.Timestamp("2023-07-01")]

ax.plot(real_s.index, real_s.values,
        color="#762a83", linewidth=1.8, label="Shiller data")
if len(ext_s) > 0:
    ax.plot(ext_s.index, ext_s.values,
            color="#762a83", linewidth=1.5, linestyle=":",
            label=f"Flat extrapolation ({len(ext_s)} months)")
    ax.axvspan(ext_s.index[0], ext_s.index[-1],
               color="#fdae61", alpha=0.25, label="Extrapolated")

ax.axvspan(pd.Timestamp("2003-05-01"), pd.Timestamp("2023-06-30"),
           color="lightgrey", alpha=0.25, label="Analysis window (May 2003 – Jun 2023)")
ax.axhline(pd_ratio.mean(), color="gray", linewidth=0.9,
           linestyle="--", label=f"Sample mean = {pd_ratio.mean():.2f}")

ax.set_ylabel("log(P/D)", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.legend(fontsize=10, frameon=False, loc="lower right")
ax.tick_params(labelsize=10)

# Title
ax.text(0, 1.04, "F I G U R E   1 1", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.00, "Shiller Log Price-Dividend Ratio  —  log(P/D) = ln(Price / Annual Dividend)",
        transform=ax.transAxes, fontsize=12, color="black",
        va="bottom", style="italic", fontweight="bold")

# Note
fig.text(0.01, -0.06,
    "Notes: The log price-dividend ratio is computed as ln(P / D×12), where P is the S&P 500 monthly price "
    "and D is the monthly dividend from Shiller (2024), Yale University (ie_data.xlsx). "
    "The ratio is used as a valuation control variable in the VAR from which inflation shocks are extracted. "
    "The orange dotted region (Jul 2023 – Dec 2024) is extrapolated using the mean of the last 6 available months "
    "— this period falls outside the core analysis window and has no effect on the main results.",
    fontsize=9, color="#555555", style="italic", va="top", wrap=True)

plt.tight_layout()
fig.subplots_adjust(bottom=0.14)
fig_path = os.path.join(FIG_OUT, "Fig11_shiller_pd.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  SAVED: Fig11_shiller_pd.png")

# ── Verification checks ───────────────────────────────────────
print(f"\n{'='*60}")
print("VERIFICATION")
print(f"{'='*60}")

checks = [
    ("Total rows = 312",     len(out) == 312,
     f"{len(out)} rows"),
    ("Starts 1999-01",       out.iloc[0].observation_date == "1999-01",
     out.iloc[0].observation_date),
    ("Ends 2024-12",         out.iloc[-1].observation_date == "2024-12",
     out.iloc[-1].observation_date),
    ("No NaN",               out.pd_ratio.isna().sum() == 0,
     f"{out.pd_ratio.isna().sum()} missing"),
    ("Range plausible",      out.pd_ratio.between(0.5, 2.5).all(),
     f"[{out.pd_ratio.min():.3f}, {out.pd_ratio.max():.3f}]"),
]

all_ok = True
for label, passed, detail in checks:
    status = "OK" if passed else "FAIL"
    if not passed:
        all_ok = False
    print(f"  {status:<5} {label:<25} {detail}")

print()
if all_ok:
    print("  All checks passed.")
    print("  shiller_pd_ratio.csv is ready.")
    print("\n  Now re-run 09a_download_fred_inflation.py")
    print("  -> It will detect shiller_pd_ratio.csv and show 14/14 OK")
else:
    print("  Some checks failed.")

print(f"\n{'='*60}")
print("09a_shiller_xlsx.py -- DONE")
print(f"{'='*60}")