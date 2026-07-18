# ============================================================
# 09b_download_fred.py
# EXERCISE 3 — Macro Data Download (FRED + Shiller)
#
# FRED series downloaded:
#   CPILFESL  — Core CPI (ex food & energy)     [VAR variable 1]
#   CPIFABSL  — Food CPI                         [VAR variable 2]
#   CPIENGSL  — Energy CPI                       [VAR variable 3]
#   DGS1      — 1-year Treasury yield            [t1]
#   DGS2      — 2-year Treasury yield            [t2]
#   DGS3      — 3-year Treasury yield            [t3]
#   DGS5      — 5-year Treasury yield            [t4]
#   DGS7      — 7-year Treasury yield            [t5]
#   DGS10     — 10-year Treasury yield           [t6]
#   DGS20     — 20-year Treasury yield           [t7]
#   DGS30     — 30-year Treasury yield           [t7]
#
# Derived series computed:
#   pi_core, pi_food, pi_energy  (monthly annualised inflation)
#   treasury_approx_returns.csv  (t1-t7 monthly return approximations)
#
# Shiller P/D ratio:
#   Loaded from shiller_pd_ratio.csv (produced by 09a_shiller.py)
#
# INPUT  : Data/raw/Exercise_3/shiller_pd_ratio.csv (already exists)
# OUTPUT : Data/raw/Exercise_3/*.csv
#          output/figures/Fig_09_data_check.png
#
# HOW TO RUN:
#   & C:/ProgramData/anaconda3/python.exe 09_download_fred.py
# ============================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── YOUR FRED API KEY ─────────────────────────────────────────
FRED_API_KEY = "d14920480d867bcd9f817e53ef541daa"   # <-- paste your key

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3 = os.path.join(BASE, "Data", "raw", "Exercise_3")
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(RAW_EX3, exist_ok=True)
os.makedirs(FIG_OUT,  exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "xtick.color": "black",
    "ytick.color": "black", "axes.labelcolor": "black",
    "text.color": "black",
})

# ── PARAMETERS ───────────────────────────────────────────────
START_DATE  = "1999-01-01"
END_DATE    = "2024-12-31"
CORE_PERIOD = pd.period_range(start="2001-05", end="2024-06", freq="M")

# Treasury durations for return approximation
# r_t = y_{t-1}/12 - duration x (y_t - y_{t-1})
# FLR portfolios: t1=1Y, t2=2Y, t3=3Y, t4=5Y, t5=7Y, t6=10Y, t7=20-30Y
TREASURY_MATURITIES = {
    "DGS1" : 1.0,   # t1
    "DGS2" : 2.0,   # t2
    "DGS3" : 3.0,   # t3
    "DGS5" : 5.0,   # t4
    "DGS7" : 7.0,   # t5
    "DGS10": 10.0,  # t6
    "DGS20": 20.0,  # t7
    "DGS30": 30.0,  # t7
}

print("=" * 65)
print("09_download_fred.py")
print("EXERCISE 3 -- Macro Data Download (FRED + Shiller)")
print("=" * 65)
print(f"\nDownload window : {START_DATE} -> {END_DATE}")
print(f"Core window     : 2001-05 -> 2024-06")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_fred(series_id, api_key, start, end,
                  max_retries=5, wait_seconds=3):
    """Downloads one FRED series as monthly pandas Series with retry."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        f"&file_type=json&observation_start={start}"
        f"&observation_end={end}&frequency=m&aggregation_method=avg"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                obs    = resp.json().get("observations", [])
                dates  = [o["date"] for o in obs]
                values = [float(o["value"]) if o["value"] != "."
                          else np.nan for o in obs]
                s = pd.Series(values,
                              index=pd.to_datetime(dates),
                              name=series_id)
                s.index = s.index.to_period("M")
                return s
            else:
                if attempt < max_retries:
                    print(f"    HTTP {resp.status_code} -- retrying ({attempt}/{max_retries})...")
                    time.sleep(wait_seconds)
                else:
                    raise ValueError(f"HTTP {resp.status_code} after {max_retries} attempts")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                raise e


def save_series(s, filename, colname):
    """Saves a Series with PeriodIndex to CSV."""
    df = s.reset_index()
    df.columns = ["observation_date", colname]
    df["observation_date"] = df["observation_date"].astype(str)
    df.to_csv(os.path.join(RAW_EX3, filename), index=False)


def to_dt(s):
    s = s.copy()
    if isinstance(s.index, pd.PeriodIndex):
        s.index = s.index.to_timestamp()
    return s


def fmt_ax(ax):
    ax.axvspan(pd.Timestamp("2001-05-01"), pd.Timestamp("2024-06-30"),
               color="lightgrey", alpha=0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.tick_params(labelsize=8)


def verify(name, s):
    if s is None or (hasattr(s, "__len__") and len(s) == 0):
        print(f"  EMPTY {name}")
        return
    idx    = s.index if isinstance(s.index, pd.PeriodIndex) \
             else pd.PeriodIndex(s.index, freq="M")
    s_core = s.reindex(CORE_PERIOD)
    n_miss = s_core.isna().sum()
    clean  = s.dropna()
    ok     = (str(idx.min()) <= "1999-01" and
              str(idx.max()) >= "2024-12" and n_miss == 0)
    status = "OK" if ok else "!!"
    print(f"  {status:<5} {name:<28} {str(idx.min()):>8} {str(idx.max()):>8} "
          f"N={len(s):3d}  miss={n_miss}  "
          f"mean={clean.mean():+8.3f}  [{clean.min():.3f}, {clean.max():.3f}]")


# ============================================================
# SECTION 1 -- DOWNLOAD ALL FRED SERIES
# ============================================================

FRED_SERIES = {
    "CPILFESL": "Core CPI (ex food & energy)  [VAR var 1]",
    "CPIFABSL": "Food CPI                      [VAR var 2]",
    "CPIENGSL": "Energy CPI                    [VAR var 3]",
    "DGS1"    : "1-year Treasury yield         [t1]",
    "DGS2"    : "2-year Treasury yield         [t2]",
    "DGS3"    : "3-year Treasury yield         [t3]",
    "DGS5"    : "5-year Treasury yield         [t4]",
    "DGS7"    : "7-year Treasury yield         [t5]",
    "DGS10"   : "10-year Treasury yield        [t6]",
    "DGS20"   : "20-year Treasury yield        [t7]",
    "DGS30"   : "30-year Treasury yield        [t7]",
}

print(f"\n{'─'*65}")
print("SECTION 1 -- Downloading FRED series")
print(f"{'─'*65}")

downloaded = {}
failed     = []

for ticker, description in FRED_SERIES.items():
    path = os.path.join(RAW_EX3, f"{ticker}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0)
        s.index = pd.PeriodIndex(s.index, freq="M")
        downloaded[ticker] = s.iloc[:, 0]
        print(f"  +  {ticker:10s} | already saved -- loaded from disk")
        continue
    time.sleep(1)
    try:
        s = download_fred(ticker, FRED_API_KEY, START_DATE, END_DATE)
        downloaded[ticker] = s
        save_series(s, f"{ticker}.csv", ticker)
        print(f"  OK {ticker:10s} | {description} | "
              f"{len(s)} obs | {s.isna().sum()} missing")
    except Exception as e:
        print(f"  FAIL {ticker:8s} | {e}")
        print(f"    -> https://fred.stlouisfed.org/series/{ticker}")
        failed.append(ticker)

if failed:
    print(f"\n  WARNING: {len(failed)} failed: {failed}")
else:
    print(f"\n  OK: All {len(FRED_SERIES)} series downloaded successfully")


# ============================================================
# SECTION 2 -- MONTHLY ANNUALISED INFLATION RATES
# ============================================================
# pi_t = 12 x [log(CPI_t) - log(CPI_{t-1})] x 100

print(f"\n{'─'*65}")
print("SECTION 2 -- Monthly annualised inflation rates")
print(f"{'─'*65}")
print("  pi_t = (CPI_t / CPI_{t-1} - 1) x 12 x 100  [exact MoM annualized]\n")

inflation = {}
for cpi_ticker, label in [("CPILFESL", "pi_core"),
                           ("CPIFABSL", "pi_food"),
                           ("CPIENGSL", "pi_energy")]:
    if cpi_ticker not in downloaded:
        print(f"  SKIP {label} -- {cpi_ticker} not available")
        continue
    # CHANGED: exact MoM annualized formula (consistent with VAR in script 11)
    # OLD: 12 * (log(CPI_t) - log(CPI_{t-1})) * 100  [log-change approximation]
    # NEW: (CPI_t / CPI_{t-1} - 1) * 12 * 100        [exact, same as script 11]
    pi = (downloaded[cpi_ticker] / downloaded[cpi_ticker].shift(1) - 1) * 12 * 100
    pi.name = label
    inflation[label] = pi
    save_series(pi, f"{label}.csv", label)
    s = pi.dropna()
    print(f"  {label:12s} | mean={s.mean():+.2f}%  std={s.std():.2f}%  "
          f"[{s.min():.2f}%, {s.max():.2f}%]")


# ============================================================
# SECTION 3 -- TREASURY RETURN APPROXIMATIONS (t1-t7)
# ============================================================
# r_t = y_{t-1}/12 - duration x (y_t - y_{t-1})
#
# FLR (2022) mapping:
#   t1 = DGS1  (1Y)
#   t2 = DGS2  (2Y)
#   t3 = DGS3  (3Y)   <- added vs previous version
#   t4 = DGS5  (5Y)
#   t5 = DGS7  (7Y)
#   t6 = DGS10 (10Y)  <- added vs previous version
#   t7 = average of DGS20 and DGS30

print(f"\n{'─'*65}")
print("SECTION 3 -- Treasury monthly return approximations (t1-t7)")
print(f"{'─'*65}")
print("  r_t = y_{t-1}/12 - duration x delta_y\n")

treasury_returns = {}
for ticker, dur in TREASURY_MATURITIES.items():
    if ticker not in downloaded:
        print(f"  SKIP {ticker} -- not available")
        continue
    y = downloaded[ticker]
    r = (y.shift(1) / 12) - dur * (y - y.shift(1))
    r.name = f"ret_{ticker.lower()}"
    treasury_returns[ticker] = r
    print(f"  {ticker:6s} (dur={dur:5.1f}y) | "
          f"mean={r.mean():+.3f}%/m  std={r.std():.3f}%/m")

# t7 = average of DGS20 and DGS30 (FLR use 20-30Y long bond)
if "DGS20" in treasury_returns and "DGS30" in treasury_returns:
    t7 = (treasury_returns["DGS20"] + treasury_returns["DGS30"]) / 2
    t7.name = "ret_t7_avg2030"
    treasury_returns["t7_avg"] = t7
    print(f"  t7     (avg 20Y+30Y) | "
          f"mean={t7.mean():+.3f}%/m  std={t7.std():.3f}%/m")

# Save all to one file (overwrites previous version with more columns)
tret_path = os.path.join(RAW_EX3, "treasury_approx_returns.csv")
df_tret   = pd.DataFrame({r.name: r for r in treasury_returns.values()})
df_tret.index = df_tret.index.astype(str)
df_tret.reset_index().rename(
    columns={"index": "observation_date"}
).to_csv(tret_path, index=False)
print(f"\n  OK: treasury_approx_returns.csv saved")
print(f"  {len(df_tret)} rows | {len(df_tret.columns)} columns")
print(f"  Columns: {list(df_tret.columns)}")


# ============================================================
# SECTION 4 -- LOAD SHILLER P/D (already produced by 09a_shiller.py)
# ============================================================

print(f"\n{'─'*65}")
print("SECTION 4 -- Shiller P/D ratio")
print(f"{'─'*65}")

shiller_path = os.path.join(RAW_EX3, "shiller_pd_ratio.csv")
pd_ratio     = None

if os.path.exists(shiller_path):
    df_sh = pd.read_csv(shiller_path)
    if df_sh["pd_ratio"].notna().sum() > 0:
        df_sh["observation_date"] = pd.PeriodIndex(
            df_sh["observation_date"], freq="M")
        pd_ratio = df_sh.set_index("observation_date")["pd_ratio"]
        print(f"  + shiller_pd_ratio.csv loaded ({len(pd_ratio)} rows)")
else:
    print(f"  !! shiller_pd_ratio.csv not found")
    print(f"     Run 09a_shiller.py first to produce it")


# ============================================================
# SECTION 5 -- VERIFICATION
# ============================================================

print(f"\n{'─'*65}")
print("SECTION 5 -- Verification checks")
print(f"{'─'*65}\n")

print("  CPI Levels:")
for t in ["CPILFESL", "CPIFABSL", "CPIENGSL"]:
    verify(t, downloaded.get(t))

print("\n  Monthly Inflation Rates (annualised %):")
for label, s in inflation.items():
    verify(label, s)

print("\n  Treasury Yields (%):")
for t in ["DGS1","DGS2","DGS3","DGS5","DGS7","DGS10","DGS20","DGS30"]:
    verify(t, downloaded.get(t))

print("\n  Treasury Monthly Returns (%/month):")
for t, r in treasury_returns.items():
    verify(f"ret_{t.lower()}", r)

print("\n  Shiller P/D Ratio:")
verify("pd_ratio", pd_ratio)


# ============================================================
# SECTION 6 -- FIGURE
# ============================================================

print(f"\n{'─'*65}")
print("SECTION 6 -- Visual check figure")
print(f"{'─'*65}")

# ── Figure 12 — Treasury yields by maturity ──────────────────
colours_t = ["#1a9641","#78c679","#addd8e","#fdae61",
             "#f46d43","#d73027","#a50026","#67001f"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.32)

for (ticker, _), col in zip(TREASURY_MATURITIES.items(), colours_t):
    if ticker in downloaded:
        s = to_dt(downloaded[ticker].dropna())
        ax.plot(s.index, s.values, color=col, linewidth=1.6,
                label=ticker.replace("DGS","") + "Y", alpha=0.9)

ax.axvspan(pd.Timestamp("2003-05-01"), pd.Timestamp("2023-06-30"),
           color="lightgrey", alpha=0.25, label="Analysis window")
ax.set_ylabel("Yield (%)", fontsize=12)
ax.legend(fontsize=10, frameon=False, ncol=4, loc="upper right")
ax.tick_params(labelsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title (above axes)
ax.text(0, 1.08, "F I G U R E   1 2", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02, "U.S. Treasury Yields by Maturity (% per year)",
        transform=ax.transAxes, fontsize=13, color="black",
        va="bottom", style="italic", fontweight="bold")

# Note — three separate lines, each capped at axes width
ax.text(0, -0.22,
    r"$\it{Notes:}$  U.S. Treasury constant-maturity yields from FRED (series DGS1 through DGS30),"
    " monthly averages, January 1999 – December 2024.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.31,
    r"Used to construct approximate monthly Treasury total returns via the duration approximation"
    r"  $r_t = y_{t-1}/12 \;-\; D \cdot \Delta y_t$,"
    r"  where $D \approx$ maturity in years and $\Delta y_t = y_t - y_{t-1}$.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.40,
    "Grey shading = core analysis window (May 2003 – Jun 2023).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")

fig12_path = os.path.join(FIG_OUT, "Fig12_treasury_yields.png")
plt.savefig(fig12_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Figure saved -> Fig12_treasury_yields.png")


# ── Figure 13 — Treasury monthly returns (long-end) ──────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.32)

for ticker, col, lab in [("DGS10","#4393c3","10Y"),
                          ("DGS20","#d73027","20Y"),
                          ("DGS30","#67001f","30Y")]:
    if ticker in treasury_returns:
        s = to_dt(treasury_returns[ticker].dropna())
        ax.plot(s.index, s.values, color=col, linewidth=1.4,
                label=f"{lab} return", alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvspan(pd.Timestamp("2003-05-01"), pd.Timestamp("2023-06-30"),
           color="lightgrey", alpha=0.25, label="Analysis window")
ax.set_ylabel("% per month", fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(labelsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0, 1.08, "F I G U R E   1 3", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02, "Treasury Monthly Total Returns — Long Maturities (10Y, 20Y, 30Y)",
        transform=ax.transAxes, fontsize=13, color="black",
        va="bottom", style="italic", fontweight="bold")

ax.text(0, -0.22,
    r"$\it{Notes:}$  Monthly total returns approximated as"
    r"  $r_t = y_{t-1}/12 \;-\; D \cdot \Delta y_t$,"
    r"  where $y_t$ = monthly average yield (FRED DGS series),"
    r"  $D \approx$ maturity in years,  $\Delta y_t = y_t - y_{t-1}$.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.31,
    "The first term captures coupon income; the second captures the price gain or loss from"
    " yield movements. Longer-maturity bonds are more volatile because a given yield change"
    " moves their price by a larger amount.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")
ax.text(0, -0.40,
    "Grey shading = core analysis window (May 2003 – Jun 2023).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top")

fig13_path = os.path.join(FIG_OUT, "Fig13_treasury_returns.png")
plt.savefig(fig13_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Figure saved -> Fig13_treasury_returns.png")


# ============================================================
# SECTION 7 -- FINAL STATUS
# ============================================================

print(f"\n{'='*65}")
print("FINAL STATUS -- Files in Data/raw/Exercise_3/")
print(f"{'='*65}\n")

required = [
    "CPILFESL.csv", "CPIFABSL.csv", "CPIENGSL.csv",
    "pi_core.csv",  "pi_food.csv",  "pi_energy.csv",
    "DGS1.csv",  "DGS2.csv",  "DGS3.csv",
    "DGS5.csv",  "DGS7.csv",  "DGS10.csv",
    "DGS20.csv", "DGS30.csv",
    "treasury_approx_returns.csv",
    "shiller_pd_ratio.csv",
]

all_ok = True
for fname in required:
    path = os.path.join(RAW_EX3, fname)
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        print(f"  OK  {fname:<45s}  {kb:6.1f} KB")
    else:
        print(f"  !!  {fname:<45s}  MISSING")
        all_ok = False

print()
if all_ok:
    print("  All files present. Ready for next step.")
else:
    print("  Some files missing -- see above.")

print(f"\n  Next scripts:")
print(f"  -> 09a_shiller.py   (already done)")
print(f"  -> 09b_ken_french.py  (stocks s1-s5)")
print(f"  -> 09c_download_bonds.py  (agency + corporate)")
print(f"\n{'='*65}")
print("09_download_fred.py -- DONE")
print(f"{'='*65}")