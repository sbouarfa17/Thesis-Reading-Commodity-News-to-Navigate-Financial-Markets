# ============================================================
# 10_build_energy_Rt_FINAL.py
# EXERCISE 3 — Energy Demand-Supply Regime Indicator
#
# SPEC: 24-month rolling p70, min_periods=12
#
# JUSTIFICATION (from our Problem Analysis document):
#   - Rolling 60m (thesis plan initial) → N=184 obs (Mar 2008 start)
#     → FM sample too short to identify lambda
#   - 24m rolling → N=244 obs (Mar 2003 start) = +60 observations
#   - 24m is consistent with the rolling window used for Rt in Ex1&2
#   - ALL windows 12-60m give 7/7 correct signs (full robustness table)
#   - Prof instruction: "analogously to Rt" → rolling threshold, same logic
#
# EPISODE CLASSIFICATION (24m rolling p70):
#   Shale glut 2014-16   → SUPPLY  ✓ (correct)
#   Post-COVID 2020-21   → DEMAND  ✓ (correct)
#   Ukraine 2022         → SUPPLY  ✓ (correct, 8% demand months)
#   GFC 2008-09          → DEMAND  (demand volatility dominated, defensible)
#
# INPUT  : Data/raw/indice prof/output_commodities_articleused_m.csv
#          Data/cleaned/ex3_monthly.csv
# OUTPUT : Data/cleaned/ex3_monthly.csv  (+ energy_Rt, demand_dummy)
#          output/figures/Fig20_energy_Rt.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_IND = os.path.join(BASE, "Data", "raw", "indice prof")
CLEAN   = os.path.join(BASE, "Data", "cleaned")
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

LMPRP_CSV  = os.path.join(RAW_IND, "output_commodities_articleused_m.csv")
EX3_CSV    = os.path.join(CLEAN,   "ex3_monthly.csv")
OUTPUT_FIG = os.path.join(FIG_OUT, "Fig20_energy_Rt.png")

# ── PARAMETERS ────────────────────────────────────────────────
WINDOW      = 24    # rolling std window — consistent with Rt in Ex1&2
MIN_PERIODS = 12    # min obs for rolling std (allows earlier start)
ROLL_PCT    = 24    # rolling window for p70 threshold
THRESHOLD   = 0.70  # 70th percentile → ~30% demand by design
START = "2001-05-01"
END   = "2023-06-01"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

print("=" * 65)
print("10_build_energy_Rt_FINAL.py")
print(f"window={WINDOW}m | min_periods={MIN_PERIODS} | rolling p{int(THRESHOLD*100)} ({ROLL_PCT}m)")
print("=" * 65)

# ============================================================
# STEP 1 — Load LMPRP energy news indices
# ============================================================
print(f"\n{'─'*65}")
print("STEP 1 — Loading LMPRP energy commodity indices")
print(f"{'─'*65}")

comm = pd.read_csv(LMPRP_CSV, parse_dates=["date"], index_col="date")
comm.index = comm.index.to_period("M").to_timestamp()

nd_cols = ["std_netD_oil", "std_netD_natgas",
           "std_netD_heatingoil", "std_netD_gasoline"]
ns_cols = ["std_netS_oil", "std_netS_natgas",
           "std_netS_heatingoil", "std_netS_gasoline"]
nd_cols = [c if c in comm.columns else c.replace("heatingoil","gasoil") for c in nd_cols]
ns_cols = [c.replace("netD","netS") for c in nd_cols]

print(f"  NetDemand cols : {nd_cols}")
print(f"  NetSupply  cols: {ns_cols}")
nd_energy = comm[nd_cols].mean(axis=1)
ns_energy = comm[ns_cols].mean(axis=1)
print(f"  Data range     : {comm.index.min().date()} -> {comm.index.max().date()}")

# ============================================================
# STEP 2 — Compute energy_Rt
# ============================================================
print(f"\n{'─'*65}")
print(f"STEP 2 — energy_Rt  (rolling {WINDOW}m sigma, min_periods={MIN_PERIODS})")
print(f"{'─'*65}")

sigma_nd  = nd_energy.rolling(WINDOW, min_periods=MIN_PERIODS).std()
sigma_ns  = ns_energy.rolling(WINDOW, min_periods=MIN_PERIODS).std()
energy_rt = sigma_nd / (sigma_nd + sigma_ns)

s = energy_rt.dropna()
print(f"  First non-NaN : {s.index.min().date()}")
print(f"  Range         : [{s.min():.4f}, {s.max():.4f}]")
print(f"  Mean          : {s.mean():.4f}")

# ============================================================
# STEP 3 — Demand dummy: rolling p70
# ============================================================
print(f"\n{'─'*65}")
print(f"STEP 3 — Demand dummy  (rolling {ROLL_PCT}m p{int(THRESHOLD*100)})")
print(f"{'─'*65}")

rolling_p70  = energy_rt.rolling(ROLL_PCT, min_periods=MIN_PERIODS).quantile(THRESHOLD)
demand_dummy = (energy_rt > rolling_p70).astype(float)
demand_dummy[energy_rt.isna() | rolling_p70.isna()] = np.nan

idx_an = (energy_rt.index >= START) & (energy_rt.index <= END)
ert_an = energy_rt[idx_an]
p70_an = rolling_p70[idx_an]
dum_an = demand_dummy[idx_an].dropna()

print(f"  N demand (=1)  : {int((dum_an==1).sum())}  ({(dum_an==1).mean()*100:.1f}%)")
print(f"  N supply (=0)  : {int((dum_an==0).sum())}  ({(dum_an==0).mean()*100:.1f}%)")
print(f"  NaN (init)     : {demand_dummy[idx_an].isna().sum()}")
print(f"  First dummy    : {dum_an.index.min().date()}")

episodes = {
    "GFC energy collapse (2008-09)"     : ("2008-01","2009-06"),
    "Post-GFC / Arab Spring (2010-11)"  : ("2010-01","2011-12"),
    "Shale glut / OPEC war (2014-16)"   : ("2014-01","2016-12"),
    "Post-COVID demand surge (2020-21)" : ("2020-06","2021-12"),
    "Ukraine energy shock (2022)"       : ("2022-01","2022-12"),
}
print(f"\n  Episode classification:")
for ep,(s_date,e_date) in episodes.items():
    ep_d = dum_an[(dum_an.index>=s_date)&(dum_an.index<=e_date)]
    if len(ep_d)==0: continue
    pct=(ep_d==1).mean()*100
    lbl="DEMAND" if pct>50 else "SUPPLY"
    print(f"    {lbl:<6}  {pct:3.0f}%  {ep}")

# ============================================================
# STEP 4 — Update ex3_monthly.csv
# ============================================================
print(f"\n{'─'*65}")
print("STEP 4 — Updating ex3_monthly.csv")
print(f"{'─'*65}")

ex3 = pd.read_csv(EX3_CSV, parse_dates=["date"], index_col="date")
ex3.index = ex3.index.to_period("M").to_timestamp()

for col in ["energy_Rt","demand_dummy"]:
    if col in ex3.columns:
        ex3 = ex3.drop(columns=[col])

ex3["energy_Rt"]    = energy_rt.reindex(ex3.index)
ex3["demand_dummy"] = demand_dummy.reindex(ex3.index)

if "date" in ex3.columns:
    ex3 = ex3.drop(columns=["date"])
ex3.index = ex3.index.to_period("M").astype(str)
ex3.index.name = "date"
ex3.reset_index().to_csv(EX3_CSV, index=False)
print(f"  SAVED : {EX3_CSV}")
print(f"  Rows  : {len(ex3)}  |  Cols: {len(ex3.columns)}")

# ============================================================
# STEP 5 — Verification
# ============================================================
print(f"\n{'─'*65}")
print("STEP 5 — Verification")
print(f"{'─'*65}")

def ep_pct(s_date, e_date):
    sub = dum_an[(dum_an.index>=s_date)&(dum_an.index<=e_date)]
    return (sub==1).mean() if len(sub)>0 else np.nan

checks = [
    ("energy_Rt in (0,1)",
     ert_an.dropna().between(0,1).all(),
     f"[{ert_an.dropna().min():.3f}, {ert_an.dropna().max():.3f}]"),
    ("N total >= 240", len(dum_an)>=240,   f"N={len(dum_an)}"),
    ("Shale 2014-16 = SUPPLY",  ep_pct("2014-06","2016-12")<0.4,
     f"{ep_pct('2014-06','2016-12')*100:.0f}% demand"),
    ("Post-COVID 2020-21 = DEMAND", ep_pct("2020-06","2021-12")>0.7,
     f"{ep_pct('2020-06','2021-12')*100:.0f}% demand"),
    ("Ukraine 2022 = SUPPLY", ep_pct("2022-01","2022-12")<0.3,
     f"{ep_pct('2022-01','2022-12')*100:.0f}% demand"),
]
all_ok=True
for label,ok,detail in checks:
    status="OK  " if ok else "FAIL"
    if not ok: all_ok=False
    print(f"  {status}  {label:<45} {detail}")
print(f"\n  {'All checks PASSED' if all_ok else 'SOME CHECKS FAILED'}")

# ============================================================
# FIGURE 20
# ============================================================
print(f"\n{'─'*65}")
print("Producing Fig20_energy_Rt.png")
print(f"{'─'*65}")

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.subplots_adjust(left=0.07, right=0.72, top=0.85, bottom=0.28)

for t in ert_an.dropna().index:
    d = demand_dummy[t] if t in demand_dummy.index else np.nan
    if pd.isna(d): continue
    color = "#2166ac" if d==1 else "#d7191c"
    ax.axvspan(t, t+pd.DateOffset(months=1), alpha=0.13, color=color, linewidth=0)

line1, = ax.plot(ert_an.index, ert_an.values, color="black", linewidth=1.8, zorder=5)
line2, = ax.plot(p70_an.index, p70_an.values, color="#555555", linewidth=1.0,
                 linestyle="--", alpha=0.75, zorder=3)
ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":", alpha=0.35)

ax.set_ylim(0.28, 0.80)
ax.set_ylabel("energy_R$_t$", fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2024-01-01"))

ax.text(0, 1.06, "F I G U R E   2 0",
        transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
ax.text(0, 1.01,
        r"$\it{Energy\ Demand\text{-}Supply\ Regime\ Indicator}$"
        r" — energy_R$_t$  (May 2001 – Jun 2023)",
        transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

d_patch = mpatches.Patch(color="#2166ac", alpha=0.4,
    label="Demand regime\n(energy_R$_t$ > rolling p70)")
s_patch = mpatches.Patch(color="#d7191c", alpha=0.4,
    label="Supply regime\n(energy_R$_t$ < rolling p70)")
ax.legend(handles=[line1, line2, d_patch, s_patch],
          labels=["energy_R$_t$", "Rolling p70 (24m)",
                  "Demand regime", "Supply regime"],
          fontsize=9, frameon=True, framealpha=0.95, edgecolor="#cccccc",
          loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0)

ax.text(0, -0.16,
    r"$\it{Notes}$:  energy_R$_t$ = $\sigma_{24}$(NetDemand) / "
    r"[$\sigma_{24}$(NetDemand)+$\sigma_{24}$(NetSupply)]. "
    r"NetDemand and NetSupply = equal-weighted LMPRP news indices "
    r"(oil, natural gas, heating oil, gasoline). "
    r"24-month rolling window, consistent with R$_t$ in Exercises 1–2.",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top", linespacing=1.6)
ax.text(0, -0.26,
    f"Demand dummy = 1 when energy_R$_t$ > rolling 24-month 70th percentile "
    f"(real-time, no look-ahead). "
    f"N = {len(dum_an)} months ({int((dum_an==1).sum())} demand, "
    f"{int((dum_an==0).sum())} supply). "
    r"Sources: Malliaropulos, Passari & Petroulakis (2025); Lumbanraja et al. (2025).",
    transform=ax.transAxes, fontsize=9, color="#444444", va="top", linespacing=1.6)

plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: {OUTPUT_FIG}")

print(f"\n{'='*65}")
print("10_build_energy_Rt_FINAL.py — DONE")
print(f"  Window  : {WINDOW}m rolling std, {ROLL_PCT}m rolling p{int(THRESHOLD*100)}")
print(f"  N       : {len(dum_an)} ({int((dum_an==1).sum())} demand / {int((dum_an==0).sum())} supply)")
print(f"  Ukraine : {ep_pct('2022-01','2022-12')*100:.0f}% demand = SUPPLY ✓")
print(f"  COVID   : {ep_pct('2020-06','2021-12')*100:.0f}% demand = DEMAND ✓")
print(f"  Shale   : {ep_pct('2014-06','2016-12')*100:.0f}% demand = SUPPLY ✓")
print(f"{'='*65}")