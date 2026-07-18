# ============================================================
# 11_var_inflation_shocks_FINAL.py
# EXERCISE 3 — VAR(4) Inflation Shocks
#
# CHANGES VS OLD VERSION:
#   OLD: YoY inflation → std(eps_core)=0.14, std(eps_energy)=4.1
#        Beta signs wrong for our 2001-2023 sample
#   NEW: MoM annualized → std(eps_core)=1.13, std(eps_energy)=32.23
#        Energy beta signs match FLR Table 2 for all asset classes
#
# FORMULA: π_t = (P_t/P_{t-1} − 1) × 12 × 100
#
# VAR SPECIFICATION (FLR 2022, 6 variables, 4 lags):
#   [π_core, π_food, π_energy, r_f, log(P/D), ugap]
#   No Cholesky identification — raw reduced-form residuals
#
# FIX vs student version:
#   - Partition check now NaN-safe (checks non-NaN rows only)
#   - eps_energy_D/S use .where() not boolean assignment
#   - FEDFUNDS splice covers post-2023-06 (shadow rate ends there)
#
# PATHS (adapt to your machine):
#   BASE = your thesis folder
#   All raw CSVs: Data/raw/Exercise_3/ or Data/raw/
#
# OUTPUTS:
#   Data/cleaned/ex3_monthly.csv  (+ eps_core, eps_energy, eps_energy_D/S)
#   output/figures/Fig21_inflation_shocks.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

# ── PATHS — adapt these to your machine ──────────────────────
BASE     = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3  = os.path.join(BASE, "Data", "raw", "Exercise_3")
RAW_MP   = os.path.join(BASE, "Data", "raw", "Mp control variable")
CLEAN    = os.path.join(BASE, "Data", "cleaned")
FIG_OUT  = os.path.join(BASE, "output", "figures")
os.makedirs(FIG_OUT, exist_ok=True)

EX3_CSV    = os.path.join(CLEAN,   "ex3_monthly.csv")
OUTPUT_FIG = os.path.join(FIG_OUT, "Fig21_inflation_shocks.png")

# ── PARAMETERS ────────────────────────────────────────────────
FULL_PERIOD  = pd.period_range("1999-01", "2024-12", freq="M")
SAMPLE_START = "2001-05"
SAMPLE_END   = "2023-06"
VAR_LAGS     = 4

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

print("=" * 65)
print("11_var_inflation_shocks_FINAL.py")
print(f"VAR({VAR_LAGS}) | MoM annualized | sample {SAMPLE_START}–{SAMPLE_END}")
print("=" * 65)


# ============================================================
# STEP 1 — MoM annualized inflation from raw CPI index levels
# π_t = (CPI_t / CPI_{t-1} − 1) × 12 × 100
# ============================================================
print(f"\n{'─'*65}")
print("STEP 1 — MoM annualized inflation")
print(f"{'─'*65}")
print("  Formula: π_t = (CPI_t / CPI_{{t-1}} − 1) × 12 × 100\n")

def mom_ann(fpath, col):
    """Read a FRED CPI CSV and compute MoM annualized % change."""
    df = pd.read_csv(fpath, parse_dates=["observation_date"])
    df["period"] = df["observation_date"].dt.to_period("M")
    df = df.set_index("period")[[col]]
    return ((df[col] / df[col].shift(1)) - 1) * 12 * 100

series = {}

for name, fname, col in [
    ("pi_core",   "CPILFESL.csv", "CPILFESL"),
    ("pi_food",   "CPIFABSL.csv", "CPIFABSL"),
    ("pi_energy", "CPIENGSL.csv", "CPIENGSL"),
]:
    # Try Exercise_3 folder first, then fall back to raw root
    candidates = [
        os.path.join(RAW_EX3, fname),
        os.path.join(BASE, "Data", "raw", fname),
    ]
    fpath = next((p for p in candidates if os.path.exists(p)), None)
    if fpath is None:
        raise FileNotFoundError(f"Cannot find {fname} — check RAW_EX3 or raw root")
    s = mom_ann(fpath, col).reindex(FULL_PERIOD)
    series[name] = s
    ss = s.dropna()
    print(f"  {name:<12}  mean={ss.mean():+.2f}%  std={ss.std():.2f}%  "
          f"[{ss.min():.1f}, {ss.max():.1f}]")


# ============================================================
# STEP 2 — Policy rate: Wu-Xia shadow rate, spliced with FEDFUNDS
# Shadow rate ends 2023-06; FEDFUNDS fills the gap after that.
# ============================================================
print(f"\n{'─'*65}")
print("STEP 2 — Policy rate (Wu-Xia shadow + FEDFUNDS splice)")
print(f"{'─'*65}")

shadow_candidates = [
    os.path.join(RAW_MP,  "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(RAW_EX3, "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(BASE, "Data", "raw", "shadowrate_US.xls - Sheet1.csv"),
]
ff_candidates = [
    os.path.join(RAW_MP,  "FEDFUNDS.csv"),
    os.path.join(RAW_EX3, "FEDFUNDS.csv"),
    os.path.join(BASE, "Data", "raw", "FEDFUNDS.csv"),
]

shadow_path = next((p for p in shadow_candidates if os.path.exists(p)), None)
ff_path     = next((p for p in ff_candidates     if os.path.exists(p)), None)

if shadow_path is None or ff_path is None:
    raise FileNotFoundError("Cannot find shadowrate or FEDFUNDS — check paths above")

sr = pd.read_csv(shadow_path, header=None, names=["yyyymm", "sr"])
sr["period"] = pd.to_datetime(sr["yyyymm"].astype(str),
                               format="%Y%m").dt.to_period("M")
sr = sr.drop_duplicates("period").set_index("period")["sr"]

ff = pd.read_csv(ff_path, parse_dates=["observation_date"])
ff["period"] = ff["observation_date"].dt.to_period("M")
ff = ff.set_index("period")["FEDFUNDS"]

rf = sr.reindex(FULL_PERIOD)
rf[rf.isna()] = ff.reindex(FULL_PERIOD)[rf.isna()]
series["policy_rate"] = rf

s = rf.dropna()
print(f"  policy_rate  mean={s.mean():+.2f}%  min={s.min():+.2f}%  non-null={len(s)}")


# ============================================================
# STEP 3 — Shiller log(P/D) and unemployment gap from ex3
# ============================================================
print(f"\n{'─'*65}")
print("STEP 3 — Shiller P/D and unemployment gap from ex3_monthly")
print(f"{'─'*65}")

ex3 = pd.read_csv(EX3_CSV)
ex3["period"] = pd.PeriodIndex(ex3["date"], freq="M")
ex3 = ex3.set_index("period")

series["pd_ratio"] = ex3["pd_ratio"].reindex(FULL_PERIOD)
series["ugap"]     = ex3["ugap"].reindex(FULL_PERIOD)

for name in ["pd_ratio", "ugap"]:
    s = series[name].dropna()
    print(f"  {name:<12}  mean={s.mean():+.4f}  std={s.std():.4f}  non-null={len(s)}")


# ============================================================
# STEP 4 — Estimate VAR(4)
# ============================================================
print(f"\n{'─'*65}")
print(f"STEP 4 — Estimating VAR({VAR_LAGS})")
print(f"{'─'*65}")
print("  Order: [pi_core, pi_food, pi_energy, policy_rate, pd_ratio, ugap]")
print("  Identification: reduced-form residuals (no Cholesky), following FLR\n")

VAR_ORDER    = ["pi_core", "pi_food", "pi_energy",
                "policy_rate", "pd_ratio", "ugap"]
var_df       = pd.DataFrame({k: series[k] for k in VAR_ORDER},
                             index=FULL_PERIOD)
var_df_clean = var_df.dropna()

print(f"  VAR sample: {var_df_clean.index[0]} → {var_df_clean.index[-1]}"
      f"  (N={len(var_df_clean)})")

model  = VAR(var_df_clean)
result = model.fit(VAR_LAGS)
print(f"  AIC={result.aic:.4f}  BIC={result.bic:.4f}")

print("  Lag comparison:")
for lag in [1, 2, 3, 4]:
    r = model.fit(lag)
    flag = "  ← chosen (FLR)" if lag == VAR_LAGS else ""
    print(f"    VAR({lag}): AIC={r.aic:.3f}  BIC={r.bic:.3f}{flag}")


# ============================================================
# STEP 5 — Extract reduced-form residuals
# ============================================================
print(f"\n{'─'*65}")
print("STEP 5 — Extracting VAR residuals")
print(f"{'─'*65}")

resid = result.resid.copy()
resid.index = var_df_clean.index[VAR_LAGS:]

eps_core   = resid["pi_core"]
eps_energy = resid["pi_energy"]

sample_mask  = ((resid.index >= pd.Period(SAMPLE_START, "M")) &
                (resid.index <= pd.Period(SAMPLE_END,   "M")))
eps_core_s   = eps_core[sample_mask]
eps_energy_s = eps_energy[sample_mask]

print(f"  Full residual range : {resid.index[0]} → {resid.index[-1]}")
print(f"  Analysis window     : {SAMPLE_START} → {SAMPLE_END}"
      f"  (N={len(eps_core_s)})")
print()
print(f"  eps_core   std={eps_core_s.std():.4f}%/yr")
print(f"  eps_energy std={eps_energy_s.std():.4f}%/yr")
print(f"  Ratio energy/core: {eps_energy_s.std()/eps_core_s.std():.1f}×")
print(f"  Corr(eps_core, eps_energy) = {eps_core_s.corr(eps_energy_s):.4f}")

print("\n  Top 8 energy shocks:")
known_events = {
    "2005-09": "Katrina/Rita supply disruption",
    "2022-03": "Ukraine war supply shock",
    "2008-11": "GFC oil demand collapse",
    "2015-01": "Shale glut / OPEC price war",
    "2006-09": "Post-Katrina mean reversion",
    "2009-06": "Post-GFC demand recovery",
    "2008-10": "GFC financial panic",
    "2022-05": "European energy embargo",
}
for p in eps_energy_s.abs().sort_values(ascending=False).head(8).index:
    ev = known_events.get(str(p), "")
    print(f"    {str(p):<8}  {eps_energy_s[p]:>+8.2f}%/yr  {ev}")


# ============================================================
# STEP 6 — Demand/supply interacted shocks (NaN-safe partition)
# ============================================================
print(f"\n{'─'*65}")
print("STEP 6 — Demand/supply interacted shocks (NaN-safe)")
print(f"{'─'*65}")
print("  eps_energy_D = eps_energy if demand_dummy==1, else 0  (NaN if dummy NaN)")
print("  eps_energy_S = eps_energy if demand_dummy==0, else 0  (NaN if dummy NaN)")
print("  Partition: D + S = eps_energy on all non-NaN rows\n")

dummy_s = ex3["demand_dummy"].reindex(
    pd.PeriodIndex(eps_core_s.index.tolist(), freq="M")
)

# NaN-safe partition: set opposite regime to 0, keep NaN where dummy is NaN
eps_energy_D = (eps_energy_s
                .where(dummy_s == 1, other=0.0)
                .where(~dummy_s.isna(), other=np.nan))
eps_energy_S = (eps_energy_s
                .where(dummy_s == 0, other=0.0)
                .where(~dummy_s.isna(), other=np.nan))

n_D   = int((dummy_s == 1).sum())
n_S   = int((dummy_s == 0).sum())
n_nan = int(dummy_s.isna().sum())
print(f"  Demand months (1): {n_D}")
print(f"  Supply months (0): {n_S}")
print(f"  NaN (init period): {n_nan}")
print(f"  eps_energy_D std = {eps_energy_D.dropna().std():.4f}%/yr")
print(f"  eps_energy_S std = {eps_energy_S.dropna().std():.4f}%/yr")

# Partition check — non-NaN rows only
valid = ~dummy_s.isna()
diff  = (eps_energy_D[valid] + eps_energy_S[valid]
         - eps_energy_s[valid]).abs().max()
partition_ok = diff < 1e-6
print(f"  Partition D+S=energy (non-NaN): {'PASS' if partition_ok else 'FAIL'}"
      f"  (max diff={diff:.2e})")


# ============================================================
# STEP 7 — Update ex3_monthly.csv with shock columns
# ============================================================
print(f"\n{'─'*65}")
print("STEP 7 — Updating ex3_monthly.csv")
print(f"{'─'*65}")

for col in ["eps_core", "eps_energy", "eps_energy_D", "eps_energy_S"]:
    if col in ex3.columns:
        ex3 = ex3.drop(columns=[col])

eps_df = pd.DataFrame({
    "eps_core":     eps_core_s,
    "eps_energy":   eps_energy_s,
    "eps_energy_D": eps_energy_D,
    "eps_energy_S": eps_energy_S,
})
ex3 = ex3.join(eps_df, how="left")

ex3.index = ex3.index.astype(str)
ex3.index.name = "date"
if "date" in ex3.columns:
    ex3 = ex3.drop(columns=["date"])
ex3.reset_index().to_csv(EX3_CSV, index=False)

print(f"  SAVED: {EX3_CSV}")
print(f"  Rows: {len(ex3)}  |  Cols: {len(ex3.columns)}")
for col in ["eps_core", "eps_energy", "eps_energy_D", "eps_energy_S"]:
    s = ex3[col].dropna()
    print(f"    {col:<18}  N={len(s)}  mean={s.mean():+.4f}  std={s.std():.4f}")


# ============================================================
# FIGURE 21 — VAR Inflation Shocks (two panels)
# Panel (a): eps_core, color = sign of shock
# Panel (b): eps_energy, color = demand (blue) / supply (red) regime
# ============================================================
print(f"\n{'─'*65}")
print("Producing Fig21_inflation_shocks.png")
print(f"{'─'*65}")

ts_core   = eps_core_s.copy()
ts_energy = eps_energy_s.copy()
ts_dummy  = dummy_s.copy()
ts_core.index   = ts_core.index.to_timestamp()
ts_energy.index = ts_energy.index.to_timestamp()
ts_dummy.index  = ts_dummy.index.to_timestamp()

fig = plt.figure(figsize=(15, 11))
fig.subplots_adjust(left=0.07, right=0.70, top=0.94,
                    bottom=0.18, hspace=0.70)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

fig.text(0.07, 0.97, "F I G U R E   2 1",
         fontsize=8, color="gray", va="top", transform=fig.transFigure)

# ── Panel (a): eps_core ──────────────────────────────────────
ca = ["#2166ac" if v >= 0 else "#d7191c" for v in ts_core.values]
ax1.bar(ts_core.index, ts_core.values, color=ca, width=25, alpha=0.82, linewidth=0)
ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax1.set_ylabel("(%/yr)", fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(0.01, 0.97,
    r"(a)  Unexpected core inflation shock — $\varepsilon_{core,t}$"
    f"     [std = {ts_core.std():.3f}%/yr]",
    transform=ax1.transAxes, fontsize=11, fontweight="bold",
    style="italic", va="top")

# ── Panel (b): eps_energy ─────────────────────────────────────
cb = []
for t in ts_energy.index:
    d = ts_dummy[t] if t in ts_dummy.index else np.nan
    cb.append("#aaaaaa" if pd.isna(d)
              else ("#2166ac" if d == 1 else "#d7191c"))

ax2.bar(ts_energy.index, ts_energy.values, color=cb, width=25, alpha=0.82, linewidth=0)
ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax2.set_ylabel("(%/yr)", fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.text(0.01, 0.97,
    r"(b)  Unexpected energy inflation shock — $\varepsilon_{energy,t}$"
    f"     [std = {ts_energy.std():.1f}%/yr,  ratio vs core = "
    f"{ts_energy.std()/ts_core.std():.0f}×]",
    transform=ax2.transAxes, fontsize=11, fontweight="bold",
    style="italic", va="top")

# Legend outside right
patches = [
    mpatches.Patch(color="#aaaaaa", alpha=0.75,
        label="Init period\n(no regime yet)"),
    mpatches.Patch(color="#2166ac", alpha=0.75,
        label="(a) positive shock\n(b) demand regime"),
    mpatches.Patch(color="#d7191c", alpha=0.75,
        label="(a) negative shock\n(b) supply regime"),
]
ax2.legend(handles=patches, fontsize=9, frameon=True, framealpha=0.95,
           edgecolor="#cccccc", loc="upper left",
           bbox_to_anchor=(1.02, 2.10), borderaxespad=0)

ax2.text(0, -0.22,
    r"$\it{Notes}$:  $\varepsilon_{core,t}$ and $\varepsilon_{energy,t}$ are "
    r"reduced-form residuals from a 6-variable VAR(4) estimated on "
    r"Jan 1999 – Dec 2024, following Fang, Liu and Roussanov (2022). "
    r"MoM annualized inflation ($\pi_t = (P_t/P_{t-1}-1)\times12\times100$). "
    r"Both panels: May 2001 – Jun 2023.",
    transform=ax2.transAxes, fontsize=9, color="#444444",
    va="top", linespacing=1.6)
ax2.text(0, -0.32,
    r"Grey = init period (2001-05 to 2003-02, demand dummy not yet available). "
    r"Panel (a): colour = sign of shock. "
    r"Panel (b): colour = demand regime (blue, energy_R$_t$ > rolling 24m p70) "
    r"/ supply regime (red).",
    transform=ax2.transAxes, fontsize=9, color="#444444",
    va="top", linespacing=1.6)

plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Fig21 saved → {OUTPUT_FIG}")


# ============================================================
# FINAL VERIFICATION
# ============================================================
print(f"\n{'='*65}")
print("FINAL VERIFICATION")
print(f"{'='*65}\n")

ex3_check = pd.read_csv(EX3_CSV)
ex3_check["period"] = pd.PeriodIndex(ex3_check["date"], freq="M")
ex3_check = ex3_check.set_index("period")

checks = [
    ("eps_core in ex3",
     "eps_core" in ex3_check.columns, "column present"),
    ("eps_energy in ex3",
     "eps_energy" in ex3_check.columns, "column present"),
    ("eps_energy_D in ex3",
     "eps_energy_D" in ex3_check.columns, "column present"),
    ("eps_energy_S in ex3",
     "eps_energy_S" in ex3_check.columns, "column present"),
    ("eps_core std in [0.5, 3.0]",
     0.5 < ex3_check["eps_core"].dropna().std() < 3.0,
     f"std={ex3_check['eps_core'].dropna().std():.3f}"),
    ("eps_energy std > 20",
     ex3_check["eps_energy"].dropna().std() > 20,
     f"std={ex3_check['eps_energy'].dropna().std():.2f}"),
    ("Partition D+S=energy",
     partition_ok, f"max diff={diff:.2e}"),
    ("N eps_core >= 250",
     ex3_check["eps_core"].notna().sum() >= 250,
     f"N={ex3_check['eps_core'].notna().sum()}"),
]

all_ok = True
for label, ok, detail in checks:
    status = "OK  " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {status}  {label:<40} {detail}")

print(f"\n  {'All checks PASSED ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")
print(f"\n{'='*65}")
print("11_var_inflation_shocks_FINAL.py — DONE")
print(f"{'='*65}")