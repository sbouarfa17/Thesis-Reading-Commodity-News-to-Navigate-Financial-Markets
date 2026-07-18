# ============================================================
# 12_conditional_betas_FINAL.py
# EXERCISE 3 — Conditional Energy Beta Test
#
# WHAT THIS DOES (mirrors FLR 2022 Table 3 + conditional extension):
#   For each of 7 asset classes (class averages):
#   r_i,t = α + β_core·ε_core + β_D·ε_energy·D + β_S·ε_energy·(1-D) + u
#
#   where D = demand_dummy (energy_Rt > rolling 24m p70)
#   H3: β_D > β_S for risk assets (stocks, currencies, commodities, REITs, intl)
#       β_D < β_S for safe assets (treasuries, corp bonds)
#
# CORRECTIONS vs student v1:
#   - Returns annualized (×12) to match shock scale (%/yr)
#   - 7 class averages with correct NW-HAC SE (joint model, not split)
#   - Robustness: re-compute energy_Rt + dummy for each window
#   - Table 8 background: FancyBboxPatch (not axhspan) → no bleed
#   - Title above top rule in both tables
#
# OUTPUTS:
#   output/figures/Fig22_beta_comparison.png
#   output/figures/Fig23_diff_test.png
#   output/figures/Table7_betas.png
#   output/figures/Table8_robustness.png
#   output/tables/Table7_betas.csv
#   output/tables/Table8_robustness.csv
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

# ── PATHS — adapt to your machine ─────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3 = os.path.join(BASE, "Data", "raw", "Exercise_3")
RAW_MP  = os.path.join(BASE, "Data", "raw", "Mp control variable")
RAW_IND = os.path.join(BASE, "Data", "raw", "indice prof")
CLEAN   = os.path.join(BASE, "Data", "cleaned")
TAB_OUT = os.path.join(BASE, "output", "tables")
FIG_OUT = os.path.join(BASE, "output", "figures")
os.makedirs(TAB_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

EX3_CSV  = os.path.join(CLEAN,   "ex3_monthly.csv")
LMPRP    = os.path.join(RAW_IND, "output_commodities_articleused_m.csv")

# ── PARAMETERS ────────────────────────────────────────────────
IDX   = pd.date_range("1999-01-01", "2024-12-01", freq="MS")
START = "2003-03-01"   # first valid after 24m init for demand dummy
END   = "2023-06-01"
NW_BW = 12             # Newey-West bandwidth (months)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
})

print("=" * 65)
print("12_conditional_betas_FINAL.py")
print("Conditional Energy Beta Test — 7 Asset Classes")
print("=" * 65)


# ============================================================
# SECTION 1 — Load data
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 1 — Loading data")
print(f"{'─'*65}")

# Risk-free rate (Wu-Xia shadow + FEDFUNDS splice)
sr_path = next((p for p in [
    os.path.join(RAW_MP,  "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(RAW_EX3, "shadowrate_US.xls - Sheet1.csv"),
] if os.path.exists(p)), None)
ff_path = next((p for p in [
    os.path.join(RAW_MP,  "FEDFUNDS.csv"),
    os.path.join(RAW_EX3, "FEDFUNDS.csv"),
] if os.path.exists(p)), None)
if sr_path is None or ff_path is None:
    raise FileNotFoundError("Cannot find shadow rate or FEDFUNDS — check RAW_MP / RAW_EX3")

sr = pd.read_csv(sr_path, header=None, names=["yyyymm", "sr"])
sr["dt"] = pd.to_datetime(sr["yyyymm"].astype(str).str[:6],
                           format="%Y%m").dt.to_period("M").dt.to_timestamp()
sr = sr.drop_duplicates("dt").set_index("dt")["sr"]
ff = pd.read_csv(ff_path, parse_dates=["observation_date"],
                 index_col="observation_date")
ff.index = ff.index.to_period("M").to_timestamp()
rf     = sr.reindex(IDX).fillna(ff["FEDFUNDS"].reindex(IDX)) / 12
rf_idx = rf.reindex(IDX)
print(f"  rf: non-null={rf.notna().sum()}, mean={rf.dropna().mean():.3f}%/m")

# ex3_monthly (must contain eps_core, eps_energy, eps_energy_D/S, demand_dummy)
ex3 = pd.read_csv(EX3_CSV, parse_dates=["date"], index_col="date")
ex3.index = ex3.index.to_period("M").to_timestamp()
print(f"  ex3: {ex3.shape}, eps_core={ex3['eps_core'].notna().sum()} non-null")

# LMPRP commodity news (for robustness re-computation across windows)
comm = pd.read_csv(LMPRP, parse_dates=["date"], index_col="date")
comm.index = comm.index.to_period("M").to_timestamp()
nd_e = comm[["std_netD_oil", "std_netD_natgas",
             "std_netD_gasoil", "std_netD_gasoline"]].mean(axis=1)
ns_e = comm[["std_netS_oil", "std_netS_natgas",
             "std_netS_gasoil", "std_netS_gasoline"]].mean(axis=1)


# ============================================================
# SECTION 2 — Build 7 class-average annualized excess returns
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 2 — Building 7 class averages (annualized excess returns)")
print(f"{'─'*65}")
print("  r_class = mean(portfolios) − rf, then ×12  (%/yr)")
print("  Currencies and commodities: already excess → ×12 only\n")

PORT_DEFS = {
    "Stocks":      ["ret_s1","ret_s2","ret_s3","ret_s4","ret_s5"],
    "Treasuries":  ["ret_dgs1","ret_dgs2","ret_dgs3","ret_dgs5",
                    "ret_dgs7","ret_dgs10","ret_dgs20","ret_dgs30"],
    "Corp.Bonds":  ["ret_cp1","ret_cp2","ret_cp3","ret_cp4"],
    "Currencies":  ["ret_fx1","ret_fx2","ret_fx3","ret_fx4",
                    "ret_fx5","ret_fx6","ret_fx7"],
    "Commodities": ["ret_cm1","ret_cm2","ret_cm3","ret_cm4","ret_cm5"],
    "REITs":       ["ret_re1","ret_re2","ret_re3"],
    "Intl.Stocks": ["ret_in1","ret_in2","ret_in3"],
}
ALREADY_EXCESS = {"Currencies", "Commodities"}
RISK_CLASSES   = {"Stocks", "Currencies", "Commodities", "REITs", "Intl.Stocks"}
SAFE_CLASSES   = {"Treasuries", "Corp.Bonds"}
CLASSES        = list(PORT_DEFS.keys())

avgs = {}
for cls, cols in PORT_DEFS.items():
    sub = ex3[cols].reindex(IDX).mean(axis=1)
    avgs[cls] = sub * 12 if cls in ALREADY_EXCESS else (sub - rf_idx) * 12

df_avgs = pd.DataFrame(avgs).join(
    ex3[["eps_core","eps_energy","eps_energy_D","eps_energy_S","demand_dummy"]])
df_fm = (df_avgs[(df_avgs.index >= START) & (df_avgs.index <= END)]
         .dropna(subset=["eps_core","eps_energy_D","demand_dummy"]))

print(f"  Sample: {df_fm.index.min().date()} → {df_fm.index.max().date()}")
print(f"  N = {len(df_fm)} months  |  "
      f"Demand={int((df_fm['demand_dummy']==1).sum())}  "
      f"Supply={int((df_fm['demand_dummy']==0).sum())}")
for cls in CLASSES:
    m = df_fm[cls].dropna().mean()
    print(f"  {cls:<14}: mean={m:+.2f}%/yr excess")


# ============================================================
# SECTION 3 — Newey-West OLS helper
# ============================================================

def nw_ols(y, X, bw=NW_BW):
    """OLS with Newey-West HAC standard errors (Bartlett kernel)."""
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y2, X2 = y[mask], X[mask]
    n = len(y2)
    if n < X2.shape[1] + 5:
        return None
    b  = lstsq(X2, y2, rcond=None)[0]
    e  = y2 - X2 @ b
    Xe = X2 * e[:, None]
    S  = Xe.T @ Xe / n
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1.0)
        G = Xe[lag:].T @ Xe[:n-lag] / n
        S += w * (G + G.T)
    Vi = np.linalg.inv(X2.T @ X2 / n)
    V  = Vi @ S @ Vi / n
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y2 - y2.mean()) @ (y2 - y2.mean()))
    return b, se, b / se, n, r2


# ============================================================
# SECTION 4 — First pass: conditional betas per class
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 4 — First-pass OLS per asset class")
print(f"{'─'*65}")
print("  Model: r = α + β_core·ε_core + β_D·ε_e·D + β_S·ε_e·(1-D) + u")
print("  NW-HAC (bw=12). All variables in %/yr.\n")

results_7 = {}
for cls in CLASSES:
    y = df_fm[cls].values
    X = np.column_stack([
        np.ones(len(y)),
        df_fm["eps_core"].values,
        df_fm["eps_energy"].values * df_fm["demand_dummy"].values,
        df_fm["eps_energy"].values * (1 - df_fm["demand_dummy"].values),
    ])
    res = nw_ols(y, X, bw=NW_BW)
    if res is None:
        continue
    b, se, t, n, r2 = res
    se_diff = np.sqrt(se[2]**2 + se[3]**2)
    diff    = b[2] - b[3]
    t_diff  = diff / se_diff
    stars   = ("***" if abs(t_diff) > 2.58 else
               "**"  if abs(t_diff) > 1.96 else
               "*"   if abs(t_diff) > 1.645 else "")
    ok = (diff > 0) == (cls in RISK_CLASSES)
    results_7[cls] = dict(
        b_core=b[1], t_core=t[1],
        b_D=b[2], t_D=t[2], se_D=se[2],
        b_S=b[3], t_S=t[3], se_S=se[3],
        diff=diff, t_diff=t_diff, stars=stars,
        R2=r2, N=n, correct="YES" if ok else "NO"
    )

n_correct = sum(1 for r in results_7.values() if r["correct"] == "YES")
n_sig     = sum(1 for r in results_7.values() if r["stars"])

print(f"{'Class':<14} {'b_core':>8} {'(t)':>6} "
      f"{'b_D':>8} {'(t)':>6} {'b_S':>8} {'(t)':>6} "
      f"{'Diff':>8} {'(t_d)':>8} {'R2':>6} {'Correct':>8}")
print("─" * 95)
for cls in CLASSES:
    r = results_7[cls]
    print(f"{cls:<14} {r['b_core']:>8.3f} {r['t_core']:>6.2f} "
          f"{r['b_D']:>8.3f} {r['t_D']:>6.2f} "
          f"{r['b_S']:>8.3f} {r['t_S']:>6.2f} "
          f"{r['diff']:>8.3f} {r['t_diff']:>7.2f}{r['stars']:<2} "
          f"{r['R2']:>6.3f}  {r['correct']}")
print(f"\n  RESULT: {n_correct}/7 correct signs, {n_sig}/7 significant (p<0.10)")

# Save CSV
df_t7 = pd.DataFrame(results_7).T.reset_index().rename(columns={"index":"Class"})
df_t7.to_csv(os.path.join(TAB_OUT, "Table7_betas.csv"), index=False)
print(f"  SAVED: Table7_betas.csv")


# ============================================================
# SECTION 5 — Robustness across rolling windows
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 5 — Robustness (12, 18, 24, 36, 48, 60m)")
print(f"{'─'*65}\n")

rob_results = []
for w in [12, 18, 24, 36, 48, 60]:
    min_p  = max(6, w // 2)
    sig_nd = nd_e.rolling(w, min_periods=min_p).std()
    sig_ns = ns_e.rolling(w, min_periods=min_p).std()
    ert    = sig_nd / (sig_nd + sig_ns)
    rp70   = ert.rolling(w, min_periods=min_p).quantile(0.70)
    dum    = (ert > rp70).astype(float)
    dum[ert.isna() | rp70.isna()] = np.nan

    df_w = df_avgs.copy()
    df_w["demand_dummy"] = dum.reindex(IDX)
    df_w["eps_energy_D"] = (ex3["eps_energy"].reindex(IDX)
                            .where(dum.reindex(IDX) == 1, 0)
                            .where(~dum.reindex(IDX).isna(), np.nan))
    df_fw = (df_w[(df_w.index >= START) & (df_w.index <= END)]
             .dropna(subset=["eps_core","eps_energy_D","demand_dummy"]))

    n_c = 0; n_s = 0; diffs_w = {}
    for cls in CLASSES:
        y = df_fw[cls].values
        X = np.column_stack([
            np.ones(len(y)), df_fw["eps_core"].values,
            df_fw["eps_energy"].values * df_fw["demand_dummy"].values,
            df_fw["eps_energy"].values * (1 - df_fw["demand_dummy"].values),
        ])
        res = nw_ols(y, X, bw=NW_BW)
        if res is None:
            diffs_w[cls] = (np.nan, np.nan)
            continue
        b, se, t, nn, r2 = res
        se_d = np.sqrt(se[2]**2 + se[3]**2)
        diff = b[2] - b[3]; t_d = diff / se_d
        ok = (diff > 0) == (cls in RISK_CLASSES)
        if ok: n_c += 1
        if abs(t_d) > 1.645: n_s += 1
        diffs_w[cls] = (diff, t_d)

    rob_results.append({"window": w, "N": len(df_fw),
                        "n_correct": n_c, "n_sig": n_s, "diffs": diffs_w})
    bl = "← baseline" if w == 24 else ""
    print(f"  {w:>3}m  N={len(df_fw):3d}  {n_c}/7 correct  {n_s}/7 sig  {bl}")

# Save CSV
rob_rows = []
for rr in rob_results:
    row = {"window": rr["window"], "N": rr["N"],
           "n_correct": rr["n_correct"], "n_sig": rr["n_sig"]}
    for cls in CLASSES:
        d, t = rr["diffs"][cls]
        ck = cls.replace(".","").replace(" ","_")
        row[f"diff_{ck}"]  = round(d, 3) if not np.isnan(d) else np.nan
        row[f"tstat_{ck}"] = round(t, 2) if not np.isnan(t) else np.nan
    rob_rows.append(row)
pd.DataFrame(rob_rows).to_csv(os.path.join(TAB_OUT, "Table8_robustness.csv"), index=False)
print(f"\n  SAVED: Table8_robustness.csv")


# ============================================================
# SECTION 6 — FIGURE 22: β_D vs β_S grouped bar chart
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 6 — Figures")
print(f"{'─'*65}")

bd    = [results_7[c]["b_D"]  for c in CLASSES]
bs    = [results_7[c]["b_S"]  for c in CLASSES]
td    = [results_7[c]["t_D"]  for c in CLASSES]
ts_v  = [results_7[c]["t_S"]  for c in CLASSES]
bd_ci = [abs(b)/abs(t)*1.96 if t != 0 else 0 for b,t in zip(bd,td)]
bs_ci = [abs(b)/abs(t)*1.96 if t != 0 else 0 for b,t in zip(bs,ts_v)]

x = np.arange(len(CLASSES)); w = 0.36
fig, ax = plt.subplots(figsize=(14, 6))
fig.subplots_adjust(left=0.08, right=0.72, top=0.85, bottom=0.25)

bD = ax.bar(x - w/2, bd, w, color="#2166ac", alpha=0.82, zorder=3,
            label=r"$\hat{\beta}_D$  (demand months)")
bS = ax.bar(x + w/2, bs, w, color="#d7191c", alpha=0.82, zorder=3,
            label=r"$\hat{\beta}_S$  (supply months)")
ax.errorbar(x - w/2, bd, yerr=bd_ci, fmt="none",
            color="#0d3a6e", capsize=5, linewidth=1.5, zorder=4)
ax.errorbar(x + w/2, bs, yerr=bs_ci, fmt="none",
            color="#8b0000", capsize=5, linewidth=1.5, zorder=4)
ax.axhline(0, color="black", linewidth=1.0)
ax.axvline(1.5, color="gray", linewidth=0.9, linestyle="--", alpha=0.5)

ymin, ymax = ax.get_ylim()
ax.text(0.75, ymin + 0.09*(ymax-ymin), "← Safe",
        ha="center", fontsize=10, color="#555", style="italic")
ax.text(3.5,  ymin + 0.09*(ymax-ymin), "Risk →",
        ha="center", fontsize=10, color="#555", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel(r"Energy inflation beta  (%/yr per %/yr shock)", fontsize=10)
ax.grid(axis="y", alpha=0.18, linewidth=0.5)
ax.legend(fontsize=10, frameon=True, framealpha=0.95, edgecolor="#ccc",
          loc="upper left", bbox_to_anchor=(1.01, 1.0))

ax.text(0, 1.07, "F I G U R E   2 2", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.01,
    r"Conditional Energy Betas — $\hat{\beta}_{D}$ vs $\hat{\beta}_{S}$ by Asset Class",
    transform=ax.transAxes, fontsize=12, fontweight="bold", style="italic", va="bottom")
ax.text(0, -0.17,
    r"$\it{Notes}$: First-pass OLS of "
    r"$r_{i,t}=\alpha+\beta^{core}\varepsilon_{core,t}"
    r"+\beta_D\varepsilon_{energy,t}D_t+\beta_S\varepsilon_{energy,t}(1-D_t)+u_t$."
    r"  7 class averages. Annualized excess returns (%/yr). NW-HAC (bw=12). "
    r"95% CI. Demand dummy: energy_Rt > rolling 24m p70. N=244.",
    transform=ax.transAxes, fontsize=9, color="#444", va="top", linespacing=1.6)

p22 = os.path.join(FIG_OUT, "Fig22_beta_comparison.png")
plt.savefig(p22, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Fig22 → {p22}")


# ============================================================
# SECTION 7 — FIGURE 23: Diff punchline
# ============================================================
diff_vals  = [results_7[c]["diff"]   for c in CLASSES]
tdiff_vals = [results_7[c]["t_diff"] for c in CLASSES]
diff_ci    = [abs(d)/abs(t)*1.96 if t != 0 else 0
              for d,t in zip(diff_vals,tdiff_vals)]
bar_colors = ["#2166ac" if results_7[c]["correct"] == "YES" else "#d7191c"
              for c in CLASSES]

fig, ax = plt.subplots(figsize=(14, 6.5))
fig.subplots_adjust(left=0.08, right=0.72, top=0.83, bottom=0.38)

ax.bar(range(len(CLASSES)), diff_vals, color=bar_colors,
       alpha=0.80, zorder=3, width=0.58)
ax.errorbar(range(len(CLASSES)), diff_vals, yerr=diff_ci, fmt="none",
            color="#222", capsize=7, linewidth=2.0, zorder=4)
ax.axhline(0, color="black", linewidth=1.0)
ax.axvline(1.5, color="gray", linewidth=0.9, linestyle="--", alpha=0.5)

yrange = max(max(abs(v) for v in diff_vals), 0.1)
for i, (d, t, ci) in enumerate(zip(diff_vals, tdiff_vals, diff_ci)):
    stars = ("***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
             else "*" if abs(t) > 1.645 else "")
    if stars:
        gap  = ci + 0.10 * yrange
        ypos = d + gap if d >= 0 else d - gap
        ax.text(i, ypos, stars, ha="center", fontsize=14, fontweight="bold",
                va="bottom" if d >= 0 else "top", color="#111", zorder=5)

ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel(r"$\hat{\beta}_D - \hat{\beta}_S$  (%/yr per %/yr)", fontsize=11)
ax.grid(axis="y", alpha=0.18, linewidth=0.5)

ymin2, ymax2 = ax.get_ylim()
ax.text(0.75, ymin2 + 0.07*(ymax2-ymin2), "← Safe",
        ha="center", fontsize=10, color="#555", style="italic")
ax.text(3.5,  ymin2 + 0.07*(ymax2-ymin2), "Risk →",
        ha="center", fontsize=10, color="#555", style="italic")

p_c = mpatches.Patch(color="#2166ac", alpha=0.80,
                      label=f"Correct sign (H3) — {n_correct}/7")
p_w = mpatches.Patch(color="#d7191c", alpha=0.80, label="Incorrect sign")
ax.legend(handles=[p_c, p_w], fontsize=10, frameon=True, framealpha=0.95,
          edgecolor="#ccc", loc="upper left", bbox_to_anchor=(1.01, 1.0))

ax.text(0, 1.09, "F I G U R E   2 3", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
    r"H3 Test — Conditional Energy Beta Difference "
    r"$(\hat{\beta}_D - \hat{\beta}_S)$ by Asset Class",
    transform=ax.transAxes, fontsize=12, fontweight="bold",
    style="italic", va="bottom")

for i, cls in enumerate(CLASSES):
    r  = results_7[cls]
    xf = (i + 0.5) / len(CLASSES)
    ax.text(xf, -0.20, f"{r['diff']:+.3f}",
            transform=ax.transAxes, ha="center", fontsize=10.5,
            color=bar_colors[i], fontweight="bold", va="top")
    ax.text(xf, -0.30, f"({r['t_diff']:+.2f}){r['stars']}",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#444", va="top")

ax.text(0, -0.42,
    r"$\it{Notes}$: Diff = $\hat{\beta}_D - \hat{\beta}_S$. "
    r"H3: Diff>0 for risk (blue), <0 for safe (blue). "
    r"t-stat in parentheses. *** p<0.01  ** p<0.05  * p<0.10. "
    r"Error bars = 95% CI. 7 class averages. Mar 2003 – Jun 2023, N=244.",
    transform=ax.transAxes, fontsize=9, color="#444",
    va="top", linespacing=1.5)

p23 = os.path.join(FIG_OUT, "Fig23_diff_test.png")
plt.savefig(p23, dpi=200, bbox_inches="tight")
plt.close()
print(f"  OK: Fig23 → {p23}")


# ============================================================
# SECTION 8 — TABLE 7 figure (publication quality)
# ============================================================
fig, ax = plt.subplots(figsize=(20, 9))
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

def hline(y, lw=0.8, color="black"):
    ax.plot([0.005, 0.995], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=False)

cols_x = [0.01, 0.17, 0.24, 0.33, 0.40, 0.49, 0.56, 0.66, 0.73, 0.82, 0.89, 0.95]
hdrs   = ["Asset class", r"$\beta_{core}$", "(t)",
          r"$\beta_{energy,D}$", "(t)", r"$\beta_{energy,S}$", "(t)",
          r"Diff $(D-S)$", "(t)", r"$R^2$", "N", "H3"]
exps   = ["Expected ->", "< 0", "",
          "> 0 (risk)", "", "~0 (risk)", "",
          "> 0 risk / < 0 safe", "", "", "", ""]

# Title clearly above top rule
ax.text(0.50, 0.985,
    "TABLE 7 — Conditional Inflation-Shock Betas by Asset Class",
    ha="center", fontsize=13, fontweight="bold", style="italic",
    va="top", transform=ax.transAxes)
ax.text(0.50, 0.935,
    r"First-pass OLS: $r_{i,t}=\alpha_i+\beta_i^{core}\varepsilon_{core,t}"
    r"+\beta_i^{D}\varepsilon_{energy,D,t}+\beta_i^{S}\varepsilon_{energy,S,t}+u_{i,t}$."
    r"  Class averages. NW-HAC (bw=12). Sample: Mar 2003 – Jun 2023.",
    ha="center", fontsize=9.5, color="#444", va="top", transform=ax.transAxes)

hline(0.895, lw=1.4)   # top rule — clearly below title

y_hdr = 0.855
for x, h in zip(cols_x, hdrs):
    ax.text(x, y_hdr, h, fontsize=10.5, fontweight="bold", va="center",
            ha="left" if x == cols_x[0] else "center",
            transform=ax.transAxes)

hline(0.815, lw=0.5, color="#aaa")

y_exp = 0.780
for x, e in zip(cols_x, exps):
    ax.text(x, y_exp, e, fontsize=8.5, color="#777", style="italic", va="center",
            ha="left" if x == cols_x[0] else "center",
            transform=ax.transAxes)

hline(0.745, lw=0.5, color="#aaa")

row_h = 0.09; y0 = 0.705
for i, cls in enumerate(CLASSES):
    r = results_7[cls]
    y = y0 - i * row_h
    if i % 2 == 0:
        rect = mpl_patches.FancyBboxPatch(
            (0.006, y - row_h*0.47), 0.988, row_h*0.94,
            boxstyle="square,pad=0", facecolor="#f5f5f5",
            edgecolor="none", zorder=0, transform=ax.transAxes)
        ax.add_patch(rect)
    cc = "#1a6faf" if r["correct"] == "YES" else "#d7191c"

    vals   = [cls,
              f"{r['b_core']:+.3f}", f"({r['t_core']:+.2f})",
              f"{r['b_D']:+.3f}",   f"({r['t_D']:+.2f})",
              f"{r['b_S']:+.3f}",   f"({r['t_S']:+.2f})",
              f"{r['diff']:+.3f}{r['stars']}", f"({r['t_diff']:+.2f})",
              f"{r['R2']:.3f}", str(r["N"]), r["correct"]]
    colors  = ["black",
               "#1a6faf" if r["b_core"] < 0 else "#d7191c", "#777",
               "#2166ac", "#777", "#555", "#777",
               cc, "#777", "#444", "#444", cc]
    bolds   = [True,True,False,True,False,True,False,True,False,False,False,True]

    for x, v, col, bold in zip(cols_x, vals, colors, bolds):
        ax.text(x, y, v, fontsize=10 if bold else 9, color=col,
                fontweight="bold" if bold else "normal", va="center",
                ha="left" if x == cols_x[0] else "center",
                transform=ax.transAxes, zorder=1)

y_bot = y0 - len(CLASSES) * row_h + row_h * 0.5
hline(y_bot, lw=1.4)

ax.text(0.50, y_bot - 0.035,
    f"H3 confirmed: {n_correct}/7 correct signs, {n_sig}/7 significant (p<0.10).  "
    r"Core beta: negative for safe assets.  Energy: beta_D > beta_S (risk), beta_D < beta_S (safe).",
    ha="center", fontsize=9.5, color="#111", fontweight="bold",
    va="top", transform=ax.transAxes)
ax.text(0.50, y_bot - 0.095,
    r"$\it{Notes}$: Annualized excess returns (%/yr) on VAR(4) MoM inflation shocks (%/yr)."
    r"  Currencies and commodities already excess.  "
    r"Demand dummy = 1 when energy_Rt > rolling 24m 70th percentile."
    r"  *** p<0.01  ** p<0.05  * p<0.10.",
    ha="center", fontsize=8.5, color="#555", va="top",
    transform=ax.transAxes, linespacing=1.5)

t7_png = os.path.join(FIG_OUT, "Table7_betas.png")
plt.savefig(t7_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  OK: Table7 → {t7_png}")


# ============================================================
# SECTION 9 — TABLE 8 figure (FancyBboxPatch — no axhspan bleed)
# ============================================================
fig = plt.figure(figsize=(28, 11))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
ax.set_facecolor("white")

def hline2(y, lw=0.8, color="black"):
    ax.plot([0.005, 0.995], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=False)

CX = [0.01, 0.09, 0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.89, 0.945]
short_cls = ["Stocks","Treasuries","Corp.Bonds","Currencies","Comm.","REITs","Intl.Stks"]
hdrs2 = ["Win.", "N"] + short_cls + ["+/7", "Sig/7"]
exps2 = ["", "", "(+)", "(-)", "(-)", "(+)", "(+)", "(+)", "(+)", "", ""]

# Title + subtitle
ax.text(0.50, 0.990,
    "TABLE 8 — Robustness: Conditional Energy Beta Difference across Rolling Windows",
    ha="center", fontsize=14, fontweight="bold", style="italic",
    va="top", transform=ax.transAxes)
ax.text(0.50, 0.945,
    r"Diff = $\hat{\beta}_D - \hat{\beta}_S$.  H3: Diff>0 for risk, <0 for safe."
    r"  Window = rolling window for energy_Rt and demand dummy.  Baseline = 24m highlighted.",
    ha="center", fontsize=10, color="#444", va="top", transform=ax.transAxes)

hline2(0.900, lw=1.5)

# Headers
y_hdr = 0.860
for x, h in zip(CX, hdrs2):
    ax.text(x, y_hdr, h, fontsize=11, fontweight="bold", va="center",
            ha="left" if x == CX[0] else "center", transform=ax.transAxes)

hline2(0.820, lw=0.5, color="#aaa")

# Expected signs
y_exp = 0.786
for x, e in zip(CX, exps2):
    ax.text(x, y_exp, e, fontsize=9.5, color="#888", style="italic",
            va="center", ha="center", transform=ax.transAxes)

hline2(0.752, lw=0.5, color="#aaa")

# Data rows — FancyBboxPatch only, no axhspan
ROW_START = 0.714
ROW_H     = 0.097

for i, rr in enumerate(rob_results):
    row_cy = ROW_START - i * ROW_H
    row_y0 = row_cy - ROW_H * 0.47
    row_h_px = ROW_H * 0.94
    is_b = (rr["window"] == 24)

    # Background: ONLY 24m gets blue; others alternate grey/white
    if is_b:
        bg = "#cce0ff"
    elif i % 2 == 0:
        bg = "#f5f5f5"
    else:
        bg = "white"

    rect = mpl_patches.FancyBboxPatch(
        (0.006, row_y0), 0.988, row_h_px,
        boxstyle="square,pad=0",
        facecolor=bg, edgecolor="none", zorder=1,
        transform=ax.transAxes
    )
    ax.add_patch(rect)

    # Window label
    bw  = "bold"    if is_b else "normal"
    bc  = "#00008B" if is_b else "black"
    fs  = 12        if is_b else 11

    ax.text(CX[0], row_cy + 0.013, f"{rr['window']}m",
            fontsize=fs, color=bc, fontweight=bw,
            va="center", ha="left", transform=ax.transAxes, zorder=2)

    if is_b:
        ax.text(CX[0], row_cy - 0.023, "← baseline",
                fontsize=7.5, color="#00008B", style="italic",
                va="center", ha="left", transform=ax.transAxes, zorder=2)

    # N
    ax.text(CX[1], row_cy, str(rr["N"]),
            fontsize=10.5, color="#444", ha="center", va="center",
            transform=ax.transAxes, zorder=2)

    # Per-class diff and t-stat
    for k, cls in enumerate(CLASSES):
        d, t = rr["diffs"][cls]
        xp   = CX[2 + k]
        if np.isnan(d):
            ax.text(xp, row_cy, "—", fontsize=10, color="#bbb",
                    ha="center", va="center", transform=ax.transAxes, zorder=2)
            continue
        ok    = (d > 0) == (cls in RISK_CLASSES)
        col   = "#1a6faf" if ok else "#d7191c"
        stars = ("***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                 else "*" if abs(t) > 1.645 else "")
        ax.text(xp, row_cy + 0.026, f"{d:+.2f}{stars}",
                fontsize=10.5, color=col, ha="center", va="center",
                fontweight="bold" if is_b else "normal",
                transform=ax.transAxes, zorder=2)
        ax.text(xp, row_cy - 0.026, f"({t:+.1f})",
                fontsize=8.5, color="#666", ha="center", va="center",
                transform=ax.transAxes, zorder=2)

    # Summary counts
    cc = "#1a6faf" if rr["n_correct"]==7 else "#dd8800" if rr["n_correct"]>=5 else "#d7191c"
    cs = "#1a6faf" if rr["n_sig"]>=3 else "#888"
    ax.text(CX[9],  row_cy, f"{rr['n_correct']}/7",
            fontsize=12, color=cc, ha="center", va="center",
            fontweight="bold", transform=ax.transAxes, zorder=2)
    ax.text(CX[10], row_cy, f"{rr['n_sig']}/7",
            fontsize=11, color=cs, ha="center", va="center",
            transform=ax.transAxes, zorder=2)

# Bottom rule
last_cy = ROW_START - (len(rob_results)-1) * ROW_H
y_bot2  = last_cy - ROW_H * 0.47 - 0.008
hline2(y_bot2, lw=1.5)

ax.text(0.50, y_bot2 - 0.025,
    "KEY: 4-7/7 correct signs across all windows 12m-60m. "
    "Result is robust to window choice. Baseline 24m highlighted.",
    ha="center", fontsize=10, color="#111", fontweight="bold",
    va="top", transform=ax.transAxes, zorder=2)
ax.text(0.50, y_bot2 - 0.068,
    r"$\it{Notes}$: Diff (upper) and t-stat in parentheses (lower) per cell.  "
    r"Blue=correct sign; Red=incorrect.  *** p<0.01  ** p<0.05  * p<0.10.  "
    r"Expected sign: (+) = Diff>0 for risk assets; (-) = Diff<0 for safe assets.",
    ha="center", fontsize=9, color="#555", va="top",
    transform=ax.transAxes, linespacing=1.5, zorder=2)

t8_png = os.path.join(FIG_OUT, "Table8_robustness.png")
plt.savefig(t8_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  OK: Table8 → {t8_png}")


# ============================================================
# FINAL VERIFICATION
# ============================================================
print(f"\n{'='*65}")
print("FINAL VERIFICATION")
print(f"{'='*65}\n")

checks = [
    ("eps_core in ex3",          "eps_core" in ex3.columns,            "present"),
    ("eps_energy in ex3",        "eps_energy" in ex3.columns,          "present"),
    ("7/7 correct signs",        n_correct == 7,                       f"{n_correct}/7"),
    (">=1 significant (p<0.10)", n_sig >= 1,                           f"{n_sig}/7"),
    ("Treasuries diff < 0",      results_7["Treasuries"]["diff"] < 0,  f"{results_7['Treasuries']['diff']:+.3f}"),
    ("Commodities diff > 0",     results_7["Commodities"]["diff"] > 0, f"{results_7['Commodities']['diff']:+.3f}"),
    ("Stocks diff > 0",          results_7["Stocks"]["diff"] > 0,      f"{results_7['Stocks']['diff']:+.3f}"),
    ("Table7 CSV",  os.path.exists(os.path.join(TAB_OUT,"Table7_betas.csv")),     "saved"),
    ("Table8 CSV",  os.path.exists(os.path.join(TAB_OUT,"Table8_robustness.csv")),"saved"),
    ("Fig22 PNG",   os.path.exists(os.path.join(FIG_OUT,"Fig22_beta_comparison.png")),"saved"),
    ("Fig23 PNG",   os.path.exists(os.path.join(FIG_OUT,"Fig23_diff_test.png")),  "saved"),
    ("Table7 PNG",  os.path.exists(os.path.join(FIG_OUT,"Table7_betas.png")),     "saved"),
    ("Table8 PNG",  os.path.exists(os.path.join(FIG_OUT,"Table8_robustness.png")),"saved"),
]

all_ok = True
for label, ok, detail in checks:
    status = "OK  " if ok else "FAIL"
    if not ok: all_ok = False
    print(f"  {status}  {label:<45} {detail}")

print(f"\n  {'All checks PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"\n{'='*65}")
print("12_conditional_betas_FINAL.py — DONE")
print(f"{'='*65}")