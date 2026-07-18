# ============================================================
# 13_fama_macbeth_FINAL.py
# EXERCISE 3 — Fama-MacBeth Second Pass + Conditional Evidence
#
# STRUCTURE:
#   Step 1 — Unconditional first-pass betas (β_core, β_energy)
#   Step 2 — FM second pass: λ_t for each month t
#   Step 3 — Conditional FM: λ in demand vs supply months
#   Step 4 — Time-series sensitivity test (7/7 evidence for H3)
#   Step 5 — Table 9 + Fig24 (λ_t series) + Fig25 (sensitivity)
#
# HONEST NOTE (documented in thesis):
#   The FM second pass is underpowered in our 2003-2023 sample.
#   Root cause: corr(β_core, β_energy) = 0.98 — near-perfect
#   collinearity in the post-2003 demand-dominated era.
#   FLR need 680 months + 38 portfolios + stagflation episodes
#   to achieve FM power. Primary H3 evidence comes from:
#     (1) First-pass beta difference β_D > β_S (script 12, 7/7)
#     (2) Time-series sensitivity: 7/7 classes show stronger
#         corr(r_i, ε_energy) in demand months than supply months
#
# OUTPUTS:
#   output/tables/Table9_FM.csv
#   output/figures/Table9_FM.png
#   output/figures/Fig24_lambda_series.png
#   output/figures/Fig25_sensitivity.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import matplotlib.dates as mdates
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

# ── PATHS — adapt to your machine ─────────────────────────────
BASE     = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_EX3  = os.path.join(BASE, "Data", "raw", "Exercise_3")
RAW_MP   = os.path.join(BASE, "Data", "raw", "Mp control variable")
CLEAN    = os.path.join(BASE, "Data", "cleaned")
TAB_OUT  = os.path.join(BASE, "output", "tables")
FIG_OUT  = os.path.join(BASE, "output", "figures")
os.makedirs(TAB_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

EX3_CSV = os.path.join(CLEAN, "ex3_monthly.csv")

# ── PARAMETERS ────────────────────────────────────────────────
IDX   = pd.date_range("1999-01-01", "2024-12-01", freq="MS")
START = "2003-03-01"
END   = "2023-06-01"
NW_BW = 12

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

print("=" * 65)
print("13_fama_macbeth_FINAL.py")
print("Fama-MacBeth Price-of-Risk + Conditional Evidence")
print("=" * 65)


# ============================================================
# SECTION 1 — Load data
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 1 — Loading data")
print(f"{'─'*65}")

sr_path = next((p for p in [
    os.path.join(RAW_MP,  "shadowrate_US.xls - Sheet1.csv"),
    os.path.join(RAW_EX3, "shadowrate_US.xls - Sheet1.csv"),
] if os.path.exists(p)), None)
ff_path = next((p for p in [
    os.path.join(RAW_MP,  "FEDFUNDS.csv"),
    os.path.join(RAW_EX3, "FEDFUNDS.csv"),
] if os.path.exists(p)), None)

sr = pd.read_csv(sr_path, header=None, names=["yyyymm", "sr"])
sr["dt"] = pd.to_datetime(sr["yyyymm"].astype(str).str[:6],
                           format="%Y%m").dt.to_period("M").dt.to_timestamp()
sr = sr.drop_duplicates("dt").set_index("dt")["sr"]
ff = pd.read_csv(ff_path, parse_dates=["observation_date"],
                 index_col="observation_date")
ff.index = ff.index.to_period("M").to_timestamp()
rf     = sr.reindex(IDX).fillna(ff["FEDFUNDS"].reindex(IDX)) / 12
rf_idx = rf.reindex(IDX)

ex3 = pd.read_csv(EX3_CSV, parse_dates=["date"], index_col="date")
ex3.index = ex3.index.to_period("M").to_timestamp()
print(f"  ex3: {ex3.shape}, eps_core={ex3['eps_core'].notna().sum()} non-null")


# ============================================================
# SECTION 2 — Build 7 class-average annualized excess returns
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 2 — Class averages")
print(f"{'─'*65}")

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
CLASSES        = list(PORT_DEFS.keys())

avgs = {}
for cls, cols in PORT_DEFS.items():
    sub = ex3[cols].reindex(IDX).mean(axis=1)
    avgs[cls] = sub * 12 if cls in ALREADY_EXCESS else (sub - rf_idx) * 12

df_avgs = pd.DataFrame(avgs).join(
    ex3[["eps_core", "eps_energy", "demand_dummy"]])
df_fm = (df_avgs[(df_avgs.index >= START) & (df_avgs.index <= END)]
         .dropna(subset=["eps_core", "eps_energy", "demand_dummy"]))

print(f"  Sample: {df_fm.index.min().date()} -> {df_fm.index.max().date()}")
print(f"  N={len(df_fm)}  Demand={int((df_fm['demand_dummy']==1).sum())}  "
      f"Supply={int((df_fm['demand_dummy']==0).sum())}")


# ============================================================
# SECTION 3 — NW-OLS helper
# ============================================================

def nw_ols(y, X, bw=NW_BW):
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y2, X2 = y[mask], X[mask]; n = len(y2)
    if n < X2.shape[1] + 5: return None
    b  = lstsq(X2, y2, rcond=None)[0]; e = y2 - X2 @ b
    Xe = X2 * e[:, None]; S = Xe.T @ Xe / n
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1.0)
        G = Xe[lag:].T @ Xe[:n-lag] / n; S += w * (G + G.T)
    Vi = np.linalg.inv(X2.T @ X2 / n); V = Vi @ S @ Vi / n
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y2 - y2.mean()) @ (y2 - y2.mean()))
    return b, se, b / se, n, r2


def stars(t):
    return "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 \
           else "*" if abs(t) > 1.645 else ""


# ============================================================
# SECTION 4 — First-pass unconditional betas
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 4 — First-pass unconditional betas")
print(f"{'─'*65}")

betas = {}
for cls in CLASSES:
    y = df_fm[cls].values
    X = np.column_stack([np.ones(len(y)),
                         df_fm["eps_core"].values,
                         df_fm["eps_energy"].values])
    res = nw_ols(y, X, bw=NW_BW)
    if res:
        b, se, t, n, r2 = res
        betas[cls] = {"b_core": b[1], "t_core": t[1],
                      "b_energy": b[2], "t_energy": t[2], "R2": r2}

Bc  = np.array([betas[cls]["b_core"]   for cls in CLASSES])
Be  = np.array([betas[cls]["b_energy"] for cls in CLASSES])
Bmat = np.column_stack([np.ones(7), Bc, Be])
corr_be = np.corrcoef(Bc, Be)[0, 1]

print(f"\n  {'Class':<14} {'b_core':>8} {'(t)':>6} {'b_energy':>10} {'(t)':>6}")
print("  " + "─" * 48)
for cls in CLASSES:
    r = betas[cls]
    print(f"  {cls:<14} {r['b_core']:>+8.3f} {r['t_core']:>+6.2f} "
          f"{r['b_energy']:>+10.4f} {r['t_energy']:>+6.2f}")
print(f"\n  corr(b_core, b_energy) = {corr_be:.3f}")
print(f"  NOTE: High collinearity -> FM second pass underpowered")


# ============================================================
# SECTION 5 — FM second pass (all months)
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 5 — FM second pass")
print(f"{'─'*65}")

lam_series = []
for t_idx in df_fm.index:
    r_t = np.array([df_fm.loc[t_idx, cls] for cls in CLASSES])
    if np.any(np.isnan(r_t)): continue
    lam   = lstsq(Bmat, r_t, rcond=None)[0]
    r_hat = Bmat @ lam
    ss_res = (r_t - r_hat) @ (r_t - r_hat)
    ss_tot = (r_t - r_t.mean()) @ (r_t - r_t.mean())
    r2_xs  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    lam_series.append({
        "date": t_idx, "alpha": lam[0],
        "lam_core": lam[1], "lam_energy": lam[2],
        "demand": df_fm.loc[t_idx, "demand_dummy"],
        "r2_xs": r2_xs,
    })

df_lam = pd.DataFrame(lam_series).set_index("date")
T      = len(df_lam)
d_lam  = df_lam[df_lam["demand"] == 1]
s_lam  = df_lam[df_lam["demand"] == 0]

print(f"  T={T}  D={len(d_lam)}  S={len(s_lam)}")
print(f"  Mean cross-sectional R^2 = {df_lam['r2_xs'].mean():.3f}\n")

fm_results = {}
for col, factor in [("lam_core", "core"), ("lam_energy", "energy")]:
    ma  = df_lam[col].mean(); ta = ma / (df_lam[col].std() / np.sqrt(T))
    md  = d_lam[col].mean();  td = md / (d_lam[col].std() / np.sqrt(len(d_lam)))
    ms  = s_lam[col].mean();  ts = ms / (s_lam[col].std() / np.sqrt(len(s_lam)))
    diff   = md - ms
    se_d   = np.sqrt(d_lam[col].var()/len(d_lam) + s_lam[col].var()/len(s_lam))
    t_diff = diff / se_d
    fm_results[factor] = {
        "all": ma, "t_all": ta,
        "demand": md, "t_d": td,
        "supply": ms, "t_s": ts,
        "diff": diff, "t_diff": t_diff,
    }
    print(f"  lambda_{factor}:")
    print(f"    ALL    = {ma:+.3f}  t={ta:+.2f} {stars(ta)}")
    print(f"    DEMAND = {md:+.3f}  t={td:+.2f} {stars(td)}")
    print(f"    SUPPLY = {ms:+.3f}  t={ts:+.2f} {stars(ts)}")
    print(f"    DIFF   = {diff:+.3f}  t={t_diff:+.2f} {stars(t_diff)}\n")


# ============================================================
# SECTION 6 — Time-series sensitivity test (7/7 H3 evidence)
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 6 — Time-series sensitivity test")
print(f"{'─'*65}")
print("  H3: |corr(r_i, eps_energy)| higher in demand than supply months\n")

d_sub = df_fm[df_fm["demand_dummy"] == 1]
s_sub = df_fm[df_fm["demand_dummy"] == 0]

sens_results = {}
n_correct = 0
for cls in CLASSES:
    cd = d_sub[cls].corr(d_sub["eps_energy"])
    cs = s_sub[cls].corr(s_sub["eps_energy"])
    ok = abs(cd) > abs(cs)
    if ok: n_correct += 1
    sens_results[cls] = {
        "corr_D": cd, "corr_S": cs,
        "b_energy": betas[cls]["b_energy"],
        "correct": ok,
    }
    print(f"  {cls:<14} b_e={betas[cls]['b_energy']:+.3f}  "
          f"corr_D={cd:+.3f}  corr_S={cs:+.3f}  "
          f"{'OK' if ok else '--'}")

print(f"\n  RESULT: {n_correct}/7 classes show |corr_D| > |corr_S|")


# ============================================================
# SECTION 7 — Save Table 9 CSV + lambda series
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 7 — Saving CSVs")
print(f"{'─'*65}")

rows = []
for factor, res in fm_results.items():
    rows.append({
        "factor":        f"lambda_{factor}",
        "all":           round(res["all"],    3),
        "t_all":         round(res["t_all"],  2),
        "demand":        round(res["demand"], 3),
        "t_demand":      round(res["t_d"],    2),
        "supply":        round(res["supply"], 3),
        "t_supply":      round(res["t_s"],    2),
        "diff_D_minus_S":round(res["diff"],   3),
        "t_diff":        round(res["t_diff"], 2),
        "N_all":         T,
        "N_demand":      len(d_lam),
        "N_supply":      len(s_lam),
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB_OUT, "Table9_FM.csv"), index=False)
df_lam.to_csv(os.path.join(TAB_OUT, "lambda_series.csv"))
print(f"  SAVED: Table9_FM.csv")
print(f"  SAVED: lambda_series.csv")


# ============================================================
# SECTION 8 — TABLE 9 figure
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 8 — TABLE 9 figure")
print(f"{'─'*65}")

fig, ax = plt.subplots(figsize=(22, 8))
ax.axis("off"); fig.patch.set_facecolor("white"); ax.set_facecolor("white")

def hline(y, lw=0.8, color="black"):
    ax.plot([0.005, 0.995], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=False)

CX = [0.01, 0.22, 0.35, 0.48, 0.61, 0.74, 0.87]
hdrs = ["Factor", "ALL months", "DEMAND months",
        "SUPPLY months", "DIFF (D-S)", "Collin. rho", "XS R2"]

ax.text(0.50, 0.985,
    "TABLE 9 — Fama-MacBeth Price-of-Risk Estimates: Unconditional and Conditional",
    ha="center", fontsize=13, fontweight="bold", style="italic",
    va="top", transform=ax.transAxes)
ax.text(0.50, 0.935,
    "Second-pass: r_i,t = alpha_t + lambda_core * beta_core_i"
    " + lambda_energy * beta_energy_i + eps_i,t."
    " FM SE = std(lambda_t)/sqrt(T). 7 class averages. Mar 2003 - Jun 2023.",
    ha="center", fontsize=9.5, color="#444", va="top", transform=ax.transAxes)

hline(0.890, lw=1.4)

ax.text(CX[1]+0.065, 0.855, f"(T={T})",
        ha="center", fontsize=9, color="#666", va="center", transform=ax.transAxes)
ax.text(CX[2]+0.065, 0.855, f"(T={len(d_lam)})",
        ha="center", fontsize=9, color="#2166ac", va="center", transform=ax.transAxes)
ax.text(CX[3]+0.065, 0.855, f"(T={len(s_lam)})",
        ha="center", fontsize=9, color="#d7191c", va="center", transform=ax.transAxes)

y_hdr = 0.825
for x, h in zip(CX, hdrs):
    ax.text(x, y_hdr, h, fontsize=10.5, fontweight="bold", va="center",
            ha="left" if x == CX[0] else "center", transform=ax.transAxes)
hline(0.790, lw=0.5, color="#aaa")

y_exp = 0.755
exp_vals = ["Expected (FLR)", "< 0 / > 0", "> 0 energy\n< 0 core",
            "~0 energy\n< 0 core", "D > S energy", "--", "--"]
for x, e in zip(CX, exp_vals):
    for j, line in enumerate(e.split("\n")):
        ax.text(x, y_exp - j*0.035, line, fontsize=8.5, color="#777",
                style="italic", va="center",
                ha="left" if x == CX[0] else "center",
                transform=ax.transAxes)
hline(0.700, lw=0.5, color="#aaa")

flabels = {
    "core":   "lambda_core  (%/yr)",
    "energy": "lambda_energy  (%/yr)",
}
row_h = 0.12; y0 = 0.660
for i, (factor, res) in enumerate(fm_results.items()):
    y = y0 - i * row_h
    if i % 2 == 0:
        rect = mpl_patches.FancyBboxPatch(
            (0.006, y - row_h*0.47), 0.988, row_h*0.94,
            boxstyle="square,pad=0", facecolor="#f5f5f5",
            edgecolor="none", zorder=0, transform=ax.transAxes)
        ax.add_patch(rect)
    ax.text(CX[0], y, flabels[factor], fontsize=11, fontweight="bold",
            va="center", transform=ax.transAxes, zorder=1)

    def fc(val, t, xp, yp, color="#111"):
        s = stars(t)
        ax.text(xp, yp+0.025, f"{val:+.2f}{s}", fontsize=11, color=color,
                ha="center", va="center", fontweight="bold",
                transform=ax.transAxes, zorder=1)
        ax.text(xp, yp-0.025, f"({t:+.2f})", fontsize=9, color="#666",
                ha="center", va="center", transform=ax.transAxes, zorder=1)

    fc(res["all"],    res["t_all"],  CX[1]+0.065, y)
    fc(res["demand"], res["t_d"],    CX[2]+0.065, y, "#2166ac")
    fc(res["supply"], res["t_s"],    CX[3]+0.065, y, "#d7191c")
    dc = "#2166ac" if (factor=="energy" and res["diff"]>0) else \
         "#d7191c" if (factor=="energy" and res["diff"]<0) else "#111"
    fc(res["diff"],   res["t_diff"], CX[4]+0.065, y, dc)
    ax.text(CX[5]+0.065, y, f"rho={corr_be:.2f}", fontsize=10,
            ha="center", va="center", color="#d7191c",
            transform=ax.transAxes, zorder=1)
    ax.text(CX[6]+0.065, y, f"{df_lam['r2_xs'].mean():.3f}", fontsize=10,
            ha="center", va="center", color="#444",
            transform=ax.transAxes, zorder=1)

y_bot = y0 - len(fm_results)*row_h + row_h*0.5
hline(y_bot, lw=1.4)
ax.text(0.50, y_bot - 0.045,
    f"Time-series sensitivity: {n_correct}/7 classes show stronger |corr(r,eps_energy)|"
    f" in demand months — consistent with H3. "
    f"FM underpowered: corr(beta_core, beta_energy)={corr_be:.2f}.",
    ha="center", fontsize=9.5, color="#111", fontweight="bold",
    va="top", transform=ax.transAxes, zorder=1)
ax.text(0.50, y_bot - 0.105,
    "Notes: FM SE = std(lambda_t)/sqrt(T). Demand = energy_Rt > rolling 24m p70. "
    "H3 predicts lambda_energy(demand)>0, lambda_energy(supply)~0. "
    "Primary H3 evidence: first-pass beta_D>beta_S (7/7, Script 12). "
    "*** p<0.01  ** p<0.05  * p<0.10.",
    ha="center", fontsize=8.5, color="#555", va="top",
    transform=ax.transAxes, linespacing=1.5, zorder=1)

t9_png = os.path.join(FIG_OUT, "Table9_FM.png")
plt.savefig(t9_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  OK: Table9 -> {t9_png}")


# ============================================================
# SECTION 9 — FIGURE 24: lambda_t time series
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 9 — FIGURE 24: lambda_t time series")
print(f"{'─'*65}")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.subplots_adjust(left=0.09, right=0.74, top=0.90, bottom=0.12, hspace=0.70)

for ax_i, (col, panel, fn) in enumerate([
    ("lam_core",   "(a)", "core"),
    ("lam_energy", "(b)", "energy"),
]):
    ax = axes[ax_i]
    ax.bar(d_lam.index, d_lam[col], color="#2166ac", alpha=0.55,
           width=28, zorder=3, label="Demand month")
    ax.bar(s_lam.index, s_lam[col], color="#d7191c", alpha=0.55,
           width=28, zorder=3, label="Supply month")
    mean_val = df_lam[col].mean()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(mean_val, color="#333", linewidth=1.4, linestyle="-",
               label=f"Mean = {mean_val:+.1f}")
    ax.set_ylabel(fr"$\lambda_t^{{{fn}}}$  (%/yr)", fontsize=10)

    # Extend top 45% so stats text never overlaps bars
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.45 * (ymax - ymin))

    ax.set_title(
        f"{panel}   Price of {fn} inflation risk — "
        fr"$\hat{{\lambda}}^{{{fn}}}$",
        fontsize=11, fontweight="bold", style="italic",
        loc="left", pad=16)

    res = fm_results[fn]
    stats_txt = (
        f"All: {res['all']:+.1f} (t={res['t_all']:+.2f}){stars(res['t_all'])}   |   "
        f"Demand: {res['demand']:+.1f} (t={res['t_d']:+.2f}){stars(res['t_d'])}   |   "
        f"Supply: {res['supply']:+.1f} (t={res['t_s']:+.2f}){stars(res['t_s'])}"
    )
    # 0.92 = inside the 45% padded zone, clear of all bars
    ax.text(0.01, 0.92, stats_txt,
            transform=ax.transAxes, fontsize=8.5, color="#444", va="top")

    ax.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor="#ccc",
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].xaxis.set_major_locator(mdates.YearLocator(3))

fig.text(0.09, 0.955, "F I G U R E   2 4",
         fontsize=8, color="gray", va="bottom")
fig.text(0.09, 0.945,
    r"Fama-MacBeth $\lambda_t$ Estimates — Core and Energy Inflation Risk",
    fontsize=12, fontweight="bold", style="italic", va="bottom")

fig.text(0.09, 0.065,
    r"$\it{Notes}$: Monthly FM second-pass slope coefficients from cross-sectional OLS:"
    r" $r_{i,t}=\alpha_t+\lambda_t^{core}\hat{\beta}_i^{core}"
    r"+\lambda_t^{energy}\hat{\beta}_i^{energy}+\epsilon_{i,t}$.",
    fontsize=8.5, color="#444", va="top", linespacing=1.5)
fig.text(0.09, 0.032,
    r"Blue bars = demand months (energy_Rt > rolling 24m p70), Red = supply months."
    r" Solid line = unconditional mean. 7 class averages. Mar 2003 - Jun 2023."
    r" FM SE = std($\lambda_t$)/$\sqrt{T}$.",
    fontsize=8.5, color="#444", va="top", linespacing=1.5)

f24 = os.path.join(FIG_OUT, "Fig24_lambda_series.png")
plt.savefig(f24, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  OK: Fig24 -> {f24}")


# ============================================================
# SECTION 10 — FIGURE 25: Sensitivity bar chart
# ============================================================
print(f"\n{'─'*65}")
print("SECTION 10 — FIGURE 25: Sensitivity")
print(f"{'─'*65}")

cls_sorted = sorted(CLASSES, key=lambda c: sens_results[c]["b_energy"])
corr_d = [sens_results[cls]["corr_D"] for cls in cls_sorted]
corr_s = [sens_results[cls]["corr_S"] for cls in cls_sorted]
be_s   = [sens_results[cls]["b_energy"] for cls in cls_sorted]

x = np.arange(len(cls_sorted)); w = 0.36
fig, ax = plt.subplots(figsize=(13, 7))
fig.subplots_adjust(left=0.09, right=0.72, top=0.78, bottom=0.30)

ax.bar(x - w/2, corr_d, w, color="#2166ac", alpha=0.80, zorder=3, label="Demand months")
ax.bar(x + w/2, corr_s, w, color="#d7191c", alpha=0.80, zorder=3, label="Supply months")
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{cls}\n(b={be_s[i]:+.3f})" for i, cls in enumerate(cls_sorted)],
    fontsize=9)
ax.set_ylabel(r"corr$(r_i,\ \varepsilon^{energy})$", fontsize=10)
ax.grid(axis="y", alpha=0.18, linewidth=0.5)
ax.legend(fontsize=10, frameon=True, framealpha=0.95, edgecolor="#ccc",
          loc="upper left", bbox_to_anchor=(1.01, 1.0))

# Title block — well above bars
ax.text(0, 1.18, "F I G U R E   2 5",
        transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
ax.text(0, 1.10,
    r"H3 Sensitivity: corr$(r_i,\ \varepsilon^{energy})$ by Regime"
    r" — Classes Sorted by $\hat{\beta}_{energy}$",
    transform=ax.transAxes, fontsize=11, fontweight="bold",
    style="italic", va="bottom")

# Two-line note below x-axis
ax.text(0, -0.22,
    r"$\it{Notes}$: corr$(r_{i,t},\ \varepsilon^{energy}_t)$ computed separately"
    r" in demand months (blue, energy_Rt > rolling 24m p70) and supply months (red).",
    transform=ax.transAxes, fontsize=9, color="#444", va="top", linespacing=1.5)
ax.text(0, -0.32,
    f"Classes sorted left-to-right by b_energy (lowest to highest). "
    f"{n_correct}/7 classes: |corr_D| > |corr_S| — energy shocks price returns "
    r"more strongly in demand months, consistent with H3. Mar 2003 - Jun 2023.",
    transform=ax.transAxes, fontsize=9, color="#444", va="top", linespacing=1.5)

f25 = os.path.join(FIG_OUT, "Fig25_sensitivity.png")
plt.savefig(f25, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  OK: Fig25 -> {f25}")


# ============================================================
# FINAL VERIFICATION
# ============================================================
print(f"\n{'='*65}")
print("FINAL VERIFICATION")
print(f"{'='*65}\n")

checks = [
    ("eps_core in ex3",          "eps_core" in ex3.columns,           "present"),
    ("eps_energy in ex3",        "eps_energy" in ex3.columns,         "present"),
    ("FM T >= 200",              T >= 200,                            f"T={T}"),
    ("Mean XS R2 > 0",           df_lam["r2_xs"].mean() > 0,          f"{df_lam['r2_xs'].mean():.3f}"),
    ("7/7 sensitivity correct",  n_correct == 7,                      f"{n_correct}/7"),
    ("Table9 CSV",   os.path.exists(os.path.join(TAB_OUT,"Table9_FM.csv")),          "saved"),
    ("Lambda CSV",   os.path.exists(os.path.join(TAB_OUT,"lambda_series.csv")),      "saved"),
    ("Table9 PNG",   os.path.exists(os.path.join(FIG_OUT,"Table9_FM.png")),          "saved"),
    ("Fig24 PNG",    os.path.exists(os.path.join(FIG_OUT,"Fig24_lambda_series.png")),"saved"),
    ("Fig25 PNG",    os.path.exists(os.path.join(FIG_OUT,"Fig25_sensitivity.png")),  "saved"),
]

all_ok = True
for label, ok, detail in checks:
    status = "OK  " if ok else "FAIL"
    if not ok: all_ok = False
    print(f"  {status}  {label:<45} {detail}")

print(f"\n  {'All checks PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"\n{'='*65}")
print("13_fama_macbeth_FINAL.py — DONE")
print(f"{'='*65}")
print(f"""
  KEY RESULTS:
    FM T={T}, mean XS R2={df_lam['r2_xs'].mean():.3f}
    lambda_core  (ALL)   = {fm_results['core']['all']:+.3f}  t={fm_results['core']['t_all']:+.2f}
    lambda_energy (ALL)  = {fm_results['energy']['all']:+.3f}  t={fm_results['energy']['t_all']:+.2f}
    lambda_energy (D)    = {fm_results['energy']['demand']:+.3f}  t={fm_results['energy']['t_d']:+.2f}
    lambda_energy (S)    = {fm_results['energy']['supply']:+.3f}  t={fm_results['energy']['t_s']:+.2f}

  H3 EVIDENCE:
    First-pass betas (Script 12): 7/7 correct signs — PRIMARY
    Sensitivity corr(r,eps_e):    {n_correct}/7 stronger in demand — SUPPORTIVE
    FM second pass lambda:         underpowered (corr={corr_be:.2f}) — DOCUMENTED
""")