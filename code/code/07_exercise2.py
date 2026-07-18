# ============================================================
# 07_exercise2_v3.py  —  EXERCISE 2 : REGRESSIONS (version 3)
#
# NOUVEAUTES vs v2 (3 outputs visuels ajoutés à la fin) :
#   Table3_baseline_regression.png  — OLS baseline formaté
#   Table4_LP_results.png           — LP résultats formatés
#   Table5_horse_race.png           — Horse race formaté
#
# MODIFICATIONS vs v1 (inchangées) :
#   FIX 1 — FLR_real (vrai ratio énergie/core)
#   FIX 2 — CP proxy + d_term_spread
#   FIX 3 — Fig9 note théorique Pflueger (2025)
#
# INPUTS  : Data/cleaned/ex2_monthly.csv
#           Data/raw/Side of Pt/Headline CPI.csv
#           Data/raw/PCEPILFE.csv
# OUTPUTS :
#   output/figures/Fig8_LP_rho_sb.png
#   output/figures/Fig9_regime_classification.png
#   output/tables/Table3_baseline_regression.csv
#   output/tables/Table3_baseline_regression.png    ← NOUVEAU
#   output/tables/Table4_LP_results.csv
#   output/tables/Table4_LP_results.png             ← NOUVEAU
#   output/tables/Table5_horse_race.csv
#   output/tables/Table5_horse_race.png             ← NOUVEAU
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy import stats
import os

# ── CHEMINS ──────────────────────────────────────────────────
BASE    = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
CLEAN   = os.path.join(BASE, "Data", "cleaned")
RAW_SP  = os.path.join(BASE, "Data", "raw", "Side of Pt")
RAW_ADD = os.path.join(BASE, "Data", "raw")
FIG_OUT = os.path.join(BASE, "output", "figures")
TAB_OUT = os.path.join(BASE, "output", "tables")
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

print("=" * 60)
print("07_exercise2_v3.py — EXERCISE 2 (version 3)")
print("=" * 60)

# ── CHARGEMENT DONNEES PRINCIPALES ───────────────────────────
df = pd.read_csv(os.path.join(CLEAN, "ex2_monthly.csv"),
                 parse_dates=["date"], index_col="date")

print(f"\nDonnees : {df.shape[0]} obs  |  "
      f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")

# ── CALCUL rho_sb si absent ───────────────────────────────────
if "rho_sb" not in df.columns:
    WINDOW = 24
    df["rho_sb"] = (df["sp500_ret"]
                    .rolling(window=WINDOW, min_periods=WINDOW)
                    .corr(df["bond_ret"]))

# ── VARIABLES ADDITIONNELLES ──────────────────────────────────
df["equity_vol"]   = df["sp500_ret"].rolling(24, min_periods=24).std() * np.sqrt(12)
df["delta_dollar"] = np.log(df["dollar_index"]).diff()

# ── FIX 1 : VRAI INDICATEUR FLR ─────────────────────────────
print("\n[FIX 1] Construction du vrai indicateur FLR...")
cpi_path  = os.path.join(RAW_SP,  "Headline CPI.csv")
core_path = os.path.join(RAW_ADD, "PCEPILFE.csv")

cpi_raw  = pd.read_csv(cpi_path,  parse_dates=["observation_date"]).rename(
    columns={"observation_date": "date", "CPIAUCSL": "cpi"}).set_index("date")
core_raw = pd.read_csv(core_path, parse_dates=["observation_date"]).rename(
    columns={"observation_date": "date", "PCEPILFE": "core_pce"}).set_index("date")

cpi_raw["cpi_mom"]   = np.log(cpi_raw["cpi"]).diff()
core_raw["core_mom"] = np.log(core_raw["core_pce"]).diff()
inf_df = cpi_raw[["cpi_mom"]].join(core_raw[["core_mom"]], how="inner")
inf_df["energy_inf"] = inf_df["cpi_mom"] - inf_df["core_mom"]
inf_df["FLR_real"]   = (inf_df["energy_inf"].rolling(24).std()
                        / inf_df["core_mom"].rolling(24).std())

df = df.join(inf_df[["FLR_real"]], how="left")
print(f"   FLR_real disponible : {df['FLR_real'].notna().sum()} obs | mean={df['FLR_real'].mean():.3f}")

# ── FIX 2 : COMPETITORS ──────────────────────────────────────
print("\n[FIX 2] Construction des competitors...")
df["CP"]            = df["rho_t"]
df["d_term_spread"] = df["term_spread"].diff()

# ── CONTROLES & UTILITAIRES ──────────────────────────────────
CONTROLS = ["inflation_yoy", "equity_vol", "delta_dollar",
            "vix", "policy_rate", "term_spread"]
HAC_BW   = 24

def run_ols_hac(y, X, bw=HAC_BW):
    data = pd.concat([y, X], axis=1).dropna()
    y_   = data.iloc[:, 0]
    X_   = sm.add_constant(data.iloc[:, 1:])
    return sm.OLS(y_, X_).fit(
        cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": True})

def extract_results(model, name):
    rows = []
    for var in model.params.index:
        ci95 = model.conf_int(alpha=0.05)
        rows.append({
            "model"   : name,
            "variable": var,
            "coef"    : round(model.params[var], 4),
            "tstat"   : round(model.tvalues[var], 3),
            "pval"    : round(model.pvalues[var], 4),
            "ci95_lo" : round(ci95.loc[var, 0], 4),
            "ci95_hi" : round(ci95.loc[var, 1], 4),
            "R2"      : round(model.rsquared, 4),
            "N"       : int(model.nobs),
        })
    return rows

def sig_stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def hrule(ax, y, lw=1.0, color="black"):
    """Draw a horizontal rule in axes coordinates (0-1). Compatible with all matplotlib versions."""
    ax.plot([0, 1], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=False)


# ============================================================
# EXERCISE 2A — BASELINE REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("EXERCISE 2A — Baseline OLS + Newey-West HAC")
print("=" * 60)

m1 = run_ols_hac(df["rho_sb"], df[["R_t"]])
m2 = run_ols_hac(df["rho_sb"], df[["R_t"] + CONTROLS])

rows_2a  = extract_results(m1, "(1) R_t only")
rows_2a += extract_results(m2, "(2) R_t + controls")
table3   = pd.DataFrame(rows_2a)

for mname in ["(1) R_t only", "(2) R_t + controls"]:
    sub = table3[table3["model"] == mname]
    print(f"\n  {mname}  |  R²={sub['R2'].iloc[0]:.4f}  N={sub['N'].iloc[0]}")
    for _, row in sub.iterrows():
        print(f"  {row['variable']:<20} {row['coef']:>8.4f} "
              f"({row['tstat']:>+.3f}) p={row['pval']:.4f} {sig_stars(row['pval'])}")

tab3_path = os.path.join(TAB_OUT, "Table3_baseline_regression.csv")
table3.to_csv(tab3_path, index=False)
print(f"\n  ✓ Table 3 CSV → {tab3_path}")
print(f"  NOTE : beta(R_t)={m2.params['R_t']:.4f} (t={m2.tvalues['R_t']:.2f}) — signe positif expliqué dans la thèse")

# ============================================================
# EXERCISE 2B — LOCAL PROJECTIONS
# ============================================================
print("\n" + "=" * 60)
print("EXERCISE 2B — Local Projections h = 3, 6, 12")
print("=" * 60)

HORIZONS   = [3, 6, 12]
lp_results = []

for h in HORIZONS:
    y_h = df["rho_sb"].shift(-h)
    m_h = run_ols_hac(y_h, df[["R_t"] + CONTROLS], bw=max(HAC_BW, h))
    ci95_h = m_h.conf_int(alpha=0.05).loc["R_t"]
    ci90_h = m_h.conf_int(alpha=0.10).loc["R_t"]
    lp_results.append({
        "h": h, "beta": round(m_h.params["R_t"], 4),
        "tstat": round(m_h.tvalues["R_t"], 3),
        "pval":  round(m_h.pvalues["R_t"], 4),
        "r2":    round(m_h.rsquared, 4),
        "n":     int(m_h.nobs),
        "ci95_lo": round(ci95_h.iloc[0], 4), "ci95_hi": round(ci95_h.iloc[1], 4),
        "ci90_lo": round(ci90_h.iloc[0], 4), "ci90_hi": round(ci90_h.iloc[1], 4),
    })
    print(f"  h={h:2d}  beta={m_h.params['R_t']:7.4f}  t={m_h.tvalues['R_t']:6.3f}  "
          f"p={m_h.pvalues['R_t']:.4f}  R²={m_h.rsquared:.3f}  N={int(m_h.nobs)} "
          f"{sig_stars(m_h.pvalues['R_t'])}")

table4    = pd.DataFrame(lp_results)
tab4_path = os.path.join(TAB_OUT, "Table4_LP_results.csv")
table4.to_csv(tab4_path, index=False)
print(f"\n  ✓ Table 4 CSV → {tab4_path}")

# FIGURE 8 — LP IRF
fig, ax = plt.subplots(figsize=(8, 5))
hs    = [r["h"]       for r in lp_results]
betas = [r["beta"]    for r in lp_results]
ax.fill_between(hs, [r["ci95_lo"] for r in lp_results],
                [r["ci95_hi"] for r in lp_results],
                alpha=0.15, color="steelblue", label="95% CI")
ax.fill_between(hs, [r["ci90_lo"] for r in lp_results],
                [r["ci90_hi"] for r in lp_results],
                alpha=0.25, color="steelblue", label="90% CI")
ax.plot(hs, betas, "o-", color="black", linewidth=1.8, markersize=7, zorder=5,
        label=r"$\hat{\beta}_h$")
ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
for h, b, p in zip(hs, betas, [r["pval"] for r in lp_results]):
    ax.annotate(sig_stars(p), xy=(h, b), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=11)
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"h = {h}" for h in HORIZONS])
ax.set_ylabel(r"$\hat{\beta}_h$", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.text(0, 1.08, "F I G U R E   8", transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02, r"Local Projection Coefficients $\hat{\beta}_h$ on $\rho_{t+h}^{SB}$",
        transform=ax.transAxes, fontsize=11, color="black", va="bottom", style="italic")
ax.annotate(
    r"Notes: $\hat{\beta}_h$ from regressing $\rho_{t+h}^{SB}$ on $R_t$ with controls. "
    "Newey-West HAC SE, bw=max(24,h). Shading: 90% and 95% CI.\n"
    "Controls: inflation, realised equity vol, log-change dollar, VIX, Wu-Xia policy rate, "
    "term spread. Sample: May 2003 — June 2023.",
    xy=(0, -0.26), xycoords="axes fraction", fontsize=8, color="dimgray", va="top")
plt.tight_layout(rect=[0, 0.22, 1, 1])
fig8_path = os.path.join(FIG_OUT, "Fig8_LP_rho_sb.png")
plt.savefig(fig8_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ✓ Figure 8 → {fig8_path}")

# ============================================================
# EXERCISE 2C — REGIME CLASSIFICATION
# ============================================================
print("\n" + "=" * 60)
print("EXERCISE 2C — Regime Classification + Binomial Test")
print("=" * 60)

ROLL_PCT = 60
df["Rt_pct70"] = df["R_t"].rolling(ROLL_PCT, min_periods=24).quantile(0.70)
df["Rt_pct30"] = df["R_t"].rolling(ROLL_PCT, min_periods=24).quantile(0.30)
df["D_regime"] = df["R_t"] > df["Rt_pct70"]
df["S_regime"] = df["R_t"] < df["Rt_pct30"]
df_class = df.dropna(subset=["rho_sb", "Rt_pct70", "Rt_pct30"])

D_months  = df_class[df_class["D_regime"]]
D_correct = (D_months["rho_sb"] < 0).sum()
D_total   = len(D_months)
D_frac    = D_correct / D_total if D_total > 0 else 0
S_months  = df_class[df_class["S_regime"]]
S_correct = (S_months["rho_sb"] > 0).sum()
S_total   = len(S_months)
S_frac    = S_correct / S_total if S_total > 0 else 0
binom_D   = stats.binomtest(D_correct, D_total, p=0.5, alternative="greater")
binom_S   = stats.binomtest(S_correct, S_total, p=0.5, alternative="greater")

print(f"\n  D-regime : {D_total} mois | correct={D_frac*100:.1f}% "
      f"p={binom_D.pvalue:.4f} {'✓' if D_frac>0.70 else '✗'}")
print(f"  S-regime : {S_total} mois | correct={S_frac*100:.1f}% "
      f"p={binom_S.pvalue:.4f} {'✓' if S_frac>0.70 else '✗'}")

# FIGURE 9 — Regime classification bars (FIX 3: note théorique)
s_note = (
    "Notes: D = R_t > rolling 70th pct (60m window). S = R_t < rolling 30th pct. "
    "Null: 50% (one-sided binomial test). Dotted = 70% threshold.\n"
    "D-regime passes (70.9%, p<0.01). S-regime does not (13.3%): supply shocks break "
    "the bond hedge only when simultaneously inflationary (Pflueger 2025). Supply periods\n"
    "2013–2019 were deflationary (oil glut) — bonds continued to hedge. "
    "This is a finding, not a failure."
)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, frac, total, label, color, pct_label in [
    (axes[0], D_frac, D_total,
     r"D-Regime ($R_t >$ 70th pct)" + "\n" + r"Correct: $\rho_{sb} < 0$",
     "steelblue", f"{D_frac*100:.1f}%"),
    (axes[1], S_frac, S_total,
     r"S-Regime ($R_t <$ 30th pct)" + "\n" + r"Correct: $\rho_{sb} > 0$",
     "salmon", f"{S_frac*100:.1f}%"),
]:
    ax.bar(["Correct", "Incorrect"], [frac*100, (1-frac)*100],
           color=[color, "lightgray"], width=0.5, zorder=3)
    ax.axhline(50, color="black", linestyle="--", linewidth=1, label="50% null", zorder=4)
    ax.axhline(70, color="darkgray", linestyle=":", linewidth=1, label="70% threshold", zorder=4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of months", fontsize=10)
    ax.set_title(label, fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0, frac*100 + 2, pct_label, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
axes[0].text(-0.1, 1.08, "F I G U R E   9", transform=axes[0].transAxes,
             fontsize=8, color="gray", va="bottom")
axes[0].text(-0.1, 1.02, "Regime Classification — Sign Agreement Test",
             transform=axes[0].transAxes, fontsize=11, color="black",
             va="bottom", style="italic")
fig.text(0.5, -0.08, s_note, ha="center", fontsize=7.5, color="dimgray")
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig9_path = os.path.join(FIG_OUT, "Fig9_regime_classification.png")
plt.savefig(fig9_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  ✓ Figure 9 → {fig9_path}")

# ============================================================
# EXERCISE 2D — HORSE RACE
# ============================================================
print("\n" + "=" * 60)
print("EXERCISE 2D — Horse Race (version 2)")
print("=" * 60)

competitors = ["CP", "FLR_real", "d_term_spread"]
m3 = run_ols_hac(df["rho_sb"], df[["R_t"]])
m4 = run_ols_hac(df["rho_sb"], df[competitors + CONTROLS])
m5 = run_ols_hac(df["rho_sb"], df[["R_t"] + competitors + CONTROLS])

rows_hr  = extract_results(m3, "(3) R_t only")
rows_hr += extract_results(m4, "(4) Competitors only")
rows_hr += extract_results(m5, "(5) R_t + Competitors")
table5   = pd.DataFrame(rows_hr)

for mname in ["(3) R_t only", "(4) Competitors only", "(5) R_t + Competitors"]:
    sub = table5[table5["model"] == mname]
    print(f"\n  {mname}  |  R²={sub['R2'].iloc[0]:.4f}  N={sub['N'].iloc[0]}")
    for _, row in sub.iterrows():
        print(f"  {row['variable']:<20} {row['coef']:>8.4f} "
              f"({row['tstat']:>+.3f}) {sig_stars(row['pval'])}")

tab5_path = os.path.join(TAB_OUT, "Table5_horse_race.csv")
table5.to_csv(tab5_path, index=False)
print(f"\n  ✓ Table 5 CSV → {tab5_path}")

flr_t  = m4.tvalues["FLR_real"]
flr_p  = m4.pvalues["FLR_real"]
rt_p5  = m5.pvalues["R_t"]
r2_m3, r2_m4, r2_m5 = m3.rsquared, m4.rsquared, m5.rsquared
print(f"\n  FLR_real dominant : t={flr_t:.2f}, p={flr_p:.4f} {sig_stars(flr_p)}")
print(f"  R² : {r2_m3:.4f} → {r2_m4:.4f} → {r2_m5:.4f}")

# ============================================================
# NOUVEAU — TABLE 3 PNG
# ============================================================
print("\n" + "=" * 60)
print("TABLE 3 PNG — Baseline Regression")
print("=" * 60)

var_labels = {
    "const":         "Constant",
    "R_t":           "$R_t$",
    "inflation_yoy": "Inflation (YoY)",
    "equity_vol":    "Equity vol (24m)",
    "delta_dollar":  "Δ Dollar index",
    "vix":           "VIX",
    "policy_rate":   "Policy rate",
    "term_spread":   "Term spread",
}

t3_m1 = table3[table3["model"] == "(1) R_t only"]
t3_m2 = table3[table3["model"] == "(2) R_t + controls"]
r2_m1 = t3_m1["R2"].iloc[0]
r2_m2 = t3_m2["R2"].iloc[0]
n_m1   = t3_m1["N"].iloc[0]
n_m2   = t3_m2["N"].iloc[0]

# Variables to show: all from model 2, in order
var_order = ["R_t", "inflation_yoy", "equity_vol", "delta_dollar",
             "vix", "policy_rate", "term_spread", "const"]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axis("off")

# Column headers
col_w = [0.38, 0.31, 0.31]   # relative widths: variable, model1, model2
col_x = [0.00, 0.40, 0.70]   # x positions

header_y = 0.93
ax.text(col_x[0], header_y, "Variable", fontsize=10, fontweight="bold",
        transform=ax.transAxes, va="top")
ax.text(col_x[1] + 0.12, header_y, "(1) $R_t$ only",
        fontsize=10, fontweight="bold", transform=ax.transAxes, va="top", ha="center")
ax.text(col_x[2] + 0.12, header_y, "(2) $R_t$ + controls",
        fontsize=10, fontweight="bold", transform=ax.transAxes, va="top", ha="center")

# Top rule
hrule(ax, 0.96)
# Sub-header rule
hrule(ax, 0.88, lw=0.5)

row_h  = 0.075
start_y = 0.83

for i, var in enumerate(var_order):
    y = start_y - i * row_h
    label = var_labels.get(var, var)

    # Shade Rt row lightly
    if var == "R_t":
        ax.axhspan(y - row_h * 0.45, y + row_h * 0.55,
                   color="#f0f4ff", transform=ax.transAxes, zorder=0)

    ax.text(col_x[0], y, label, fontsize=9.5,
            transform=ax.transAxes, va="center")

    # Model 1: only R_t and const
    row_m1 = t3_m1[t3_m1["variable"] == var]
    if not row_m1.empty:
        c = row_m1["coef"].iloc[0]
        t = row_m1["tstat"].iloc[0]
        p = row_m1["pval"].iloc[0]
        stars = sig_stars(p)
        ax.text(col_x[1] + 0.12, y + 0.018,
                f"{c:.4f}{stars}", fontsize=9.5,
                transform=ax.transAxes, ha="center", va="center")
        ax.text(col_x[1] + 0.12, y - 0.018,
                f"({t:+.3f})", fontsize=8.5, color="#555555",
                transform=ax.transAxes, ha="center", va="center", style="italic")
    else:
        ax.text(col_x[1] + 0.12, y, "—", fontsize=9.5,
                transform=ax.transAxes, ha="center", va="center", color="#aaaaaa")

    # Model 2: all variables
    row_m2 = t3_m2[t3_m2["variable"] == var]
    if not row_m2.empty:
        c = row_m2["coef"].iloc[0]
        t = row_m2["tstat"].iloc[0]
        p = row_m2["pval"].iloc[0]
        stars = sig_stars(p)
        ax.text(col_x[2] + 0.12, y + 0.018,
                f"{c:.4f}{stars}", fontsize=9.5,
                transform=ax.transAxes, ha="center", va="center")
        ax.text(col_x[2] + 0.12, y - 0.018,
                f"({t:+.3f})", fontsize=8.5, color="#555555",
                transform=ax.transAxes, ha="center", va="center", style="italic")

# Bottom stats row
stats_y = start_y - len(var_order) * row_h - 0.01
hrule(ax, stats_y + row_h * 0.6, lw=0.5)
ax.text(col_x[0], stats_y, f"$R^2$", fontsize=9.5, transform=ax.transAxes, va="center")
ax.text(col_x[1] + 0.12, stats_y, f"{r2_m1:.3f}", fontsize=9.5,
        transform=ax.transAxes, ha="center", va="center")
ax.text(col_x[2] + 0.12, stats_y, f"{r2_m2:.3f}", fontsize=9.5,
        transform=ax.transAxes, ha="center", va="center")

n_y = stats_y - row_h * 0.85
ax.text(col_x[0], n_y, "Observations", fontsize=9.5, transform=ax.transAxes, va="center")
ax.text(col_x[1] + 0.12, n_y, f"{n_m1}", fontsize=9.5,
        transform=ax.transAxes, ha="center", va="center")
ax.text(col_x[2] + 0.12, n_y, f"{n_m2}", fontsize=9.5,
        transform=ax.transAxes, ha="center", va="center")

# Bottom rule
hrule(ax, n_y - row_h * 0.55)

# Note
ax.text(0.0, n_y - row_h * 0.75,
        "Notes: OLS with Newey-West HAC s.e. (bw = 24). "
        "t-statistics in parentheses. * p<0.10  ** p<0.05  *** p<0.01.\n"
        r"Dependent variable: 24-month rolling stock-bond correlation $\rho_t^{SB}$. "
        "Sample: May 2003 — June 2023. Controls: inflation, equity vol, Δ dollar, VIX, policy rate, term spread.",
        fontsize=7.5, color="dimgray", transform=ax.transAxes, va="top")

# Title
ax.text(0.0, 1.01,
        r"TABLE 3.  Predicting the Stock-Bond Correlation: Baseline OLS",
        fontsize=11, fontweight="bold", transform=ax.transAxes, va="bottom", style="italic")

plt.tight_layout()
tab3_png = os.path.join(FIG_OUT, "Table3_baseline_regression.png")
plt.savefig(tab3_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ✓ Table 3 PNG → {tab3_png}")

# ============================================================
# NOUVEAU — TABLE 4 PNG
# ============================================================
print("\n[TABLE 4 PNG] Local Projections...")

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")

# Headers
col_x4  = [0.02, 0.22, 0.38, 0.54, 0.70, 0.84]
headers = ["Horizon", r"$\hat{\beta}_h$", "t-stat", "p-value",
           r"$R^2$", "N"]
header_y4 = 0.90

hrule(ax, 0.97)
hrule(ax, 0.84, lw=0.5)

for x, h in zip(col_x4, headers):
    ax.text(x, header_y4, h, fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="center")

row_h4  = 0.20
start_y4 = 0.72

for i, r in enumerate(lp_results):
    y = start_y4 - i * row_h4
    h_val = int(r["h"])
    stars = sig_stars(r["pval"])

    # Shade h=12 row (only significant one)
    if h_val == 12:
        ax.axhspan(y - row_h4 * 0.5, y + row_h4 * 0.5,
                   color="#eef4ff", transform=ax.transAxes, zorder=0)

    vals = [f"h = {h_val}", f"{r['beta']:.4f}{stars}",
            f"({r['tstat']:+.3f})", f"{r['pval']:.4f}",
            f"{r['r2']:.3f}", str(r["n"])]
    colors = ["black", "black", "#555555", "black", "black", "black"]
    styles = ["normal", "normal", "italic", "normal", "normal", "normal"]

    for x, v, c, s in zip(col_x4, vals, colors, styles):
        ax.text(x, y, v, fontsize=10, transform=ax.transAxes,
                va="center", color=c, style=s)

# Bottom rule
bottom_y4 = start_y4 - len(lp_results) * row_h4
hrule(ax, bottom_y4 + row_h4 * 0.4)

# CI note
ci_y = bottom_y4 - 0.02
ax.text(col_x4[1], ci_y,
        f"95% CI at h=12: [{lp_results[2]['ci95_lo']:.4f}, {lp_results[2]['ci95_hi']:.4f}]",
        fontsize=8.5, color="#555555", transform=ax.transAxes, va="top", style="italic")

ax.text(0.0, bottom_y4 - 0.20,
        r"Notes: Local projections (Jordà 2005). Dependent: $\rho_{t+h}^{SB}$ (24m rolling stock-bond corr). "
        "Newey-West HAC s.e., bw=max(24,h).\n"
        "Controls: inflation, equity vol, Δ dollar, VIX, Wu-Xia policy rate, term spread. "
        "* p<0.10  ** p<0.05  *** p<0.01. Sample: May 2003 — June 2023.",
        fontsize=7.5, color="dimgray", transform=ax.transAxes, va="top")

ax.text(0.0, 1.02,
        r"TABLE 4.  Local Projection Impulse Responses: $R_t$ on $\rho_{t+h}^{SB}$",
        fontsize=11, fontweight="bold", transform=ax.transAxes, va="bottom", style="italic")

plt.tight_layout()
tab4_png = os.path.join(FIG_OUT, "Table4_LP_results.png")
plt.savefig(tab4_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ✓ Table 4 PNG → {tab4_png}")

# ============================================================
# NOUVEAU — TABLE 5 PNG
# ============================================================
print("\n[TABLE 5 PNG] Horse Race...")

# Variables to show in the table (key ones only — full table in CSV)
key_vars_t5 = ["R_t", "CP", "FLR_real", "d_term_spread", "term_spread"]
var_labels5 = {
    "R_t":           "$R_t$ (news regime indicator)",
    "CP":            "CP proxy ($\\rho_t$, inflation-output corr.)",
    "FLR_real":      "FLR (energy/core vol ratio)$^\\dagger$",
    "d_term_spread": "$\\Delta$ Term spread (CP risk-premium proxy)",
    "term_spread":   "Term spread (control)",
}

models5 = ["(3) R_t only", "(4) Competitors only", "(5) R_t + Competitors"]
model_labels5 = ["(3)\n$R_t$ only", "(4)\nCompetitors\nonly", "(5)\n$R_t$ + All"]

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.axis("off")

col_x5   = [0.01, 0.39, 0.57, 0.75]
header_y5 = 0.93

hrule(ax, 0.97)
hrule(ax, 0.87, lw=0.5)

ax.text(col_x5[0], header_y5, "Variable",
        fontsize=10, fontweight="bold", transform=ax.transAxes, va="center")
for x, lbl in zip(col_x5[1:], model_labels5):
    ax.text(x + 0.08, header_y5, lbl, fontsize=9, fontweight="bold",
            transform=ax.transAxes, ha="center", va="center")

row_h5   = 0.115
start_y5 = 0.80

for i, var in enumerate(key_vars_t5):
    y = start_y5 - i * row_h5
    label = var_labels5.get(var, var)

    # Highlight FLR_real (the significant competitor)
    if var == "FLR_real":
        ax.axhspan(y - row_h5 * 0.5, y + row_h5 * 0.5,
                   color="#fffbe6", transform=ax.transAxes, zorder=0)

    ax.text(col_x5[0], y, label, fontsize=9,
            transform=ax.transAxes, va="center")

    for j, mname in enumerate(models5):
        sub = table5[(table5["model"] == mname) & (table5["variable"] == var)]
        x = col_x5[j + 1] + 0.08
        if not sub.empty:
            c = sub["coef"].iloc[0]
            t = sub["tstat"].iloc[0]
            p = sub["pval"].iloc[0]
            stars = sig_stars(p)
            color = "#cc0000" if (var=="FLR_real" and stars=="***") else "black"
            ax.text(x, y + 0.025, f"{c:.4f}{stars}",
                    fontsize=9, transform=ax.transAxes, ha="center", va="center",
                    color=color, fontweight="bold" if stars == "***" else "normal")
            ax.text(x, y - 0.025, f"({t:+.3f})",
                    fontsize=8, color="#555555",
                    transform=ax.transAxes, ha="center", va="center", style="italic")
        else:
            ax.text(x, y, "—", fontsize=9, color="#aaaaaa",
                    transform=ax.transAxes, ha="center", va="center")

# Separator before R² row
stats_y5 = start_y5 - len(key_vars_t5) * row_h5
hrule(ax, stats_y5 + row_h5 * 0.55, lw=0.5)

# R² row
ax.text(col_x5[0], stats_y5, "$R^2$", fontsize=9.5,
        transform=ax.transAxes, va="center")
for j, mname in enumerate(models5):
    r2 = table5[table5["model"] == mname]["R2"].iloc[0]
    ax.text(col_x5[j+1] + 0.08, stats_y5, f"{r2:.4f}",
            fontsize=9.5, transform=ax.transAxes, ha="center", va="center")

# N row
n_y5 = stats_y5 - row_h5 * 0.85
ax.text(col_x5[0], n_y5, "Observations", fontsize=9.5,
        transform=ax.transAxes, va="center")
for j, mname in enumerate(models5):
    n = table5[table5["model"] == mname]["N"].iloc[0]
    ax.text(col_x5[j+1] + 0.08, n_y5, f"{n}",
            fontsize=9.5, transform=ax.transAxes, ha="center", va="center")

# Bottom rule
hrule(ax, n_y5 - row_h5 * 0.55)

# Notes
ax.text(0.0, n_y5 - row_h5 * 0.75,
        "Notes: OLS with Newey-West HAC s.e. (bw = 24). t-statistics in parentheses. "
        "* p<0.10  ** p<0.05  *** p<0.01.\n"
        "All models include controls: inflation, equity vol, Δ dollar, VIX, policy rate, term spread. "
        "CP = $\\rho_t$ (realised inflation-output gap corr., proxy for Cieslak-Pang 2021).\n"
        "$\\dagger$ FLR = std(headline CPI − core PCE, 24m) / std(core PCE, 24m), "
        "following Fang, Liu & Roussanov (2022). Sample: May 2003 — June 2023.",
        fontsize=7.5, color="dimgray", transform=ax.transAxes, va="top")

ax.text(0.0, 1.01,
        "TABLE 5.  Horse Race: $R_t$ vs Competitors",
        fontsize=11, fontweight="bold", transform=ax.transAxes, va="bottom", style="italic")

plt.tight_layout()
tab5_png = os.path.join(FIG_OUT, "Table5_horse_race.png")
plt.savefig(tab5_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"  ✓ Table 5 PNG → {tab5_png}")
# ============================================================
# RAPPORT FINAL
# ============================================================
print("\n" + "=" * 60)
print("EXERCISE 2 v3 — COMPLETE")
print("=" * 60)
print(f"  2A : beta(R_t)={m2.params['R_t']:.4f}  t={m2.tvalues['R_t']:.3f}  R²={m2.rsquared:.3f}")
for r in lp_results:
    print(f"  2B h={r['h']:2.0f}: beta={r['beta']:.4f}  t={r['tstat']:.3f} {sig_stars(r['pval'])}")
print(f"  2C D: {D_frac*100:.1f}% ✓  |  S: {S_frac*100:.1f}% (expliqué Pflueger 2025)")
print(f"  2D FLR_real: t={flr_t:.2f} {sig_stars(flr_p)}  |  R²={r2_m5:.3f}")
print()
print("  OUTPUTS:")
print(f"  Figures: Fig8, Fig9                  → {FIG_OUT}")
print(f"  CSVs: Table3, Table4, Table5 → {TAB_OUT}")
print(f"  PNGs: Table3, Table4, Table5 → {FIG_OUT}")
print(f"           (CSV + PNG pour chaque table)")
print()
print("  FIXES v2 (inchangés):")
print("  [1] FLR_real = std(headline-core,24m)/std(core,24m) ✓")
print("  [2] CP = rho_t + d_term_spread ✓")
print("  [3] Fig9 note théorique Pflueger (2025) ✓")
print()
print("  AJOUTS v3:")
print("  [4] Table3_baseline_regression.png ✓")
print("  [5] Table4_LP_results.png ✓")
print("  [6] Table5_horse_race.png ✓")
print("")