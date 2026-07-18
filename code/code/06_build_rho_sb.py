# ============================================================
# 06_build_rho_sb.py
# PHASE 3 — Step 1 & 2
#
# Step 1 : Calcul de rho_sb (rolling 24-month stock-bond correlation)
# Step 2 : Figure 7 — validation descriptive (version clean)
#
# INPUT  : Data/cleaned/ex2_monthly.csv
# OUTPUT : Data/cleaned/ex2_monthly.csv (avec rho_sb ajoutee)
#          output/figures/Fig7_rho_sb.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── CHEMINS ──────────────────────────────────────────────────
BASE   = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
CLEAN  = os.path.join(BASE, "Data", "cleaned")
OUTPUT = os.path.join(BASE, "output", "figures")
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.labelcolor': 'black',
    'text.color': 'black',
})

print("=" * 60)
print("06_build_rho_sb.py")
print("PHASE 3 — Step 1 & 2 : rho_sb + Figure 7")
print("=" * 60)

# ── CHARGEMENT ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(CLEAN, "ex2_monthly.csv"),
                 parse_dates=["date"], index_col="date")

print(f"\nDonnees chargees : {df.shape[0]} obs  |  "
      f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")

# ============================================================
# STEP 1 — CALCUL DE rho_sb
# Rolling 24-month Pearson correlation entre sp500_ret et bond_ret
# Variable dependante de tout l'exercice 2
# ============================================================
print("\n[STEP 1] Calcul de rho_sb...")

WINDOW = 24
df["rho_sb"] = (df["sp500_ret"]
                .rolling(window=WINDOW, min_periods=WINDOW)
                .corr(df["bond_ret"]))

rho_valid = df["rho_sb"].dropna()

print(f"   ✓ rho_sb calcule : {len(rho_valid)} observations")
print(f"   Periode  : {rho_valid.index[0].strftime('%Y-%m')} → "
      f"{rho_valid.index[-1].strftime('%Y-%m')}")
print(f"   Moyenne  : {rho_valid.mean():.3f}")
print(f"   Mediane  : {rho_valid.median():.3f}")
print(f"   Min      : {rho_valid.min():.3f}")
print(f"   Max      : {rho_valid.max():.3f}")

neg_pct = (rho_valid < 0).mean() * 100
pos_pct = (rho_valid > 0).mean() * 100
print(f"\n   % mois rho_sb < 0 (bonds hedgent) : {neg_pct:.1f}%")
print(f"   % mois rho_sb > 0 (hedge brise)   : {pos_pct:.1f}%")

pre2022  = rho_valid[rho_valid.index < "2022-01-01"].mean()
post2022 = rho_valid[rho_valid.index >= "2022-01-01"].mean()
print(f"\n   Moyenne pre-2022  : {pre2022:.3f}  (attendu : negatif)")
print(f"   Moyenne post-2022 : {post2022:.3f}  (attendu : positif)")

# Sauvegarde ex2_monthly.csv avec rho_sb
df.to_csv(os.path.join(CLEAN, "ex2_monthly.csv"))
print(f"\n   ✓ ex2_monthly.csv mis a jour avec rho_sb")

# ============================================================
# STEP 2 — FIGURE 7 : rho_sb dans le temps (version clean)
# ============================================================
print("\n[STEP 2] Generation Figure 7...")

fig, ax = plt.subplots(figsize=(13, 5))

rho_plot = df["rho_sb"].dropna()

# Ligne principale
ax.plot(rho_plot.index, rho_plot, color="black", linewidth=1.4, zorder=3)
ax.axhline(0, color="black", linestyle="--", linewidth=0.7, zorder=2)

# Shading
ax.fill_between(rho_plot.index, rho_plot, 0,
                where=(rho_plot < 0), alpha=0.15, color="steelblue",
                zorder=1, label=r"Bonds hedge stocks ($\rho < 0$)")
ax.fill_between(rho_plot.index, rho_plot, 0,
                where=(rho_plot > 0), alpha=0.15, color="salmon",
                zorder=1, label=r"Hedge breaks ($\rho > 0$)")

# Axes
ax.set_ylim(-1.10, 1.10)
ax.set_ylabel(r"$\rho_t^{SB}$", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(rho_plot.index[0], rho_plot.index[-1])

# Legende discrete
ax.legend(loc="lower left", frameon=False, fontsize=9)

# Titre
ax.text(0, 1.08, "F I G U R E   7", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
        r"Rolling 24-Month Stock$-$Bond Correlation $-$ $\rho_t^{SB}$",
        transform=ax.transAxes, fontsize=11, color="black",
        va="bottom", style="italic")

# Note
ax.annotate(
    "Notes: 24-month rolling Pearson correlation between monthly S&P 500 Total Return "
    "(SPXT, Bloomberg) and 10-year US Treasury approximate monthly return.\n"
    r"Bond return = $(y_{t-1}/12) - 9 \times \Delta y_t$, "
    "where $y_t$ = monthly average DGS10 (FRED). "
    "Blue = bonds hedge stocks. Red = hedge breaks.",
    xy=(0, -0.22), xycoords="axes fraction",
    fontsize=8, color="dimgray", va="top")

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig_path = os.path.join(OUTPUT, "Fig7_rho_sb.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"   ✓ Figure 7 sauvegardee → {fig_path}")

# ============================================================
# RAPPORT FINAL
# ============================================================
print("\n" + "=" * 60)
print("STEP 1 & 2 COMPLETS")
print("=" * 60)
print(f"   rho_sb : {len(rho_valid)} obs  |  "
      f"moy={rho_valid.mean():.3f}  |  "
      f"[{rho_valid.min():.3f}, {rho_valid.max():.3f}]")
print(f"   Pre-2022 : {pre2022:.3f}  |  Post-2022 : {post2022:.3f}")
print(f"\n   Figure 7 → {fig_path}")
print("\nProchaine etape : 07_exercise2_regressions.py")