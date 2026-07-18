# ============================================================
# 05_clean_merge_ex2.py  —  EXERCISE 2 : DATA CLEANING & MERGE
# Compatible pandas 3.x
#
# Ce code :
#   1. Nettoie sp500tr.csv  (index mensuel SPXT → returns mensuels)
#   2. Nettoie DGS10.csv    (yield journalier → return obligataire mensuel)
#   3. Fusionne avec master_Rt.csv (déjà produit en exercice 1)
#   4. Produit ex2_monthly.csv  (dataset complet pour l'exercice 2)
#   5. Produit Figure 7 : rho_sb dans le temps
#
# INPUTS  (dans Data/raw/stock_bond/) :
#   - sp500tr.csv   : index SPXT mensuel Bloomberg (date, px_last)
#   - DGS10.csv     : yield 10Y journalier FRED (observation_date, DGS10)
#
# INPUT existant (dans Data/cleaned/) :
#   - master_Rt.csv : produit par 02_clean_merge.py + 03_build_Rt.py
#
# OUTPUT (dans Data/cleaned/) :
#   - ex2_monthly.csv
# OUTPUT (dans output/figures/) :
#   - Fig7_rho_sb.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── CHEMINS ──────────────────────────────────────────────────
BASE   = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
RAW_SB = os.path.join(BASE, "Data", "raw", "stock_bond")
CLEAN  = os.path.join(BASE, "Data", "cleaned")
OUTPUT = os.path.join(BASE, "output", "figures")
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

print("=" * 60)
print("05_clean_merge_ex2.py — EXERCISE 2 : DATA CLEANING")
print("=" * 60)

# ============================================================
# ETAPE 1 — NETTOYAGE sp500tr.csv
# Index mensuel SPXT Bloomberg → returns mensuels
# ============================================================
print("\n[ETAPE 1] Nettoyage S&P 500 Total Return (sp500tr.csv)...")

sp = pd.read_csv(os.path.join(RAW_SB, "sp500tr.csv"), parse_dates=["date"])
sp = sp.rename(columns={"px_last": "spxt_level"})

# Aligne sur le début du mois — compatible pandas 3.x
sp["date"] = sp["date"].dt.to_period("M").dt.to_timestamp()
sp = sp.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

# Vérifie les valeurs manquantes
n_na = sp["spxt_level"].isna().sum()
print(f"   Valeurs manquantes : {n_na}")

# Calcule le return mensuel : r_t = (P_t / P_{t-1}) - 1
sp["sp500_ret"] = sp["spxt_level"].pct_change()

# Rapport outliers (garde tout — événements réels)
outliers_s = sp[sp["sp500_ret"].abs() > 0.10]
print(f"   Outliers |return| > 10% (réels, conservés) :")
for _, row in outliers_s.iterrows():
    print(f"     {row['date'].strftime('%Y-%m')} : {row['sp500_ret']*100:.1f}%")

print(f"   ✓ {len(sp)} observations  |  "
      f"{sp['date'].iloc[0].strftime('%Y-%m')} → {sp['date'].iloc[-1].strftime('%Y-%m')}")
print(f"   Return moyen : {sp['sp500_ret'].mean()*100:.2f}%/mois  |  "
      f"Vol : {sp['sp500_ret'].std()*100:.2f}%/mois")

# ============================================================
# ETAPE 2 — NETTOYAGE DGS10.csv
# Yield journalier FRED → return obligataire mensuel approximé
# ============================================================
print("\n[ETAPE 2] Nettoyage DGS10 (10Y Treasury yield daily, FRED)...")

dgs = pd.read_csv(os.path.join(RAW_SB, "DGS10.csv"))
dgs = dgs.rename(columns={"observation_date": "date", "DGS10": "y10"})
dgs["date"] = pd.to_datetime(dgs["date"])

# FRED marque les weekends/jours fériés avec "." → NaN
dgs["y10"] = pd.to_numeric(dgs["y10"], errors="coerce")
n_miss = dgs["y10"].isna().sum()
print(f"   Valeurs manquantes (weekends/fériés) supprimées : {n_miss}")
dgs = dgs.dropna(subset=["y10"])

print(f"   Yield 10Y : min={dgs['y10'].min():.2f}%  |  max={dgs['y10'].max():.2f}%")

# Agrège en mensuel : moyenne du mois — compatible pandas 3.x
dgs["month"] = dgs["date"].dt.to_period("M").dt.to_timestamp()
dgs_m = dgs.groupby("month")["y10"].mean().reset_index()
dgs_m = dgs_m.rename(columns={"month": "date"})
dgs_m = dgs_m.sort_values("date").reset_index(drop=True)

# Convertit en décimal
dgs_m["y10_decimal"] = dgs_m["y10"] / 100

# Calcule le return obligataire mensuel approximé
# bond_ret_t = (y_{t-1}/12) - Duration x Δy_t
# Duration = 9 pour un bond 10 ans (approximation standard)
DURATION = 9.0
dgs_m["delta_y"]  = dgs_m["y10_decimal"].diff()
dgs_m["bond_ret"] = (dgs_m["y10_decimal"].shift(1) / 12) - DURATION * dgs_m["delta_y"]

# Rapport outliers (garde tout — événements réels)
outliers_b = dgs_m[dgs_m["bond_ret"].abs() > 0.04]
print(f"   Outliers |bond_ret| > 4% (réels, conservés) :")
for _, row in outliers_b.iterrows():
    print(f"     {row['date'].strftime('%Y-%m')} : {row['bond_ret']*100:.1f}%")

print(f"   ✓ {len(dgs_m)} observations  |  "
      f"{dgs_m['date'].iloc[0].strftime('%Y-%m')} → {dgs_m['date'].iloc[-1].strftime('%Y-%m')}")
print(f"   Return moyen : {dgs_m['bond_ret'].mean()*100:.2f}%/mois  |  "
      f"Vol : {dgs_m['bond_ret'].std()*100:.2f}%/mois")

# ============================================================
# ETAPE 3 — CHARGEMENT master_Rt.csv
# ============================================================
print("\n[ETAPE 3] Chargement master_Rt.csv (produit en Exercice 1)...")

rt = pd.read_csv(os.path.join(CLEAN, "master_Rt.csv"),
                 parse_dates=["date"], index_col="date")

# Compatible pandas 3.x
rt.index = rt.index.to_period("M").to_timestamp()

print(f"   ✓ {len(rt)} obs  |  "
      f"{rt.index[0].strftime('%Y-%m')} → {rt.index[-1].strftime('%Y-%m')}")
print(f"   Colonnes : {list(rt.columns)}")

# ============================================================
# ETAPE 4 — FUSION (LEFT JOIN sur master_Rt)
# ============================================================
print("\n[ETAPE 4] Fusion des séries...")

sp_join   = sp.set_index("date")[["sp500_ret", "spxt_level"]]
bond_join = dgs_m.set_index("date")[["bond_ret", "y10_decimal"]]

df = rt.copy()
df = df.join(sp_join,   how="left")
df = df.join(bond_join, how="left")

print(f"   sp500_ret  : {df['sp500_ret'].notna().sum()} obs non-nulles")
print(f"   bond_ret   : {df['bond_ret'].notna().sum()} obs non-nulles")

# ============================================================
# ETAPE 5 — CALCUL rho_sb
# Rolling 24-month stock-bond correlation
# Variable dependante de l'exercice 2
# ============================================================
print("\n[ETAPE 5] Calcul rho_sb (rolling 24-month stock-bond correlation)...")

WINDOW = 24
df["rho_sb"] = (df["sp500_ret"]
                .rolling(window=WINDOW, min_periods=WINDOW)
                .corr(df["bond_ret"]))

rho_valid = df["rho_sb"].dropna()
print(f"   ✓ {len(rho_valid)} observations")
print(f"   Période : {rho_valid.index[0].strftime('%Y-%m')} → "
      f"{rho_valid.index[-1].strftime('%Y-%m')}")
print(f"   Moyenne : {rho_valid.mean():.3f}")
print(f"   Min : {rho_valid.min():.3f}  |  Max : {rho_valid.max():.3f}")
neg_pct = (rho_valid < 0).mean() * 100
print(f"   % mois avec rho_sb < 0 (bonds hedgent) : {neg_pct:.1f}%")

# ============================================================
# ETAPE 6 — SAUVEGARDE ex2_monthly.csv
# ============================================================
print("\n[ETAPE 6] Sauvegarde ex2_monthly.csv...")

cols_out = [
    "R_t",           # Indicateur regime demande/offre (Ex1)
    "rho_t",         # Correlation inflation-output gap (Ex1)
    "rho_sb",        # Correlation stock-bond 24m rolling <- CIBLE Ex2
    "sp500_ret",     # Return mensuel S&P 500 TR <- NOUVEAU
    "bond_ret",      # Return mensuel 10Y Treasury <- NOUVEAU
    "spxt_level",    # Niveau index SPXT (reference)
    "y10_decimal",   # Yield 10Y decimal (reference)
    "inflation_yoy", # Controle
    "ugap",          # Controle
    "policy_rate",   # Controle (Wu-Xia shadow rate)
    "vix",           # Controle
    "dollar_index",  # Controle
    "term_spread",   # Controle
    "indpro_growth", # Controle
]
cols_out = [c for c in cols_out if c in df.columns]

out_path = os.path.join(CLEAN, "ex2_monthly.csv")
df[cols_out].to_csv(out_path)
print(f"   ✓ ex2_monthly.csv → {out_path}")
print(f"     Shape : {df[cols_out].shape}")
print(f"     Colonnes : {cols_out}")

# ============================================================
# ETAPE 7 — FIGURE 7 : rho_sb dans le temps
# ============================================================
print("\n[ETAPE 7] Generation Figure 7...")

fig, ax = plt.subplots(figsize=(13, 4.5))

rho_plot = df["rho_sb"].dropna()
ax.plot(rho_plot.index, rho_plot, color="black", linewidth=1.4, zorder=3)
ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
ax.fill_between(rho_plot.index, rho_plot, 0,
                where=(rho_plot < 0), alpha=0.15, color="steelblue")
ax.fill_between(rho_plot.index, rho_plot, 0,
                where=(rho_plot > 0), alpha=0.15, color="salmon")

# Annotations
ax.annotate("Demand regime\n(bonds hedge)",
            xy=(pd.Timestamp("2006-01-01"), -0.72),
            fontsize=8.5, color="steelblue", style="italic",
            fontweight="bold", ha="center")
ax.annotate("Ukraine / supply shock",
            xy=(pd.Timestamp("2022-09-01"), 0.60),
            fontsize=8.5, color="firebrick", style="italic",
            fontweight="bold", ha="center")

ax.set_ylim(-1.15, 1.15)
ax.set_ylabel(r"$\rho_t^{SB}$", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.text(0, 1.08, "F I G U R E   7", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
        r"Rolling Stock$-$Bond Correlation $-$ $\rho_t^{SB}$",
        transform=ax.transAxes, fontsize=11, color="black",
        va="bottom", style="italic")
ax.annotate(
    "Notes: 24-month rolling Pearson correlation between monthly S&P 500 Total Return "
    "(SPXT, Bloomberg) and 10-year US Treasury approximate monthly return.\n"
    "Bond return = (y_{t-1}/12) - 9 x delta_y_t, where y_t = monthly average DGS10 (FRED). "
    "Blue = bonds hedge stocks. Red = hedge breaks.",
    xy=(0, -0.26), xycoords="axes fraction",
    fontsize=8, color="dimgray", va="top")

plt.tight_layout(rect=[0, 0.12, 1, 1])
fig_path = os.path.join(OUTPUT, "Fig7_rho_sb.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"   ✓ Figure 7 sauvegardee → {fig_path}")

# ============================================================
# RAPPORT FINAL
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print(df[["sp500_ret", "bond_ret", "rho_sb"]].describe().round(4).to_string())
print(f"\nFichiers produits :")
print(f"  → {out_path}")
print(f"  → {fig_path}")
print("\nProchaine etape : 06_exercise2_regressions.py")