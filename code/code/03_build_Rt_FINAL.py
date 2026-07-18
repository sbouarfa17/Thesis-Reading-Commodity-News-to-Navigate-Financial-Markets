# ============================================================
# 03_build_Rt_FINAL.py
# Corrections appliquees :
# - Fig2 : label "Supply regime (-)" deplace vers 2013 (zone rho_t < 0)
# - Notes Fig2 : ugap = NAIRU minus UNRATE explicite
# - Notes Fig1 : Rt mesure la volatilite relative, pas la direction
# - Aucun changement de formule ni de fenetre
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

CLEAN  = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\cleaned"
OUTPUT = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\output\figures"
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.labelcolor': 'black',
    'text.color': 'black',
})

# ── CHARGEMENT ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(CLEAN, "macro_monthly.csv"),
                 parse_dates=["date"], index_col="date")
df = df.loc["2001-05-01":"2023-06-01"].copy()

# ── CONSTRUCTION DE Rt ───────────────────────────────────────
# Rt = sigma(NetDemand) / [sigma(NetDemand) + sigma(NetSupply)]
# Fenetre baseline : 24 mois. Robustesse 12-60 dans 04_local_projections.py
# NOTE : Rt mesure la VOLATILITE relative du signal demande vs offre,
#        pas la direction. Un Rt eleve signifie que les nouvelles de demande
#        ont ete plus variables recemment, pas necessairement positives.
window = 24
df["sigma_demand"] = df["net_demand"].rolling(window=window, min_periods=window).std()
df["sigma_supply"] = df["net_supply"].rolling(window=window, min_periods=window).std()
df["R_t"] = df["sigma_demand"] / (df["sigma_demand"] + df["sigma_supply"])

# ── CONSTRUCTION DE rho_t ────────────────────────────────────
# ugap = NAIRU - UNRATE : positif quand economie forte (proxy du output gap)
# Coherent avec Campbell, Pflueger et Viceira (2020) qui utilisent le output gap
# rho_t > 0 : regime demande (inflation et ugap co-bougent)
# rho_t < 0 : regime offre / stagflation (inflation monte, ugap baisse)
df["rho_t"] = df["inflation_yoy"].rolling(window=window, min_periods=window).corr(df["ugap"])

# ── STATS ────────────────────────────────────────────────────
print("=== Rt ===")
print(f"Moyen : {df['R_t'].mean():.3f}")
print(f"Min   : {df['R_t'].min():.3f}")
print(f"Max   : {df['R_t'].max():.3f}")
print()
print("=== rho_t ===")
print(f"Moyen 2001-2007 : {df.loc[:'2007','rho_t'].mean():.3f}")
print(f"Moyen 2009-2023 : {df.loc['2009':,'rho_t'].mean():.3f}")
print()
corr_val = df[['R_t','rho_t']].corr().iloc[0,1]
corr_str = f"{corr_val:+.2f}"
print(f"Correlation Rt / rho_t : {corr_val:.3f}")

# ── SAUVEGARDE CSV ───────────────────────────────────────────
df[["R_t","rho_t","inflation_yoy","ugap","policy_rate",
    "vix","dollar_index","term_spread","indpro_growth"]].to_csv(
    os.path.join(CLEAN, "master_Rt.csv"))
print("\n✓ master_Rt.csv sauvegarde")

# ── HELPER : shading data-driven ─────────────────────────────
def add_regime_shading(ax, rt_series):
    rt = rt_series.dropna()
    in_d, in_s = False, False
    d_start, s_start = None, None
    for date, val in rt.items():
        if val > 0.5:
            if not in_d:
                d_start = date; in_d = True
            if in_s:
                ax.axvspan(s_start, date, alpha=0.10, color='salmon', zorder=1)
                in_s = False
        else:
            if not in_s:
                s_start = date; in_s = True
            if in_d:
                ax.axvspan(d_start, date, alpha=0.10, color='steelblue', zorder=1)
                in_d = False
    last = rt.index[-1]
    if in_d: ax.axvspan(d_start, last, alpha=0.10, color='steelblue', zorder=1)
    if in_s: ax.axvspan(s_start, last, alpha=0.10, color='salmon', zorder=1)

# ════════════════════════════════════════════════════════════
# FIGURE 1 : Rt
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))
add_regime_shading(ax, df['R_t'])
ax.plot(df.index, df['R_t'], color='black', linewidth=1.4, zorder=3)
ax.axhline(0.5, color='black', linestyle='--', linewidth=0.7, zorder=2)

ax.annotate('Supply constraints\n(OPEC / bottlenecks)',
            xy=(pd.Timestamp('2006-01-01'), 0.32),
            xytext=(pd.Timestamp('2005-06-01'), 0.22),
            fontsize=8.5, color='firebrick', style='italic',
            fontweight='bold', ha='center')
ax.annotate('GFC rebound\n(stimulus demand)',
            xy=(pd.Timestamp('2009-08-01'), 0.68),
            xytext=(pd.Timestamp('2009-08-01'), 0.82),
            fontsize=8.5, color='steelblue', style='italic',
            fontweight='bold', ha='center')
ax.annotate('Oil glut / Shale\n(supply dominance)',
            xy=(pd.Timestamp('2016-01-01'), 0.38),
            xytext=(pd.Timestamp('2016-01-01'), 0.23),
            fontsize=8.5, color='firebrick', style='italic',
            fontweight='bold', ha='center')
ax.annotate('COVID recovery',
            xy=(pd.Timestamp('2021-04-01'), 0.62),
            xytext=(pd.Timestamp('2021-01-01'), 0.82),
            fontsize=8.5, color='steelblue', style='italic',
            fontweight='bold', ha='center')
ax.text(pd.Timestamp('2022-09-01'), 0.22,
        'Ukraine', fontsize=8.5, color='firebrick',
        style='italic', fontweight='bold', ha='center')

ax.set_ylim(0.15, 0.95)
ax.set_ylabel('$R_t$', fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.text(0, 1.08, 'F I G U R E   1', transform=ax.transAxes,
        fontsize=8, color='gray', va='bottom')
ax.text(0, 1.02,
        r'News-Based Demand/Supply Regime Indicator $-$ $R_t$',
        transform=ax.transAxes, fontsize=11, color='black',
        va='bottom', style='italic')

# CORRECTION : note precise que Rt mesure la volatilite, pas la direction
ax.annotate(
    r'Notes: $R_t = \dfrac{\sigma(\mathrm{NetDemand}_t)}{\sigma(\mathrm{NetDemand}_t)'
    r' + \sigma(\mathrm{NetSupply}_t)}$, 24-month rolling window, '
    'composite indices from Malliaropulos et al. (2025).\n'
    r'$R_t$ captures the relative volatility of demand vs. supply news, not their direction. '
    r'Blue shading $=$ demand-dominated ($R_t > 0.5$). Red shading $=$ supply-dominated ($R_t < 0.5$).',
    xy=(0, -0.22), xycoords='axes fraction',
    fontsize=8, color='dimgray', va='top')

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(os.path.join(OUTPUT, "Fig1_Rt.png"), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 1 sauvegardee")

# ════════════════════════════════════════════════════════════
# FIGURE 2 : rho_t
# CORRECTION : label "Supply regime (-)" deplace vers 2013
#              (zone ou rho_t est effectivement negatif)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))

ax.plot(df.index, df['rho_t'], color='black', linewidth=1.4)
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.fill_between(df.index, df['rho_t'], 0,
                where=(df['rho_t'] > 0), alpha=0.15, color='steelblue')
ax.fill_between(df.index, df['rho_t'], 0,
                where=(df['rho_t'] < 0), alpha=0.15, color='salmon')

# Label "Demand regime" : en 2022 (rho_t ~ 0.95, clairement positif)
ax.text(pd.Timestamp('2022-06-01'), 1.02,
        'Demand regime (+)',
        fontsize=8.5, color='steelblue', ha='center',
        style='italic', fontweight='bold')

# CORRECTION : "Supply regime" deplace vers 2013-2014 (rho_t ~ -0.8, clairement negatif)
ax.text(pd.Timestamp('2013-06-01'), -1.02,
        'Supply regime (\u2212)',
        fontsize=8.5, color='firebrick', ha='center',
        style='italic', fontweight='bold')

ax.set_ylim(-1.18, 1.18)
ax.set_ylabel(r'$\rho_t$', fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.text(0, 1.08, 'F I G U R E   2', transform=ax.transAxes,
        fontsize=8, color='gray', va='bottom')
ax.text(0, 1.02,
        r'Rolling Correlation: CPI Inflation and Unemployment Gap $-$ $\rho_t$',
        transform=ax.transAxes, fontsize=11, color='black',
        va='bottom', style='italic')

# CORRECTION : note precise que ugap = NAIRU - UNRATE
ax.annotate(
    r'Notes: 24-month rolling correlation between annualised CPI inflation '
    r'and the unemployment gap (ugap $=$ NAIRU $-$ UNRATE, a positive-output-gap proxy).'
    '\nPositive values correspond to demand-dominated regimes '
    'as in Campbell, Pflueger and Viceira (2020). '
    'Negative values indicate supply-dominated (stagflationary) dynamics.',
    xy=(0, -0.22), xycoords='axes fraction',
    fontsize=8, color='dimgray', va='top')

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(OUTPUT, "Fig2_rho_t.png"), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 2 sauvegardee")

# ════════════════════════════════════════════════════════════
# FIGURE 3 : dual axis — inchange
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(13, 4.5))
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_linewidth(0.8)

l1, = ax1.plot(df.index, df['R_t'],   color='black', linewidth=1.4)
l2, = ax2.plot(df.index, df['rho_t'], color='black', linewidth=1.2, linestyle='--')

ax1.set_ylabel('$R_t$', fontsize=12, style='italic')
ax2.set_ylabel(r'$\rho_t$', fontsize=12, style='italic')
ax1.tick_params(axis='y', colors='black')
ax2.tick_params(axis='y', colors='black')
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax1.text(0, 1.08, 'F I G U R E   3', transform=ax1.transAxes,
         fontsize=8, color='gray', va='bottom')
ax1.text(0, 1.02,
         f'Regime Alignment $-$ $R_t$ and $\\rho_t$  (correlation: {corr_str})',
         transform=ax1.transAxes, fontsize=11, color='black',
         va='bottom', style='italic')

fig.legend([l1, l2],
           [r'$R_t$  $-$  regime indicator  (left axis)',
            r'$\rho_t$  $-$  inflation$-$output gap correlation  (right axis)'],
           loc='lower center', bbox_to_anchor=(0.5, -0.06),
           ncol=2, frameon=False, fontsize=9,
           handlelength=2.5, columnspacing=2)

ax1.annotate(
    f'Notes: Unconditional correlation between $R_t$ and $\\rho_t$: {corr_str}. '
    r'Both series use a 24-month rolling window. ugap $=$ NAIRU $-$ UNRATE.',
    xy=(0, -0.22), xycoords='axes fraction',
    fontsize=8, color='dimgray', va='top')

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(os.path.join(OUTPUT, "Fig3_Rt_rho.png"), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 3 sauvegardee")

print("\n" + "=" * 60)
print("03_build_Rt_FINAL.py COMPLETE.")
print(f"  Rt moyen        : {df['R_t'].mean():.3f}")
print(f"  Rt min / max    : {df['R_t'].min():.3f} / {df['R_t'].max():.3f}")
print(f"  Corr(Rt, rho_t) : {corr_val:.3f}")
print("Figures 1-3 sauvegardees. master_Rt.csv sauvegarde.")
