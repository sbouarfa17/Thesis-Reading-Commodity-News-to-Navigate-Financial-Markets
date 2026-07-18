# ============================================================
# 08_figure_three_versions.py
# Génère 3 versions du graphique combiné Rt + rho_t + rho_sb
#
# INPUT  : Data/cleaned/master_Rt.csv
#          Data/cleaned/ex2_monthly.csv
# OUTPUT : output/figures/FigX_A_singleaxis.png
#          output/figures/FigX_B_dualaxis.png
#          output/figures/FigX_C_panels.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import os

# ── CHEMINS ──────────────────────────────────────────────────
BASE   = r"C:\Users\Sofia\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
CLEAN  = os.path.join(BASE, "Data", "cleaned")
OUTPUT = os.path.join(BASE, "output", "figures")
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.labelcolor': 'black',
    'text.color': 'black',
})

# ── CHARGEMENT ───────────────────────────────────────────────
master = pd.read_csv(os.path.join(CLEAN, "master_Rt.csv"),
                     parse_dates=["date"], index_col="date")
ex2    = pd.read_csv(os.path.join(CLEAN, "ex2_monthly.csv"),
                     parse_dates=["date"], index_col="date")

df = master[["R_t", "rho_t"]].join(ex2[["rho_sb"]], how="inner")
df = df.dropna(subset=["R_t", "rho_t", "rho_sb"])

print(f"Sample : {df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
print(f"N = {len(df)} observations")

dates = df.index
corr_rt_rho   = df["R_t"].corr(df["rho_t"])
corr_rt_rhosb = df["R_t"].corr(df["rho_sb"])
print(f"Corr(Rt, rho_t)  : {corr_rt_rho:+.3f}")
print(f"Corr(Rt, rho_sb) : {corr_rt_rhosb:+.3f}")

# ── Constantes visuelles ──────────────────────────────────────
XMIN   = dates[0]
XMAX   = dates[-1]
XLOC   = mdates.YearLocator(2)
XFMT   = mdates.DateFormatter("%Y")
C_BLUE = "#2166AC"

NOTE_SHORT = (
    r"Notes: 24-month rolling window.  "
    r"$\rho_t^{\pi,u}$ = corr(CPI inflation, unemployment gap).  "
    r"$\rho_t^{SB}$ = corr(S\&P 500 TR, 10y Treasury TR).  "
    r"Dashed lines at 0 and 0.5."
)
NOTE_LONG = (
    r"Notes: All series use a 24-month rolling window.  "
    r"$\rho_t^{\pi,u}$ = rolling corr(CPI inflation, unemployment gap).  "
    r"$\rho_t^{SB}$ = rolling corr(S\&P 500 TR, 10y Treasury TR), "
    r"bond return $= (y_{t-1}/12) - 9 \times \Delta y_t$.  Sample: 2003–2023."
)

# ════════════════════════════════════════════════════════════
# VERSION A — axe unique [-1, 1.5], graphique tall (13×9)
# Les 3 courbes partagent la même échelle
# Rt flotte entre 0.3–0.75 | rho_t autour de 0–1 | rho_sb autour de -0.5–0
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 9))

l1, = ax.plot(dates, df["R_t"],    color="black",  lw=1.8, ls="solid",  zorder=4)
l2, = ax.plot(dates, df["rho_t"],  color="black",  lw=1.2, ls="dashed", zorder=3)
l3, = ax.plot(dates, df["rho_sb"], color=C_BLUE,   lw=1.5, ls="solid",  zorder=2, alpha=0.9)

ax.axhline(0,   color="gray",  ls="--", lw=0.6, alpha=0.45)
ax.axhline(0.5, color="black", ls="--", lw=0.6, alpha=0.35)

ax.set_ylim(-1, 1.5)
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5])
ax.set_ylabel(r"$R_t$  /  Correlation", fontsize=11, style="italic")
ax.spines["right"].set_visible(False)
ax.xaxis.set_major_locator(XLOC)
ax.xaxis.set_major_formatter(XFMT)
ax.set_xlim(XMIN, XMAX)

ax.text(0, 1.05, "F I G U R E   3", transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")
ax.text(0, 1.02,
    r"Regime Alignment $-$ $R_t$,  $\rho_t^{\pi,u}$  and  $\rho_t^{SB}$"
    f"   (corr $R_t$/$\\rho^{{\\pi,u}}$: {corr_rt_rho:+.2f}"
    f"  |  corr $R_t$/$\\rho^{{SB}}$: {corr_rt_rhosb:+.2f})",
    transform=ax.transAxes, fontsize=11, color="black", va="bottom", style="italic")

fig.legend([l1, l2, l3],
    [r"$R_t$  $-$  regime indicator",
     r"$\rho_t^{\pi,u}$  $-$  inflation$-$output gap corr",
     r"$\rho_t^{SB}$  $-$  stock$-$bond correlation"],
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=3, frameon=False, fontsize=9, handlelength=2.5, columnspacing=1.8)

ax.annotate(NOTE_SHORT, xy=(0, -0.07), xycoords="axes fraction",
            fontsize=8, color="dimgray", va="top")

plt.tight_layout(rect=[0, 0.05, 1, 1])
path_a = os.path.join(OUTPUT, "FigX_A_singleaxis.png")
plt.savefig(path_a, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n✓ Version A → {path_a}")

# ════════════════════════════════════════════════════════════
# VERSION B — double axe
# Axe gauche : Rt [0.15–0.95]
# Axe droit  : rho_t + rho_sb [-1.35–1.35]
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(0.8)
ax1.spines["right"].set_visible(False)

l3, = ax2.plot(dates, df["rho_sb"], color=C_BLUE,  lw=1.4, ls="solid",  zorder=2, alpha=0.9)
l2, = ax2.plot(dates, df["rho_t"],  color="black", lw=1.2, ls="dashed", zorder=3)
l1, = ax1.plot(dates, df["R_t"],    color="black", lw=1.7, ls="solid",  zorder=4)

ax1.axhline(0.5, color="black", ls="--", lw=0.6, alpha=0.4)
ax2.axhline(0,   color="gray",  ls="--", lw=0.55, alpha=0.5)

ax1.set_ylim(0.15, 0.95)
ax2.set_ylim(-1.35, 1.35)
ax1.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax2.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax1.set_ylabel(r"$R_t$", fontsize=13, style="italic")
ax2.set_ylabel(r"$\rho_t^{\pi,u}$  /  $\rho_t^{SB}$", fontsize=12, style="italic")
ax1.xaxis.set_major_locator(XLOC)
ax1.xaxis.set_major_formatter(XFMT)
ax1.set_xlim(XMIN, XMAX)

ax1.text(0, 1.08, "F I G U R E   X", transform=ax1.transAxes,
         fontsize=8, color="gray", va="bottom")
ax1.text(0, 1.02,
    r"Regime Alignment $-$ $R_t$,  $\rho_t^{\pi,u}$  and  $\rho_t^{SB}$"
    f"   (corr $R_t$/$\\rho^{{\\pi,u}}$: {corr_rt_rho:+.2f}"
    f"  |  corr $R_t$/$\\rho^{{SB}}$: {corr_rt_rhosb:+.2f})",
    transform=ax1.transAxes, fontsize=11, color="black", va="bottom", style="italic")

fig.legend([l1, l2, l3],
    [r"$R_t$  $-$  regime indicator  (left axis)",
     r"$\rho_t^{\pi,u}$  $-$  inflation$-$output gap corr  (right axis)",
     r"$\rho_t^{SB}$  $-$  stock$-$bond correlation  (right axis)"],
    loc="lower center", bbox_to_anchor=(0.5, -0.07),
    ncol=3, frameon=False, fontsize=9, handlelength=2.5, columnspacing=1.8)

ax1.annotate(NOTE_LONG, xy=(0, -0.20), xycoords="axes fraction",
             fontsize=8, color="dimgray", va="top")

plt.tight_layout(rect=[0, 0.10, 1, 1])
path_b = os.path.join(OUTPUT, "FigX_B_dualaxis.png")
plt.savefig(path_b, dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Version B → {path_b}")

# ════════════════════════════════════════════════════════════
# VERSION C — 2 panels empilés, axe x partagé
# Panel (a) : Rt gauche + rho_t droite  (= Fig3 originale)
# Panel (b) : rho_sb seul avec fill     (= Fig7 originale)
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 8))
gs  = gridspec.GridSpec(2, 1, hspace=0.06, height_ratios=[1, 1], figure=fig)

ax_a  = fig.add_subplot(gs[0])
ax_ar = ax_a.twinx()
ax_b  = fig.add_subplot(gs[1], sharex=ax_a)

# Panel (a) ───────────────────────────────────────────────
ax_ar.spines["top"].set_visible(False)
ax_ar.spines["right"].set_visible(True)
ax_ar.spines["right"].set_linewidth(0.8)
ax_a.spines["right"].set_visible(False)

ax_ar.plot(dates, df["rho_t"], color="black", lw=1.2, ls="dashed", zorder=3)
ax_a.plot(dates,  df["R_t"],   color="black", lw=1.7, ls="solid",  zorder=4)

ax_a.axhline(0.5,  color="black", ls="--", lw=0.6, alpha=0.35)
ax_ar.axhline(0,   color="gray",  ls="--", lw=0.55, alpha=0.5)

ax_a.set_ylim(0.15, 0.95)
ax_ar.set_ylim(-1.35, 1.35)
ax_a.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax_ar.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax_a.set_ylabel(r"$R_t$", fontsize=12, style="italic")
ax_ar.set_ylabel(r"$\rho_t^{\pi,u}$", fontsize=12, style="italic")

ax_a.text(0.01, 0.95, "(a)", transform=ax_a.transAxes,
          fontsize=10, color="gray", va="top")
ax_a.text(0.045, 0.95,
    r"$R_t$ (left axis)  and  $\rho_t^{\pi,u}$ inflation–output gap corr (right axis)"
    f"  — corr: {corr_rt_rho:+.2f}",
    transform=ax_a.transAxes, fontsize=9.5, color="black", va="top", style="italic")

plt.setp(ax_a.get_xticklabels(), visible=False)

# Panel (b) ───────────────────────────────────────────────
ax_b.spines["right"].set_visible(False)

ax_b.plot(dates, df["rho_sb"], color=C_BLUE, lw=1.5, ls="solid", zorder=3)
ax_b.fill_between(dates, df["rho_sb"], 0,
    where=(df["rho_sb"] < 0), alpha=0.15, color=C_BLUE, zorder=1,
    label=r"Bonds hedge stocks ($\rho^{SB} < 0$)")
ax_b.fill_between(dates, df["rho_sb"], 0,
    where=(df["rho_sb"] > 0), alpha=0.15, color="salmon", zorder=1,
    label=r"Hedge breaks ($\rho^{SB} > 0$)")
ax_b.axhline(0, color="black", ls="--", lw=0.7)

ax_b.set_ylim(-1.10, 1.10)
ax_b.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax_b.set_ylabel(r"$\rho_t^{SB}$", fontsize=12, style="italic")
ax_b.legend(loc="lower left", frameon=False, fontsize=8.5)

ax_b.text(0.01, 0.95, "(b)", transform=ax_b.transAxes,
          fontsize=10, color="gray", va="top")
ax_b.text(0.045, 0.95,
    r"$\rho_t^{SB}$  24-month rolling stock–bond correlation"
    f"  — corr with $R_t$: {corr_rt_rhosb:+.2f}",
    transform=ax_b.transAxes, fontsize=9.5, color="black", va="top", style="italic")

ax_b.xaxis.set_major_locator(XLOC)
ax_b.xaxis.set_major_formatter(XFMT)
ax_b.set_xlim(XMIN, XMAX)

fig.text(0.012, 0.985, "F I G U R E   X", fontsize=7.5, color="gray", va="top")
fig.text(0.012, 0.972,
    r"Regime Alignment $-$ $R_t$,  $\rho_t^{\pi,u}$  and  $\rho_t^{SB}$",
    fontsize=12, color="black", va="top", style="italic")

fig.text(0.012, 0.022, NOTE_LONG, fontsize=7.5, color="dimgray", va="bottom")

path_c = os.path.join(OUTPUT, "FigX_C_panels.png")
plt.savefig(path_c, dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Version C → {path_c}")

print("\n✓ Toutes les figures générées.")