# ============================================================
# 02_clean_merge.py
# ============================================================

import pandas as pd
import numpy as np
import os

RAW_SIDE   = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\raw\Side of Pt"
RAW_MP     = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\raw\Mp control variable"
RAW_ADD    = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\raw\additional controls"
RAW_IND    = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\raw\indice prof"
CLEAN      = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance\Data\cleaned"

os.makedirs(CLEAN, exist_ok=True)

# ── CHARGEMENT ───────────────────────────────────────────────

cpi = pd.read_csv(os.path.join(RAW_SIDE, "Headline CPI.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()

unrate = pd.read_csv(os.path.join(RAW_SIDE, "UNRATE.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()

nairu_q = pd.read_csv(os.path.join(RAW_SIDE, "NAIRU.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()

fedfunds = pd.read_csv(os.path.join(RAW_MP, "FEDFUNDS.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()

shadow_raw = pd.read_csv(os.path.join(RAW_MP, "shadowrate_US.xls - Sheet1.csv"),
    header=None, names=["date", "shadow_rate"])
shadow_raw["date"] = pd.to_datetime(shadow_raw["date"].astype(str), format="%Y%m")
shadow = shadow_raw.set_index("date")["shadow_rate"]

t10y2y_d = pd.read_csv(os.path.join(RAW_ADD, "T10Y2Y.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()
t10y2y = t10y2y_d.resample("MS").mean()

vix_d = pd.read_csv(os.path.join(RAW_ADD, "old twusdol1999 2005.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()
vix = vix_d.resample("MS").mean()

twexb_d = pd.read_csv(os.path.join(RAW_ADD, "TWEXB.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()
twexb = twexb_d.resample("MS").mean()

dtwexbgs_d = pd.read_csv(os.path.join(RAW_ADD, "new twusdol1999 2005.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()
dtwexbgs = dtwexbgs_d.resample("MS").mean()

scale = dtwexbgs.loc["2006-01-01"] / twexb.loc["2005-12-01"]
twexb_rebased = twexb * scale
dollar = pd.concat([
    twexb_rebased[twexb_rebased.index < "2006-01-01"],
    dtwexbgs[dtwexbgs.index >= "2006-01-01"]
])

indpro = pd.read_csv(os.path.join(RAW_ADD, "INDPRO.csv"),
    parse_dates=["observation_date"], index_col="observation_date").squeeze()

comm = pd.read_csv(os.path.join(RAW_IND, "output_commodities_articleused_m.csv"),
    parse_dates=["date"], index_col="date")
net_demand = comm["std_netD_composite"]
net_supply  = comm["std_netS_composite"]

# ── TRANSFORMATIONS ──────────────────────────────────────────

inflation_yoy = cpi.pct_change(12) * 100

dates_monthly = pd.date_range("1999-01-01", "2024-12-01", freq="MS")
nairu_monthly = nairu_q.reindex(dates_monthly).interpolate(method="linear")
ugap = nairu_monthly - unrate

policy_rate = fedfunds.copy()
shadow_m = shadow.resample("MS").last()
zlb = (policy_rate.index >= "2009-07-01") & (policy_rate.index <= "2015-12-01")
policy_rate[zlb] = shadow_m.reindex(policy_rate.index)[zlb]

indpro_growth = np.log(indpro).diff() * 100

# ── FUSION ───────────────────────────────────────────────────

master = pd.DataFrame({
    "inflation_yoy" : inflation_yoy,
    "ugap"          : ugap,
    "policy_rate"   : policy_rate,
    "vix"           : vix,
    "dollar_index"  : dollar,
    "term_spread"   : t10y2y,
    "indpro_growth" : indpro_growth,
    "net_demand"    : net_demand,
    "net_supply"    : net_supply,
})
master.index.name = "date"
master = master.sort_index()

# ── VÉRIFICATION ─────────────────────────────────────────────

print("Shape:", master.shape)
print("\nValeurs manquantes:")
print(master.isnull().sum())
print("\nStatistiques descriptives:")
print(master.describe().round(3))

# ── SAUVEGARDE ───────────────────────────────────────────────

master.to_csv(os.path.join(CLEAN, "macro_monthly.csv"))
print("\n✓ Sauvegardé : Data/cleaned/macro_monthly.csv")