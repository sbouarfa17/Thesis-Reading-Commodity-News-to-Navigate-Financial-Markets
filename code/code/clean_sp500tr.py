# ============================================================
# clean_sp500tr.py
# Nettoie le fichier Bloomberg SPXT copié-collé
# → produit sp500tr.csv avec 2 colonnes propres : date, px_last
# ============================================================
# INSTRUCTIONS :
#   1. Mets ce fichier dans ton dossier code/
#   2. Mets SPT500.csv dans data/raw/stock_bond/
#   3. Lance le script
#   4. Il génère data/raw/stock_bond/sp500tr.csv
# ============================================================

import pandas as pd
import re
import os

# ── CHEMINS ──────────────────────────────────────────────────
BASE      = r"C:\Users\bouaso22\OneDrive - Université Paris-Dauphine\M1 Finance\thèse M1 Finance"
INPUT     = os.path.join(BASE, "Data", "raw", "stock_bond", "SPT500.csv")
OUTPUT    = os.path.join(BASE, "Data", "raw", "stock_bond", "sp500tr.csv")

# ── LECTURE BRUTE ────────────────────────────────────────────
with open(INPUT, encoding="utf-8-sig") as f:
    lines = f.readlines()

# ── EXTRACTION DES PAIRES (date, prix) ───────────────────────
# Format Bloomberg : colonnes répétées Date;LastPrice;BidPrice;Date;LastPrice;BidPrice;...
# On cherche les lignes qui ressemblent à : "MM/DD/YY; ;1234.56"

date_price_pairs = []

# Regex : date format MM/DD/YY ou MM/DD/YYYY
date_pattern = re.compile(r'(\d{2}/\d{2}/\d{2,4})')
price_pattern = re.compile(r'[\d,]+\.\d+')

for line in lines:
    # Ignore les lignes d'en-tête et métadonnées
    if "SPXT" in line or "Range" in line or "Fields" in line or \
       "View" in line or "Date" in line or "High" in line or \
       "Low" in line or "Average" in line or "Net Chg" in line or \
       "Page" in line or "S&P" in line:
        continue

    # Découpe par ";"
    parts = [p.strip() for p in line.split(";")]

    # Cherche toutes les paires date+prix dans la ligne
    i = 0
    while i < len(parts):
        part = parts[i]
        # Cherche une date
        date_match = date_pattern.match(part.strip().replace("H", "").replace("L", "").strip())
        if date_match and i + 1 < len(parts):
            date_str = date_match.group(1)
            # Cherche le prix dans les colonnes suivantes
            # Bloomberg met parfois un espace avant le prix
            price_str = None
            for j in range(i+1, min(i+3, len(parts))):
                clean = parts[j].replace(",", "").replace("H", "").replace("L", "").strip()
                if price_pattern.match(clean):
                    price_str = clean
                    break
            if price_str:
                try:
                    price = float(price_str)
                    date_price_pairs.append((date_str, price))
                except:
                    pass
        i += 1

# ── CREATION DU DATAFRAME ─────────────────────────────────────
df = pd.DataFrame(date_price_pairs, columns=["date", "px_last"])

# Parse les dates (format MM/DD/YY ou MM/DD/YYYY)
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
mask = df["date"].isna()
df.loc[mask, "date"] = pd.to_datetime(
    df.loc[mask, "date_raw"] if "date_raw" in df.columns else df.loc[mask, "date"],
    format="%m/%d/%Y", errors="coerce"
)

# Supprime les doublons (Bloomberg répète parfois les données sur plusieurs pages)
df = df.dropna(subset=["date", "px_last"])
df = df.drop_duplicates(subset=["date"])

# Filtre 1999-2024
df = df[(df["date"] >= "1999-01-01") & (df["date"] <= "2024-12-31")]

# Trie par date croissante
df = df.sort_values("date").reset_index(drop=True)

# Formate la date en YYYY-MM-DD
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

# ── SAUVEGARDE ───────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False)

# ── AFFICHAGE ────────────────────────────────────────────────
print("=" * 50)
print(f"✓ sp500tr.csv sauvegardé → {OUTPUT}")
print(f"  Nombre de lignes : {len(df)}")
print(f"  Période : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
print()
print("Premières lignes :")
print(df.head(10).to_string(index=False))
print()
print("Dernières lignes :")
print(df.tail(10).to_string(index=False))
