# Neuschnee-Vergleich

Vergleicht täglich den Neuschnee eines Modells mit zwei Quellen, Zelle für Zelle über der Schweiz:

| Quelle | Was | Wie gelesen |
|---|---|---|
| PowderGuide | Prognose ICON-EU, Neuschnee (`snow_gsp__lsfwe`), 24 h des nächsten Tages, 1 mm = 1 cm | Farbkacheln (Zoom 6) über die Farbskala zurück in cm |
| SLF / WhiteRisk | gemessener Neuschnee 24 h (HN1D) bis 06 UTC | codiertes Bild der Seite (Rot = Wert 1–100 cm, Blau = Daten vorhanden) |
| Modell | Summe der Stundenwerte im selben Zeitraum, nach 24 h Setzung | `data/latest.json` → `snowdata-*.json.gz` der Modell-Seite |

Kennzahlen (1-km-Zellen und 10-km-Blöcke): Bias, MAE, RMSE, Pearson/Spearman, Anteil innerhalb ±2 cm / ±50 %,
Erkennung ≥1/5/10/20 cm (POD, FAR, CSI, Frequenz-Bias), Klassen-Übereinstimmung, nach Höhe und Region.
Der Bericht enthält Links, um dieselbe Karte bei PowderGuide und WhiteRisk anzusehen, und den Durchschnitt über alle bisherigen Tage.

## Einrichten

1. Neues Repo anlegen – **privat**, sonst sind Code und Berichte öffentlich.
2. Den Inhalt dieses Ordners hochladen (inkl. `.github/`).
3. *Settings → Secrets and variables → Actions → New repository secret*:
   `MODEL_SITE` = Adresse der Modell-Seite (die mit `data/latest.json`), z. B. `https://…github.io/…/`.
4. *Settings → Actions → General → Workflow permissions*: «Read and write» (für den Branch `history`).
5. *Actions → Compare new snow → Run workflow* für den ersten Lauf. Danach täglich 08:41 UTC.

Ergebnisse: Zusammenfassung jedes Laufs, Artefakt `report-…` (Bericht, JSON, Karten),
Branch `history` mit `history.csv`, `latest.md` und `reports/<Datum>/`.

## Lokal

```
pip install -r requirements.txt
MODEL_SITE=https://… python compare_newsnow.py --out out --history out/history.csv
```

Optionen: `--pg-run 20261117`, `--pg-day DAY072` (Standard `auto` = morgen), `--hn-date 2026-11-17`, `--sm-file datei.json.gz`.

## Hinweise

- Die Anfragen tragen eine neutrale Kennung; pro Lauf sind es ~5 Anfragen bei PowderGuide und 2 beim SLF.
- PowderGuide-Kacheln sind für deren Karte gedacht; ein Wechsel auf ICON-EU direkt vom DWD (Open Data) wäre die saubere Alternative.
- Ein Tag ist Zufall – erst über Wochen wird der Bias belastbar genug zum Kalibrieren.
