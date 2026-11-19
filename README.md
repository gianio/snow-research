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

---

# Validierungsdaten sammeln

Zusätzlich zum täglichen Kartenvergleich sammeln drei Skripte Punktwerte (Koordinate, Zeit, Wert) aus vielen Quellen.
Der Workflow *Collect validation data* (07:17 und 18:47 UTC, mit Zufallsverzögerung) legt sie auf dem Branch `data` ab.

| Skript | Quelle | Grösse | Zeit |
|---|---|---|---|
| `scrape_snowmaps.py` | SLF/WhiteRisk Neuschneekarte HN1D/HN3D | `slf_hn/hn24`, `hn72` | gemessen, zurück bis `--days-back` |
| | SLF Schneehöhenkarte (nur mit Variable `SLF_HS_URL`, Vorlage mit `{date}`) | `slf_hs/hs` | gemessen |
| | SLF IMIS-Stationen (alle ~200) | `imis/hs`, `imis/ta` | gemessen, stündlich (API liefert nur 24 h → 2 Läufe/Tag) |
| | snow-forecast.com, Orte in `config/snowforecast_resorts.txt` | `snowforecast/snow_period` (Prognose je Tagesdrittel, Berg/Mitte/Tal), `hs_reported` | Prognose + gemeldete Schneehöhe |
| | PowderGuide ICON-D2/EU, ExoLabs (`--sources powderguide,exolabs`) | `hn24`, `hs` | **aus**: deren Kachel-Server sagt in robots.txt `Disallow: /` |
| `scrape_gipfelbuch.py` | neue Verhältnisberichte auf gipfelbuch.ch (über die Sitemaps) | `surface_top/bottom`, `snowline_m`, `aper_below_m`, `ski_from_m`, `skidepot_m`, `snow_cm_new/total`, `avalanche_perception`, `rating` | Tourdatum, Koordinate des Routenpunkts |
| `validate_obs.py` | alles oben gegen die nächste Modellzelle | Bias, MAE, RMSE, r, POD/FAR | |

Punkte für Karten: Raster über der Schweiz (`--grid-km`, Standard 5 km, nur wo die SLF-Karte Daten hat) plus alle IMIS-Stationen –
so liegen alle Kartenquellen auch genau an den Stationen vor.

Format (`snowval/store.py`): `data/obs/<quelle>/<grösse>/<tag>.csv.gz` mit
`source, var, kind, lat, lon, elev, valid_from, valid_to, issued, lead_h, value, unit, point_id, ref`.
`python -m snowval.store --data data --db data/obs.sqlite` macht daraus eine SQLite-Tabelle.

### gipfelbuch.ch
Ohne Login zeigt die Seite 3 Berichte pro Besuch. Das Skript umgeht das nicht; mit eigenem (kostenlosem) Konto
als Secrets `GIPFELBUCH_EMAIL` / `GIPFELBUCH_PASSWORD` liest es alle neuen Berichte, sonst die 3 neusten.
Autor:innen werden nicht gespeichert. Die Texte sind deren Inhalte – Repo privat halten, nicht weitergeben.
Der Login-Ablauf ist ohne Konto noch ungetestet: wenn er scheitert, meldet das Skript «the site asks for a login».

### Unauffällig, aber anständig
`snowval/net.py`: ein normaler Browser-Kopf (pro Lauf einer), Cookies, Referer der Seite, keine Angaben zu dir,
zufällige Pausen (Seiten 2–8 s, Kacheln kürzer), Cache, Rückzug bei 429/5xx. Bewusst **nicht**: IP-/Proxy-Rotation,
Captcha-Lösen, Sperren umgehen, robots.txt ignorieren. Pro Lauf ~25 Anfragen an SLF, ~60 an snow-forecast.com,
und bei gipfelbuch.ch die Sitemaps plus eine pro neuem Bericht.

### Einrichten
- Secrets: `MODEL_SITE` (wie oben), optional `GIPFELBUCH_EMAIL`, `GIPFELBUCH_PASSWORD`.
- Variable (optional): `SLF_HS_URL`.
- Erster Lauf mit Rückblick: *Actions → Collect validation data → Run workflow*, `days_back` z. B. 30.

```
pip install -r requirements.txt
python scrape_snowmaps.py --data data --days-back 7 --cache .cache
python scrape_gipfelbuch.py --data data --since-days 7
MODEL_SITE=https://… python validate_obs.py --data data --out val
```

---

# Snowmapper-Validierungscockpit (täglich)

Workflow *Snowmapper validation cockpit* (täglich 08:53 UTC, nach der SLF-Neuschneekarte) baut eine HTML-Seite
und legt sie als Artefakt `snowmapper-cockpit-<datum>` ab (`index.html` öffnen). Die Kennzahlen stehen zusätzlich in
der Zusammenfassung des Laufs, der Verlauf auf dem Branch `cockpit` (`cockpit_history.csv`).

| Teil | Inhalt |
|---|---|
| Neuschnee | SNOW-Layer gegen SLF HN1D für jeden vergangenen Tag des Laufs: Karten, Höhe, Region, Schwellen (POD/FAR/CSI) |
| Kalibrierung | `calibrate_newsnow.py`: rechnet den Neuschnee mit anderer Regen/Schnee-Grenze, SLR, Geländefaktor und Niederschlagsfaktor neu; die beste Variante wird ohne den Prüftag gewählt |
| Schneedecke | Variant A gegen IMIS aus dem Manifest: Bias je Station, Drift, Niederschlagsfaktor an den Grenzen |
| Schneetyp | Variant A (ski6 + Wind) an der Koordinate jedes Gipfelbuch-Berichts gegen die gemeldete Oberfläche |
| Punkte | `validate_obs.py` über alle gesammelten Quellen (Branch `data`) |

Nötig: Secret `MODEL_SITE`. Für den Schneetyp-Teil zusätzlich der Workflow *Collect validation data*.
In einem öffentlichen Repo speichert dieser nur die Fakten aus den Gipfelbuch-Berichten, keine Texte (`--no-text`).

```
MODEL_SITE=https://… python calibrate_newsnow.py --out calib
MODEL_SITE=https://… python snowmapper_cockpit.py --data data --out cockpit
```

Plan für die nächsten Verbesserungen: [`docs/PLAN-neuschnee-schneetyp.md`](docs/PLAN-neuschnee-schneetyp.md)
