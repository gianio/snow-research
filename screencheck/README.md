# Screenshots gegen das Modell prüfen

Karten-Screenshots von WhiteRisk oder SLF Pro werden georeferenziert, mit der Farbskala der Karte abgelesen und mit
Snowmapper verglichen:

| Grösse | Modell |
|---|---|
| Neuschnee | App-Layer `SNOW` über dieselben 24 h |
| Schneehöhe | Variant A (SNOWPACK), flacher Hang, auf der Höhe der Station bzw. Zelle |
| Schneeoberfläche | Variant A `ski6` / `ski18` / `wind` |

## Ablauf

1. **Bilder ablegen:** pro Satz ein Ordner unter `screencheck/inbox/<satz>/`, z. B. `screencheck/inbox/2026-12-14/`.
   Dateinamen mit diesen Wörtern füllen das Manifest gleich richtig vor:
   `hn24_stations`, `hs_stations`, `hn24_map`, `hs_map`, `hn_observed`, `surface`.
2. **Manifest anlegen:** `python screencheck/screencheck.py init <satz>`.
   Das Werkzeug sucht die Ortsmarker (kleine Ringe) und trägt sie ein. Du ergänzt in `manifest.json`:
   - die Ortsnamen zu den gefundenen Ringen; bekannt sind die Orte in `CITIES` in `screencheck.py`, weitere lassen sich ergänzen;
   - Datum/Zeit (`valid_local`): Schweizer Zeit, mit `Z` am Ende für UTC. Die WhiteRisk-Flächenkarten zeigen UTC.
3. **Prüfen:** `MODEL_SITE=… python screencheck/screencheck.py run <satz> [--archive archive]`.
   Das Ergebnis liegt in `screencheck/results/<satz>/`: `report.md`, `results.json` und je Bild ein Kontrollbild.
   **Zuerst die Kontrollbilder ansehen:** Die magentafarbene Landesgrenze muss auf der Grenze der Karte liegen.

Beide Ordner sind in Git ausgeschlossen: Die Screenshots zeigen SLF-Produkte, teils aus einem persönlichen SLF-Pro-Konto.

## Georeferenz

- **Orte** (`"points": [[x, y, "Basel"], …]`, mindestens 3): Jeder Punkt wird automatisch auf den Ortsmarker geschoben.
  Ohne Marker gibst du die Orte von Auge an, mit `"refine": false`.
- **Landesgrenze** (`"border"`): für die SLF-Pro-Karten in Schweizer Projektion, ohne Orte.
- **Feinjustierung:** Danach wird die offizielle Grenze (swisstopo) auf die gezeichnete Grenze eingepasst.
  Bei SLF Pro geschieht das immer, bei anderen Karten mit `"fit_border": {"gray": [70, 140], "max_sat": 22}`.
  Der Bericht nennt den mittleren Abstand vorher und nachher.

## Kartenarten (`kind`)

| kind | was | Hinweis |
|---|---|---|
| `stations` | WhiteRisk-Stationskreise | gelesen an jeder IMIS-Schneestation; graue Quadrate = kein Messwert, werden übersprungen |
| `raster` | WhiteRisk-Flächenkarte | jede 1-km-Zelle in der Schweiz; `"reference": "slf_hn1d"` prüft das Ablesen gegen die exakten SLF-Werte |
| `obs_points` | SLF Pro «Neuschnee beobachtet» | farbige Kreise nach der Legendenfarbe, graue Ringe = 0 |
| `obs_symbols` | SLF Pro «Schneeoberfläche» | Symbole nach Farbe erkannt; Bedeutung in `symbol_legend`, z. B. `"cyan": {"name": "Pulver", "coarse": "powder"}` |

Höher aufgelöste Bilder gehen genauso: Marker- und Symbolgrössen wachsen mit der Bildgrösse.

## Was die Zahlen bedeuten

Die Karten zeigen Farbstufen, keine Zahlen. Deshalb kommt das Modell in dieselben Stufen.
«gleiche Stufe» und «±1 Stufe» sind die fairen Masse, der Bias wird aus den Stufenmitten gerechnet.

Liegt ein Bild ausserhalb des Zeitraums des aktuellen Modelllaufs (heute −5 Tage), braucht es das Prognose-Archiv
(`--archive`, Workflow *Archive forecasts*). Ohne Archiv lassen sich ältere Bilder nicht prüfen.
