# Plan: Neuschnee- und Schneetyp-Layer verbessern – mit Daten, die es schon gibt

Stand 19.11.2026, Modell-Lauf 202611181918. Die Zahlen stammen aus dem Cockpit
(`snowmapper_cockpit.py`) und aus `calibrate_newsnow.py` über 5 Frühwintertage:
genug, um die Richtung zu sehen, zu wenig für endgültige Parameter.

## Ausgangslage in Zahlen

| | Wert | Bedeutung |
|---|---|---|
| Neuschnee Modell / SLF HN1D (18.11.) | **2.7×** | der Layer zeigt fast dreimal so viel wie gemessen |
| Bias nach Höhe (18.11.) | +3 / +8 / +15 cm (1500–2000 / 2000–2500 / > 2500 m) | Fehler wächst mit der Höhe |
| Fehlalarme ≥ 10 cm / ≥ 20 cm | 55 % / 89 % | grosse Mengen werden viel zu oft gezeigt |
| Bester Faktor auf das heutige Modell | **0.34** | ein Drittel wäre im Mittel richtig |
| Beste Variante, am weggelassenen Tag geprüft | MAE 1.8 statt 5.1 cm | eine Kalibrierung bringt viel, auch ehrlich geprüft |
| Variant A: IMIS-Niederschlagsfaktor am Minimum 0.5 | **45 %** der Wetterpunkte | die IMIS-Korrektur stösst an ihre Grenze |
| Variant A vs IMIS Schneehöhe | +6 cm Bias, Drift WATCH (+36 % der Schneehöhe) | trotz Korrektur zu viel Schnee |

**Kernbefund.** Zwei voneinander unabhängige Prüfungen sagen dasselbe: Der
Neuschnee-Layer gegen die SLF-Karte und Variant A gegen die IMIS-Schneehöhen
bekommen zu viel Wasser. Variant A nutzt ICON-CH1/CH2 von MeteoSchweiz, der
App-Layer dieselben Wetterpunkte. Ein gemeinsamer Fehler im Niederschlags-Input
erklärt beides besser als jede Schneephysik.
Gegen die SLF-Karte allein lässt sich nur das Produkt aus Wasser und Lockerheit
bestimmen (Niederschlag × SLR); Schritt 1 trennt die beiden.

---

## A · Neuschnee-Layer

### A1 · Niederschlags-Input prüfen (zuerst, 1–2 Tage, grösster Hebel)

1. **Deakkumulation und Nahtstellen.** ICON-`TOT_PREC` ist seit Laufbeginn
   aufsummiert. Zu prüfen in `tools/ogd_extract.py`, `pipeline/ogd_site.py`,
   `pipeline/wx_archive.py`, `variant_a/forcing.py`:
   - wird je Stunde die Differenz zur Vorstunde genommen (nicht der Stand)?
   - zählen Stunden doppelt, wo CH1 → CH2 → D2 → seamless aneinandergesetzt werden,
     oder wo Archiv- und Prognosestunden überlappen?
   - Einheit kg m⁻² **pro Stunde**, nicht pro Ausgabeschritt (3 h / 6 h)?
2. **Gegen gemessenen Niederschlag.** SwissMetNet (MeteoSchweiz Open Data, ~160
   Stationen mit Niederschlag, stündlich): für die letzten 10 Tage `PREC` aus der
   Modelldatei an der Station gegen die Messung. Erwartung: Verhältnis 1.0–1.3
   (Messgeräte fangen Schnee 10–30 % zu wenig, das ist korrigierbar). Ein
   Verhältnis um 2–3 bestätigt einen Fehler aus Punkt 1.
3. Ergebnis als eigene Zeile ins Cockpit (Niederschlag Modell / Messung).

*Erwartung:* Liegt hier ein Faktor 2, sinkt das Modell/SLF-Verhältnis von 2.7 auf
~1.3, der Variant-A-Niederschlagsfaktor geht wieder Richtung 1.0, und die Drift
verschwindet. Kein Parameter muss dafür «verbogen» werden.

### A2 · Regen/Schnee-Grenze mit Luftfeuchte (Feuchttemperatur)

Heute: Schneeanteil = Sigmoid der Lufttemperatur um 1 °C (`model/factors.py:temperature_factor`).
In trockener Luft schneit es bis +2…3 °C, in feuchter schon bei +0.5 °C Regen.
- Relative Feuchte holen: Open-Meteo `relative_humidity_2m`, ICON-CH1 `RELHUM_2M`/`TD_2M` (OGD).
- Schneeanteil aus Temperatur **und** Feuchte (bivariate logistische Funktion nach
  Jennings et al. 2018, oder einfach Feuchttemperatur mit Schwelle ~0.5 °C).
- Dasselbe als `PSUM_PH`-Spalte in die SNOWPACK-Eingabe von Variant A
  (SNOWPACK liest den Phasenanteil direkt; heute entscheidet es selbst über die
  Lufttemperatur).
- Prüfung: Schneefallgrenze je Ereignis gegen die IMIS-Stationen in verschiedenen
  Höhen (bei welcher Höhe beginnt HS zu steigen?) und gegen Gipfelbuch-«Schnee ab … m».

### A3 · Neuschneedichte statt fester SLR-Gerade

Heute: SLR = 10 − 0.5·T, 6…15 (`config/model_params.yaml: slr`). In den Alpen
macht Wind kalten Neuschnee dichter. Kalibriert wird das Produkt mit A1 zusammen:
- Dichte als Funktion von Temperatur **und** Wind (wie in SNOWPACK, `HN_DENSITY`-Modelle),
  ρ ≈ 70–90 kg/m³ bei Windstille und Kälte, 120–200 kg/m³ bei > 8 m/s.
- Kalibrierung: `calibrate_newsnow.py` mit SLR-Parametern als Suchraum, *nachdem*
  A1 den Niederschlag festgelegt hat. Sonst tauschen sich die beiden aus.
- Bonus: Variant A rechnet die Neuschneedichte schon. Vergleich App-SLR ↔
  SNOWPACK-Dichte an den virtuellen Hängen zeigt die Lücke ohne neue Daten.

### A4 · Stationskorrektur für den App-Layer (wie Variant A sie hat)

Der App-Layer hat keine Korrektur durch Messungen. Vorschlag, ohne neue Quellen:
- Jeden Morgen das Verhältnis gemessen/Modell der letzten 24 h aus SLF HN1D und
  IMIS-HS-Zunahme bilden, log-Verhältnis räumlich glätten (Kriging oder IDW, 30 km),
  auf [0.5, 2] begrenzen.
- Auf die vergangenen Stunden voll anwenden (die Analyse wird stimmig), auf die
  Prognose mit abnehmendem Gewicht (z. B. halbe Wirkung nach 24 h, keine nach 72 h).
- Jede vierte Station bleibt draussen (Holdout), wie `variant_a/imis.py:is_holdout`.

### A5 · Feinere Niederschlagsverteilung

Variant A nutzt bereits das ICON-CH1-Niederschlagsfeld (1 km, `variant_a/precip.py`,
`precip_ch1.npz`). Der App-Layer interpoliert Wetterpunkte (0.03°) per IDW. Das
gleiche Feld auch für den App-Layer verwenden; der Höhen- und Luvfaktor
(`altitude_precip_factor`, `orographic_factor`) wird dann kleiner oder fällt weg,
weil ICON-CH1 die Orografie schon enthält. Die Kalibrierung zeigt schon heute:
Geländefaktor ganz weg oder voll an macht kaum Unterschied (Median 1.0).

### Ziele (im Cockpit sichtbar)

| Kennzahl | heute | Ziel |
|---|---|---|
| Modell / SLF | 2.7 | 0.8–1.2 |
| MAE 10-km-Blöcke | 7.8 cm | < 2 cm an Schneefalltagen |
| Bias nach Höhe | +3 … +15 cm | flach, < ±2 cm in jedem Band |
| CSI ≥ 10 cm | 0.44 | > 0.6 |
| Fehlalarme ≥ 20 cm | 89 % | < 40 % |

---

## B · Schneetyp-Layer (Variant A / SNOWPACK)

### B1 · Zuerst A1 und A2 übernehmen

Zu viel Wasser bedeutet zu viel und zu tiefer Pulver und zu wenige Krusten.
Sobald der IMIS-Faktor nicht mehr am Minimum klebt, stimmen Schneehöhe und
Schichtung besser, ohne dass `classify.py` angefasst wird.

### B2 · SNOWPACK an den IMIS-Stationen mit **gemessenem** Antrieb

Die wichtigste Diagnose, die es noch nicht gibt: SNOWPACK an ~30 IMIS-Stationen mit
deren eigenen Messungen (TA, RH, VW, ISWR/RSWR, HS → Niederschlag aus HS-Zunahme,
`ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE`). Das ist der operationelle Weg des SLF.
- Stimmen Schichten und Oberfläche mit gemessenem Antrieb, liegt der Fehler im
  Wetter-Input (→ A1–A3).
- Stimmen sie auch dann nicht, liegt er in der Klassierung (→ B4).
- Die IMIS-API liefert alles Nötige (siehe `scrape_snowmaps.py`, Quelle `imis`).

### B3 · Oberflächentemperatur als direkte Prüfung von Harsch und Sulz

IMIS misst die Schneeoberflächentemperatur (`TSS_30MIN_MEAN`) und die reflektierte
Strahlung (`RSWR`). Schmelz-Gefrier-Krusten entstehen, wenn TSS tagsüber 0 °C
erreicht und nachts deutlich darunter fällt. Täglich vergleichen:
- SNOWPACK-Oberflächentemperatur der flachen Läufe ↔ IMIS TSS (Bias, Tagesgang);
- Anteil der Tage mit «0 °C erreicht» (Sulz/Harsch möglich) ↔ Modellklasse.
Das prüft die Physik hinter `spring_corn`, `carrying_crust`, `breaking_crust`
direkt, ohne Personenberichte.

### B4 · Klassen mit Bodenwahrheit nachjustieren

Bodenwahrheit, die es schon gibt:
- Gipfelbuch-Berichte (`surface_top`/`surface_bottom`, Login nötig für mehr als 3/Tag);
- die Community-Berichte der App (`variant_a/community.py`, heute im Schattenmodus);
- das SLF-Lawinenbulletin (CAAML von aws.slf.ch): Lawinenprobleme mit Expositionen
  und Höhen – «Triebschnee» und «Nassschnee» sind direkte Hinweise für den Wind-
  und den Nässe-Layer.

Vorgehen:
1. Das Cockpit sammelt täglich die Verwechslungstabelle (gemeldet ↔ Modell).
2. Nach ~200 Berichten die Schwellen in `variant_a/classify.py` (`POWDER_DENSITY_MAX`,
   `CRUST_DENSITY_MIN`, `CRUST_HARD_MIN`, `WET_LWC_MIN`, `WIND_MIN`, `CRUST_GATE`)
   per Gittersuche auf ausgewogene Trefferquote optimieren, mit Holdout nach Datum.
3. Optional: ein kleiner Klassifikator (Gradient Boosting) auf .pro-Merkmalen
   (Dichte, Härte, Korntyp, LWC der obersten 10 cm, Stunden seit letztem Schneefall,
   max. TA 24 h, Strahlung je Exposition, Wind). Er dient als zweite Meinung und
   zeigt, welche Merkmale die Klassen trennen.

### B5 · Wind-Layer gegen Bulletin und Berichte

Täglich je Warnregion: Anteil `wind_slab`/`drift` in den Expositionen und Höhen,
die das Bulletin als Triebschnee-Problem nennt, gegen den Anteil ausserhalb. Ein
guter Wind-Layer zeigt dort deutlich mehr Triebschnee.

### B6 · Wenig/kein Schnee gegen Satellit

Copernicus HR-S&I (Sentinel-2, 20 m, frei mit Registrierung) oder MODIS MOD10A1:
Schnee ja/nein. Prüft `no_snow`/`thin_cover` und die Ausaperung nach Exposition,
gerade im Früh- und Spätwinter, wenn der Layer am meisten Fehler macht.

---

## Reihenfolge

| Woche | Schritt | Messbar im Cockpit an |
|---|---|---|
| 1 | A1 Niederschlag prüfen und reparieren | Modell/SLF, Variant-A-Faktor am Minimum, Drift |
| 1 | Gipfelbuch-Login als Secret, Sammel-Workflow aktiv | Anzahl Schneetyp-Berichte |
| 2 | A2 Feuchttemperatur (App + `PSUM_PH` für SNOWPACK) | Bias nach Höhe, Schneefallgrenze |
| 2–3 | A3 Dichte/SLR kalibrieren (`calibrate_newsnow.py`) | MAE 10 km am weggelassenen Tag |
| 3 | B2 SNOWPACK an IMIS mit gemessenem Antrieb | Schneehöhe und TSS an Stationen |
| 3–4 | A4 Stationskorrektur App-Layer, A5 ICON-CH1-Feld | MAE, CSI ≥ 10 cm |
| 4 | B3 Oberflächentemperatur TSS | Bias TSS, Krustentage |
| 5–6 | B4/B5 Klassen nachjustieren (genug Berichte) | Schneetyp-Trefferquote, Verwechslungstabelle |
| laufend | B6 Satellit für Schneebedeckung | Treffer Schnee ja/nein |

Jede Änderung zuerst im Schattenbetrieb: Das Cockpit rechnet die Variante parallel
(wie der Community-Schattenmodus), und sie geht erst live, wenn sie am
weggelassenen Tag besser ist.
