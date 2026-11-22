#!/usr/bin/env python3
"""Data files for the Snowmapper validation dashboard (Claude artifact).

Writes one JSON file per dataset, {rows, meta}, into --out. meta carries this
script's own text, so every number on the dashboard can be traced back.

  newsnow_days     new-snow layer vs SLF map HN1D / HN2D / HN3D, per day
  newsnow_elev     the same, per height band (all days pooled)
  calibration      what-if variants (calibrate_newsnow.py), pooled over the days
  calibration_loo  best variant chosen without the day it is scored on
  imis_hs          Variant A snow height vs IMIS measured, latest hour, per station
  imis_hn24        IMIS snow-height gain over the last 24 h vs the new-snow layer
  imis_daily       with --data: per day (06 UTC) bias / MAE of both, from the collected IMIS history
  va_status        Variant A drift / correction status from its manifest
  screenshots      screencheck results (WhiteRisk / SLF Pro screenshots)
  observer_hn      SLF Pro observed new snow vs model
  surface_obs      SLF Pro snow-surface symbols with Variant A classes
  coverage         which of the requested data types are in, and from where
  trend            the three steering numbers per day (new-snow ratio and 10 km error, holdout bias, surface hit rate)

    MODEL_SITE=https://… python dashboard_data.py --out dash_data --screencheck screencheck/results/2026-11-21 \
        [--data data --archive archive]      # collected IMIS history and archived runs: earlier days on the map
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "screencheck"))

import calibrate_newsnow as cal                                  # noqa: E402
from compare_newsnow import blocks, load_model, sample_slf, stats   # noqa: E402

UTC = dt.timezone.utc
# simple snow-surface comparison: three classes only
OBS3 = {"newsnow": "Pulver", "faceted": "Pulver", "wet": "Nass", "carrying": "Kruste", "breaking": "Kruste"}


def model3(ski6):
    if not ski6 or ski6 == "none":
        return None
    if ski6.startswith("powder") or ski6 == "loose_faceted":
        return "Pulver"
    return {"corn": "Nass", "wet": "Nass", "hard": "Kruste", "crust": "Kruste"}.get(ski6)
BANDS = [(0, 1500), (1500, 2000), (2000, 2500), (2500, 5000)]
PERIODS = {"HN1D": 24, "HN2D": 48, "HN3D": 72}


def holdout_bias(man):
    """Variant A manifest -> mean bias (cm) at holdout stations, their count, and at the assimilated ones."""
    st_ = (man.get("validation") or {}).get("stations") or []
    ho = [x["bias"] for x in st_ if x.get("holdout") and x.get("bias") is not None]
    us = [x["bias"] for x in st_ if not x.get("holdout") and x.get("bias") is not None]
    return (float(np.mean(ho)) if ho else None), len(ho), (float(np.mean(us)) if us else None)


def r(v, p=2):
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dash_data")
    ap.add_argument("--site", default=os.environ.get("MODEL_SITE", ""))
    ap.add_argument("--screencheck", nargs="*", help="screencheck results folders (results.json)")
    ap.add_argument("--archive", help="archive_forecast.py folder: earlier model runs, to show how the model changed")
    ap.add_argument("--data", help="collected observations (collect.yml, branch data): IMIS history for earlier days")
    ap.add_argument("--days", type=int, default=30, help="days of IMIS history on the map")
    a = ap.parse_args()
    if not a.site:
        sys.exit("set MODEL_SITE")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_text()
    site = a.site.rstrip("/") + "/"
    sm = load_model(site)
    lat, lon, elev = sm["lat"], sm["lon"], sm["elev"]
    ts = np.array([t.timestamp() for t in sm["times"]])
    cum = np.concatenate([np.zeros((1,) + sm["snow"].shape[1:], np.float32), np.cumsum(sm["snow"], 0)])
    now_i = sm["meta"].get("now_index", len(ts))
    last_past = sm["times"][now_i - 1]
    made = dt.datetime.now(UTC).isoformat(timespec="minutes")
    base_meta = [{"label": "dashboard_data.py", "value": code, "type": "code", "lang": "python"},
                 {"label": "Modell-Lauf", "value": sm["version"]}, {"label": "Erstellt", "value": made}]

    def save(name, rows, extra=None):
        (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": base_meta + (extra or [])}, ensure_ascii=False))
        print(f"{name}: {len(rows)} rows")

    # ── new snow vs SLF maps, every period and day the run covers in the past ──
    from scrape_snowmaps import slf_encoded, SLF
    from snowval.net import Polite
    net = Polite(delay=(.2, .5))
    days_rows, elev_acc = [], {p: {b: [[], []] for b in BANDS} for p in PERIODS}
    d = last_past.date()
    for k in range(8):
        day = d - dt.timedelta(days=k)
        for per, hours in PERIODS.items():
            try:
                m = slf_encoded(net, f"{SLF}/hn/map/{per}/{day.isoformat()}/encoded", f"{SLF}/hn/map/colorscale")
            except Exception:                                    # noqa: BLE001
                continue
            t1 = m["valid"]
            t0 = t1 - dt.timedelta(hours=hours)
            if t1 > last_past + dt.timedelta(hours=1):
                continue                                         # window ends in forecast hours
            i0, i1 = np.searchsorted(ts, t0.timestamp()), np.searchsorted(ts, t1.timestamp())
            if i1 - i0 < hours * 0.9:
                continue
            ref = sample_slf(m, lat, lon)
            mod = np.where(np.isfinite(ref), cum[i1] - cum[i0], np.nan)
            ok = np.isfinite(ref) & np.isfinite(mod)
            s = stats(ref[ok], mod[ok])
            rb, mb = blocks(np.where(ok, ref, np.nan), 10), blocks(np.where(ok, mod, np.nan), 10)
            okb = np.isfinite(rb) & np.isfinite(mb)
            det = s.get("detect", {})
            days_rows.append({"period": per, "day": day.isoformat(), "window_end": t1.isoformat(),
                              "n": s["n"], "slf_mean": r(s.get("mean_ref")), "model_mean": r(s.get("mean_sm")),
                              "bias": r(s.get("bias")), "mae": r(s.get("mae")), "mae_10km": r(np.abs(mb[okb] - rb[okb]).mean()),
                              "ratio": r(mod[ok].sum() / max(1e-6, ref[ok].sum()), 3), "pearson": r(s.get("pearson"), 3),
                              "csi5": r(det.get(5, {}).get("csi"), 3), "csi10": r(det.get(10, {}).get("csi"), 3),
                              "pod10": r(det.get(10, {}).get("pod"), 3), "far10": r(det.get(10, {}).get("far"), 3)})
            for b in BANDS:
                sel = ok & (elev >= b[0]) & (elev < b[1])
                elev_acc[per][b][0].append(ref[sel])
                elev_acc[per][b][1].append(mod[sel])
    days_rows.sort(key=lambda x: (x["period"], x["day"]))
    save("newsnow_days", days_rows, [{"label": "Vergleich", "value": "SLF-Neuschneekarte (snow-maps-hs.slf.ch, codiertes Bild) "
                                      "gegen die Summe des Neuschnee-Layers im selben Zeitraum, nur Zellen mit SLF-Daten"}])
    erows = []
    for per in PERIODS:
        for b in BANDS:
            rr = np.concatenate(elev_acc[per][b][0]) if elev_acc[per][b][0] else np.array([])
            mm = np.concatenate(elev_acc[per][b][1]) if elev_acc[per][b][1] else np.array([])
            if rr.size:
                erows.append({"period": per, "band": f"{b[0]}–{b[1]} m", "n": int(rr.size), "slf_mean": r(rr.mean()),
                              "model_mean": r(mm.mean()), "bias": r((mm - rr).mean())})
    save("newsnow_elev", erows)

    # ── calibration ──
    c = cal.run(sm, 5, log=lambda *x: None)
    pooled = []
    for name, (tc, slr_max, alpha, ps) in c["variants"].items():
        sc = [c["scores"][name][dd] for dd in c["days"] if c["scores"][name].get(dd)]
        if not sc:
            continue
        pooled.append({"variant": name, "rain_snow_c": tc, "slr_max": slr_max, "terrain_alpha": alpha, "precip_factor": ps,
                       "bias": r(np.mean([x["bias"] for x in sc])), "mae": r(np.mean([x["mae"] for x in sc])),
                       "mae_10km": r(np.mean([x["mae_10km"] for x in sc])), "ratio": r(np.mean([x["ratio"] for x in sc]), 3),
                       "current": name == "aktuell"})
    pooled.sort(key=lambda x: x["mae_10km"])
    keep = [x for x in pooled if x["current"]] + [x for x in pooled if not x["current"]][:12]
    save("calibration", keep, [{"label": "Tage", "value": ", ".join(c["days"])},
                               {"label": "Globaler Faktor", "value": f"{c['global_factor']:.3f}"}])
    save("calibration_loo", [{"day": x["day"], "picked": x["picked"], "mae_picked": r(x["held_out"]["mae"]),
                              "mae_current": r(x["current"]["mae"]), "bias_picked": r(x["held_out"]["bias"]),
                              "bias_current": r(x["current"]["bias"])}
                             for x in c["leave_one_day_out"] if x["held_out"] and x["current"]])

    # ── IMIS: snow height (Variant A) and 24 h gain (new-snow layer) ──
    from screencheck import Models
    M = Models(site, a.archive)
    meas = net.get("https://measurement-api.slf.ch/public/api/imis/measurements", light=True, cache=False).json()
    st = {s["code"]: s for s in M.stations if s.get("type", "SNOW_FLAT") == "SNOW_FLAT"}
    by = {}
    for x in meas:
        if x.get("HS") is None or x["station_code"] not in st:
            continue
        by.setdefault(x["station_code"], []).append((dt.datetime.fromisoformat(x["measure_date"].replace("Z", "+00:00")), float(x["HS"])))
    # history: what collect.yml stored (hourly HS per station), for one value per day at 06 UTC
    hist = {}
    if a.data:
        from snowval import store
        since = (last_past.date() - dt.timedelta(days=a.days + 1)).isoformat()
        for x in store.load(a.data, "imis", "hs", since=since):
            if x["point_id"] in st and isinstance(x["value"], float):
                hist.setdefault(x["point_id"], {})[dt.datetime.fromisoformat(x["valid_to"].replace("Z", "+00:00"))] = x["value"]
    for code_, series in by.items():
        hist.setdefault(code_, {}).update(dict(series))
    # Variant A nudges its precipitation with IMIS snow heights, except at every 4th station
    # (holdout): only those are an independent test of its snow height
    holdout = {x["station"]: bool(x.get("holdout")) for x in (M.man.get("validation") or {}).get("stations", [])}
    hs_rows, hn_rows = [], []
    for code_, series in by.items():
        series.sort()
        s = st[code_]
        t_last, hs_last = series[-1]
        mv = M.va_hs(s["lat"], s["lon"], s.get("elevation"), t_last)
        band = next((f"{b[0]}–{b[1]} m" for b in BANDS if b[0] <= (s.get("elevation") or 0) < b[1]), "")
        if mv is not None:
            hs_rows.append({"station": code_, "label": s["label"], "elev": s.get("elevation"), "band": band,
                            "lat": s["lat"], "lon": s["lon"], "time": t_last.isoformat(), "measured": r(hs_last, 1),
                            "model": r(mv, 1), "diff": r(mv - hs_last, 1),
                            "holdout": holdout.get(code_), "used_by_model": "nein (Holdout)" if holdout.get(code_) else
                            ("ja" if code_ in holdout else "unbekannt")})
        t_first, hs_first = series[0]
        if (t_last - t_first).total_seconds() >= 22 * 3600:
            mh = M.newsnow(s["lat"], s["lon"], t_first, t_last)
            if mh is not None:
                gain = max(0.0, hs_last - hs_first)
                hn_rows.append({"station": code_, "label": s["label"], "elev": s.get("elevation"), "band": band,
                                "from": t_first.isoformat(), "to": t_last.isoformat(), "measured_gain": r(gain, 1),
                                "model_newsnow": r(mh, 1), "diff": r(mh - gain, 1)})
    # one row per station and day: snow height at 06 UTC, and its gain over the 24 h before
    day_rows = []
    for code_, ser in hist.items():
        s = st[code_]
        band = next((f"{b[0]}–{b[1]} m" for b in BANDS if b[0] <= (s.get("elevation") or 0) < b[1]), "")
        for k in range(a.days):
            t6 = dt.datetime.combine(last_past.date() - dt.timedelta(days=k), dt.time(6), UTC)
            h1, h0 = ser.get(t6), ser.get(t6 - dt.timedelta(hours=24))
            if h1 is None:
                continue
            base = {"station": code_, "label": s["label"], "elev": s.get("elevation"), "band": band, "lat": s["lat"],
                    "lon": s["lon"], "day": t6.date().isoformat(), "holdout": holdout.get(code_)}
            mv = M.va_hs(s["lat"], s["lon"], s.get("elevation"), t6)
            if mv is not None:
                day_rows.append({**base, "layer": "hs", "measured": r(h1, 1), "model": r(mv, 1), "diff": r(mv - h1, 1)})
            if h0 is not None:
                mh = M.newsnow(s["lat"], s["lon"], t6 - dt.timedelta(hours=24), t6)
                if mh is not None:
                    g = max(0.0, h1 - h0)
                    day_rows.append({**base, "layer": "hn24", "measured": r(g, 1), "model": r(mh, 1), "diff": r(mh - g, 1)})
    # per day: how far off the model was, to see changes over the season
    summ = {}
    for x in day_rows:
        if x["layer"] == "hs" and not x["holdout"]:
            continue                                     # only holdout stations test Variant A's snow height
        summ.setdefault((x["day"], x["layer"]), []).append(x["diff"])
    save("imis_daily", [{"day": d_, "layer": l_, "n": len(v), "bias": r(float(np.mean(v)), 1), "mae": r(float(np.mean(np.abs(v))), 1)}
                        for (d_, l_), v in sorted(summ.items())],
         [{"label": "Werte", "value": "IMIS 06 UTC; hs nur Holdout-Stationen, hn24 = Zunahme der Schneehöhe in 24 h"}])
    save("imis_hs", sorted(hs_rows, key=lambda x: x["station"]),
         [{"label": "Modell", "value": "Variant A (SNOWPACK), flacher Hang, auf Stationshöhe; 3 nächste Wetterpunkte"},
          {"label": "Unabhängig", "value": "Nur Holdout-Stationen: Variant A korrigiert den Niederschlag mit den übrigen"}])
    save("imis_hn24", sorted(hn_rows, key=lambda x: x["station"]),
         [{"label": "Hinweis", "value": "Gemessene Zunahme der Schneehöhe = untere Grenze des Neuschnees (Setzung)"}])

    # ── Variant A status ──
    man = M.man
    dr = (man.get("drift") or {})
    pf = list((man.get("precip_factor") or {}).values())
    vi = (man.get("validation") or {}).get("imis") or {}
    save("va_status", [{"generated": man.get("generated"), "drift_all": (dr.get("all") or {}).get("status"),
                        "drift_independent": (dr.get("independent") or {}).get("status"),
                        "drift_bias_cm": r((dr.get("all") or {}).get("bias_cm"), 1),
                        "drift_rel_bias": r((dr.get("all") or {}).get("rel_bias"), 3),
                        "trend_cm_per_day": r((dr.get("all") or {}).get("trend_cm_per_day"), 2),
                        "imis_bias_cm": r(vi.get("bias_cm"), 1), "imis_mae_cm": r(vi.get("mae_cm"), 1),
                        "imis_stations": vi.get("stations"),
                        "pf_at_min_share": r(sum(1 for x in pf if x <= 0.5001) / len(pf), 3) if pf else None,
                        "pf_at_max_share": r(sum(1 for x in pf if x >= 1.999) / len(pf), 3) if pf else None,
                        "weather_points": len(pf)}])

    # ── screenshots (screencheck), every result set given ──
    shots, obs, surf = [], [], []
    superseded = {"IMG_4743_surface_observed.png", "IMG_4744_hn_observed.png"}   # same days again, with legend, in the doc set
    for d in a.screencheck or []:
        d = Path(d)
        if not (d / "results.json").exists():
            continue
        res = json.loads((d / "results.json").read_text())
        mf = HERE / "screencheck" / "inbox" / d.name / "manifest.json"
        meta_ = {x["file"]: x for x in json.loads(mf.read_text())} if mf.exists() else {}
        for f, v in res.items():
            it = meta_.get(f, {})
            s = v.get("summary") or {}
            rc = (v.get("reference_check") or {}).get("same")
            shots.append({"product": it.get("title") or f, "file": f, "set": d.name, "kind": it.get("kind"),
                          "n": s.get("n") or len(v.get("rows", [])),
                          "same_class": r(s.get("same_class"), 3), "within_one": r(s.get("within_one"), 3),
                          "model_higher": r(s.get("model_higher"), 3), "model_lower": r(s.get("model_lower"), 3),
                          "bias_mid_cm": r(s.get("bias_mid_cm"), 1), "reading_check": r(rc, 3),
                          "reliable": None if rc is None else rc >= 0.6})
            if f in superseded:
                continue
            day = (it.get("valid_local") or "")[:10]
            if it.get("kind") == "obs_points":
                leg = {"slf_obs_hn": [0, 5, 10, 20, 30, 40, 60, 80, 100], "wr_hn": [1, 10, 20, 30, 40, 60, 80, 100]}[it.get("legend", "wr_hn")]
                lab = lambda k, e=leg: "–" if k is None else ("< %g" % e[0] if k < 0 else (f"{e[k]}–{e[k + 1]}" if k + 1 < len(e) else f"≥{e[k]}"))  # noqa: E731
                for x in v["rows"]:
                    if x.get("model_bin") is None:
                        continue
                    obs.append({"key": f"{day}/{x['id']}", "day": day, "lat": r(x["lat"], 4), "lon": r(x["lon"], 4),
                                "observed_class": lab(x.get("shot_bin")), "model_cm": r(x.get("model"), 1),
                                "model_class": lab(x.get("model_bin")), "class_diff": x["model_bin"] - x["shot_bin"]})
            if it.get("kind") == "obs_symbols" and it.get("legend_roi"):
                for x in v["rows"]:
                    o3, m3 = OBS3.get(x["symbol"]), model3(x.get("va_ski6"))
                    if o3 is None:
                        continue                         # wind, surface hoar, other: not in the simple comparison
                    surf.append({"key": f"{day}/{x['id']}", "day": day, "symbol": x["symbol"], "observed": o3,
                                 "model": m3 or "keine Klasse", "lat": r(x["lat"], 4), "lon": r(x["lon"], 4),
                                 "va_ski6": x.get("va_ski6"), "hit": None if m3 is None else o3 == m3})
    save("screenshots", shots, [{"label": "Werkzeug", "value": "screencheck/screencheck.py (Farbstufen abgelesen, georeferenziert)"},
                                {"label": "Ablesen geprüft", "value": "Abgelesene Stufe gegen exakte SLF-Karte bzw. IMIS-Messung; unter 60 % unzuverlässig"}])
    save("observer_hn", obs, [{"label": "Legende", "value": "SLF Pro: 0–5, 5–10, 10–20, 20–30, 30–40, 40–60, 60–80, 80–100, >100 cm"}])
    save("surface_obs", surf, [{"label": "Vergleich", "value": "Drei Klassen. Beobachtung: Neuschnee/kantig = Pulver, feucht = Nass, "
                                "tragfähig/brüchig = Kruste (Wind, Reif, andere nicht gewertet). Modell (Variant A ski6): Pulver/lockerer "
                                "Altschnee = Pulver, Sulz/nass = Nass, hart/Kruste = Kruste; ohne Klasse = nicht vergleichbar."}])

    # ── one set of points for the map, and the same as GeoJSON for the app ──
    pts = []
    seen_days = {(x["layer"], x["day"]) for x in day_rows}
    for x in day_rows:
        pts.append({"layer": x["layer"], "day": x["day"], "id": x["station"], "label": f"{x['station']} {x['label']}",
                    "lat": x["lat"], "lon": x["lon"], "elev": x["elev"], "observed": x["measured"], "model": x["model"],
                    "diff": x["diff"], "unit": "cm", "hit": None,
                    "note": ("Holdout" if x["holdout"] else "zur Korrektur genutzt") if x["layer"] == "hs"
                    else "06→06 UTC, gemessen = Zunahme der Schneehöhe (untere Grenze)"})
    for x in hs_rows:
        if ("hs", x["time"][:10]) in seen_days:
            continue
        pts.append({"layer": "hs", "day": x["time"][:10], "id": x["station"], "label": f"{x['station']} {x['label']}", "lat": x["lat"],
                    "lon": x["lon"], "elev": x["elev"], "observed": x["measured"], "model": x["model"], "diff": x["diff"], "unit": "cm",
                    "hit": None, "note": "Holdout" if x["holdout"] else "zur Korrektur genutzt"})
    for x in hn_rows:
        if ("hn24", x["to"][:10]) in seen_days:
            continue
        s_ = st.get(x["station"], {})
        pts.append({"layer": "hn24", "day": x["to"][:10], "id": x["station"], "label": f"{x['station']} {x['label']}", "lat": s_.get("lat"),
                    "lon": s_.get("lon"), "elev": x["elev"], "observed": x["measured_gain"], "model": x["model_newsnow"], "diff": x["diff"],
                    "unit": "cm", "hit": None, "note": "gemessen = Zunahme der Schneehöhe (untere Grenze)"})
    for x in obs:
        pts.append({"layer": "obs_hn", "day": x["day"], "id": x["key"], "label": f"Beobachter {x['day']}", "lat": x["lat"], "lon": x["lon"],
                    "elev": None, "observed": x["observed_class"], "model": x["model_class"], "diff": x["class_diff"], "unit": "Stufen",
                    "hit": x["class_diff"] == 0, "note": f"Modell {x['model_cm']} cm"})
    for x in surf:
        pts.append({"layer": "surface", "day": x["day"], "id": x["key"], "label": f"Oberfläche {x['day']}", "lat": x["lat"], "lon": x["lon"],
                    "elev": None, "observed": x["observed"], "model": x["model"], "diff": None, "unit": "",
                    "hit": x["hit"], "note": f"Variant A ski6: {x['va_ski6']}"})
    save("map_points", [p_ for p_ in pts if p_["lat"] is not None])
    (out / "validation_points.geojson").write_text(json.dumps({"type": "FeatureCollection", "generated": made, "model_run": sm["version"],
        "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p_["lon"], p_["lat"]]},
                      "properties": {k: v for k, v in p_.items() if k not in ("lat", "lon")}} for p_ in pts if p_["lat"] is not None]},
        ensure_ascii=False))
    border = json.loads((HERE / "screencheck" / "ch_border.json").read_text())
    (out / "border.json").write_text(json.dumps({"type": "FeatureCollection", "attribution": "© swisstopo", "features": [
        {"type": "Feature", "properties": {"name": "Schweiz"}, "geometry": {"type": "MultiPolygon", "coordinates": [[r_] for r_ in border["rings"]]}}]}))

    # ── model versions: the same days, scored for each archived run and the live one ──
    ver_rows, va_rows = [], []
    live_hn1d = [x for x in days_rows if x["period"] == "HN1D"]
    for x in live_hn1d:
        ver_rows.append({"run": sm["version"], "generated": sm["generated"], "day": x["day"], "slf_mean": x["slf_mean"],
                         "model_mean": x["model_mean"], "bias": x["bias"], "mae_10km": x["mae_10km"], "ratio": x["ratio"]})
    if a.archive and Path(a.archive).is_dir():
        for f in sorted(Path(a.archive).glob("snowmapper_*.npz")):
            z = np.load(f)
            if "hourly" not in z.files or not bool(z["hourly"]):
                continue
            meta = json.loads(str(z["meta"]))
            gen = dt.datetime.fromisoformat(meta["generated"].replace("Z", "+00:00"))
            tts = np.array([dt.datetime.fromisoformat(t).timestamp() for t in z["times"]])
            snow = z["h_SNOW"].astype(np.float32) * meta["snow_scale"] * meta.get("ns_settle", 1.0)
            for x in live_hn1d + [{"day": (dt.date.fromisoformat(live_hn1d[0]["day"]) - dt.timedelta(days=k)).isoformat()} for k in (1, 2, 3)]:
                day = x["day"]
                try:
                    m = slf_encoded(net, f"{SLF}/hn/map/HN1D/{day}/encoded", f"{SLF}/hn/map/colorscale")
                except Exception:                            # noqa: BLE001
                    continue
                t1 = m["valid"]
                t0 = t1 - dt.timedelta(hours=24)
                if t1 > gen or t0.timestamp() < tts[0]:
                    continue                                 # only days this run had as past hours
                sel = (tts >= t0.timestamp()) & (tts < t1.timestamp())
                ref = sample_slf(m, z["lat"], z["lon"])
                mod = np.where(np.isfinite(ref), snow[sel].sum(0), np.nan)
                ok = np.isfinite(ref) & np.isfinite(mod)
                rb, mb = blocks(np.where(ok, ref, np.nan), 10), blocks(np.where(ok, mod, np.nan), 10)
                okb = np.isfinite(rb) & np.isfinite(mb)
                ver_rows.append({"run": meta["version"], "generated": meta["generated"], "day": day,
                                 "slf_mean": r(ref[ok].mean()), "model_mean": r(mod[ok].mean()), "bias": r((mod[ok] - ref[ok]).mean()),
                                 "mae_10km": r(np.abs(mb[okb] - rb[okb]).mean()), "ratio": r(mod[ok].sum() / max(1e-6, ref[ok].sum()), 3)})
        for f in sorted(Path(a.archive).glob("variant_a_*.npz")):
            man_ = json.loads(str(np.load(f)["manifest"]))
            dr_ = (man_.get("drift") or {}).get("all") or {}
            vi_ = (man_.get("validation") or {}).get("imis") or {}
            pf_ = list((man_.get("precip_factor") or {}).values())
            hb_, hn_, ub_ = holdout_bias(man_)
            va_rows.append({"generated": man_.get("generated"), "holdout_bias_cm": r(hb_, 1), "holdout_n": hn_,
                            "assimilated_bias_cm": r(ub_, 1), "drift": dr_.get("status"), "drift_bias_cm": r(dr_.get("bias_cm"), 1),
                            "drift_rel_bias": r(dr_.get("rel_bias"), 3), "imis_bias_cm": r(vi_.get("bias_cm"), 1),
                            "imis_mae_cm": r(vi_.get("mae_cm"), 1),
                            "pf_at_min_share": r(sum(1 for v_ in pf_ if v_ <= 0.5001) / len(pf_), 3) if pf_ else None})
    hb_, hn_, ub_ = holdout_bias(man)
    va_rows.append({"generated": man.get("generated"), "holdout_bias_cm": r(hb_, 1), "holdout_n": hn_,
                    "assimilated_bias_cm": r(ub_, 1), "drift": (dr.get("all") or {}).get("status"),
                    "drift_bias_cm": r((dr.get("all") or {}).get("bias_cm"), 1), "drift_rel_bias": r((dr.get("all") or {}).get("rel_bias"), 3),
                    "imis_bias_cm": r(vi.get("bias_cm"), 1), "imis_mae_cm": r(vi.get("mae_cm"), 1),
                    "pf_at_min_share": r(sum(1 for v_ in pf if v_ <= 0.5001) / len(pf), 3) if pf else None})
    ver_rows.sort(key=lambda x: (x["day"], x["generated"] or ""))
    save("model_versions", ver_rows, [{"label": "Vergleich", "value": "Jeder Lauf nur für Tage, die er als Vergangenheit kannte, gegen SLF HN1D"}])
    save("va_versions", sorted(va_rows, key=lambda x: x["generated"] or ""))

    # ── trend: the three numbers to steer by, one value per day ──
    # new snow: per day the first run that had the day as past (what the app showed the morning after)
    trend = []
    first = {}
    for x in ver_rows:
        if x["ratio"] is not None and (x["day"] not in first or (x["generated"] or "") < (first[x["day"]]["generated"] or "")):
            first[x["day"]] = x
    for x in days_rows:
        if x["period"] == "HN1D" and x["day"] not in first:
            first[x["day"]] = {**x, "run": sm["version"]}
    for day, x in sorted(first.items()):
        trend.append({"metric": "hn_ratio", "day": day, "value": x["ratio"], "run": x.get("run"), "lo": 0.9, "hi": 1.1})
        trend.append({"metric": "hn_mae10", "day": day, "value": x["mae_10km"], "run": x.get("run"), "lo": 0, "hi": 2})
    # snow height at holdout stations, per Variant A cycle (last one of a day)
    by_day = {}
    for x in sorted(va_rows, key=lambda x: x["generated"] or ""):
        if x.get("holdout_bias_cm") is not None and x.get("generated"):
            by_day[x["generated"][:10]] = x
    for day, x in sorted(by_day.items()):
        trend.append({"metric": "hs_holdout", "day": day, "value": x["holdout_bias_cm"], "run": x["generated"], "lo": -2, "hi": 2,
                      "n": x["holdout_n"], "other": x["assimilated_bias_cm"]})
    # snow surface: hit rate where the model has a class, and how often it has one
    for day in sorted({x["day"] for x in surf}):
        rows_ = [x for x in surf if x["day"] == day]
        cls = [x for x in rows_ if x["hit"] is not None]
        if cls:
            trend.append({"metric": "surf_hit", "day": day, "value": r(sum(x["hit"] for x in cls) / len(cls), 3), "n": len(cls),
                          "lo": 0.7, "hi": 1})
        trend.append({"metric": "surf_cover", "day": day, "value": r(len(cls) / len(rows_), 3), "n": len(rows_), "lo": 0.9, "hi": 1})
    save("trend", trend, [{"label": "Ziele", "value": "Neuschnee Modell ÷ SLF 0.9–1.1, Fehler 10 km < 2 cm; Holdout-Bias ±2 cm; "
                           "Oberfläche ≥ 70 % Treffer, ≥ 90 % mit Modellklasse"},
                          {"label": "Neuschnee", "value": "je Tag der erste Lauf, der den Tag als Vergangenheit hatte (Archiv), sonst der aktuelle"},
                          {"label": "Schneehöhe", "value": "Variant-A-Manifest: mittlerer Bias der Holdout-Stationen, letzter Zyklus des Tages"}])

    # ── coverage of the requested data types ──
    n = lambda rows, per=None: sum(1 for x in rows if per is None or x.get("period") == per)  # noqa: E731
    cov = [
        {"data": "Lokale Beobachter-Messungen", "status": "drin", "source": "SLF Pro Screenshots 19.–22.11.", "rows": len(obs),
         "next": "regelmässig neue Screenshots"},
        {"data": "Schneehöhe-Messung", "status": "drin", "source": "IMIS-Schnittstelle (Desktop-Screenshot: Ablesen unzuverlässig)", "rows": len(hs_rows),
         "next": "für manuelle Stationen grösser gezoomte Screenshots"},
        {"data": "Neuschnee 24 h", "status": "drin", "source": "SLF-Karte HN1D + IMIS-Zunahme + Screenshot", "rows": n(days_rows, "HN1D") + len(hn_rows),
         "next": "Beobachter-Messbretter über Screenshots"},
        {"data": "Neuschnee 48 h", "status": "drin", "source": "SLF-Karte HN2D", "rows": n(days_rows, "HN2D"), "next": "–"},
        {"data": "Neuschnee 72 h / 3 und 7 Tage", "status": "drin", "source": "SLF-Karte HN3D; Stationsbilder 3/7 Tage nur als Screenshot (unsicher)", "rows": n(days_rows, "HN3D"), "next": "–"},
        {"data": "Neuschneekarte 72 h", "status": "drin", "source": "SLF-Karte HN3D exakt + Screenshot 22.11. (Ablesen 82 % gleich)", "rows": n(days_rows, "HN3D"), "next": "–"},
        {"data": "Schneeoberfläche", "status": "drin", "source": "SLF Pro Screenshots 17.–22.11. mit Legende", "rows": len(surf),
         "next": "mehr Tage sammeln"},
    ]
    save("coverage", cov)


if __name__ == "__main__":
    main()
