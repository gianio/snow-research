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
  va_status        Variant A drift / correction status from its manifest
  screenshots      screencheck results (WhiteRisk / SLF Pro screenshots)
  observer_hn      SLF Pro observed new snow vs model
  surface_obs      SLF Pro snow-surface symbols with Variant A classes
  coverage         which of the requested data types are in, and from where

    MODEL_SITE=https://… python dashboard_data.py --out dash_data --screencheck screencheck/results/2026-11-21
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
BANDS = [(0, 1500), (1500, 2000), (2000, 2500), (2500, 5000)]
PERIODS = {"HN1D": 24, "HN2D": 48, "HN3D": 72}


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
    M = Models(site)
    meas = net.get("https://measurement-api.slf.ch/public/api/imis/measurements", light=True, cache=False).json()
    st = {s["code"]: s for s in M.stations if s.get("type", "SNOW_FLAT") == "SNOW_FLAT"}
    by = {}
    for x in meas:
        if x.get("HS") is None or x["station_code"] not in st:
            continue
        by.setdefault(x["station_code"], []).append((dt.datetime.fromisoformat(x["measure_date"].replace("Z", "+00:00")), float(x["HS"])))
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
                    surf.append({"key": f"{day}/{x['id']}", "day": day, "symbol": x["symbol"], "observed": x.get("reported"),
                                 "lat": r(x["lat"], 4), "lon": r(x["lon"], 4), "va_ski6": x.get("va_ski6"),
                                 "va_ski18": x.get("va_ski18"), "va_wind": x.get("va_wind"), "hit": x.get("hit")})
    save("screenshots", shots, [{"label": "Werkzeug", "value": "screencheck/screencheck.py (Farbstufen abgelesen, georeferenziert)"},
                                {"label": "Ablesen geprüft", "value": "Abgelesene Stufe gegen exakte SLF-Karte bzw. IMIS-Messung; unter 60 % unzuverlässig"}])
    save("observer_hn", obs, [{"label": "Legende", "value": "SLF Pro: 0–5, 5–10, 10–20, 20–30, 30–40, 40–60, 60–80, 80–100, >100 cm"}])
    save("surface_obs", surf, [{"label": "Treffer", "value": "Neuschnee/kantig → Pulver; feucht → Sulz/nass; tragfähig/brüchig → Kruste/hart; "
                                "Wind → Windlayer oder ski18-Wind; Oberflächenreif → ski18 surface_hoar"}])

    # ── one set of points for the map, and the same as GeoJSON for the app ──
    pts = []
    for x in hs_rows:
        pts.append({"layer": "hs", "day": x["time"][:10], "id": x["station"], "label": f"{x['station']} {x['label']}", "lat": x["lat"],
                    "lon": x["lon"], "elev": x["elev"], "observed": x["measured"], "model": x["model"], "diff": x["diff"], "unit": "cm",
                    "hit": None, "note": "Holdout" if x["holdout"] else "zur Korrektur genutzt"})
    for x in hn_rows:
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
                    "elev": None, "observed": x["symbol"], "model": f"{x['va_ski6']} / {x['va_ski18']}", "diff": None, "unit": "",
                    "hit": x["hit"], "note": f"Wind: {x['va_wind']}"})
    save("map_points", [p_ for p_ in pts if p_["lat"] is not None])
    (out / "validation_points.geojson").write_text(json.dumps({"type": "FeatureCollection", "generated": made, "model_run": sm["version"],
        "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p_["lon"], p_["lat"]]},
                      "properties": {k: v for k, v in p_.items() if k not in ("lat", "lon")}} for p_ in pts if p_["lat"] is not None]},
        ensure_ascii=False))
    border = json.loads((HERE / "screencheck" / "ch_border.json").read_text())
    (out / "border.json").write_text(json.dumps({"type": "FeatureCollection", "attribution": "© swisstopo", "features": [
        {"type": "Feature", "properties": {"name": "Schweiz"}, "geometry": {"type": "MultiPolygon", "coordinates": [[r_] for r_ in border["rings"]]}}]}))

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
