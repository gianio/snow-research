#!/usr/bin/env python3
"""Validate the model against every collected point observation.

Each observation (snowval/store.py) is matched to the nearest model cell
and compared where the model has the same quantity:

  hn24 / hn72            slf_hn maps, powderguide forecast       vs model new snow over the same window
  hs_gain24              derived here from IMIS HS (06 -> 06 UTC)  vs model new snow 24 h
                         (a lower bound: settling eats part of it)
  snow_cm_new            gipfelbuch "20 cm Neuschnee"           vs model new snow of the 72 h before the tour
  snow_period            snow-forecast.com forecast periods      vs model new snow over the period
  hs, hs_reported        IMIS, ExoLabs, resort reports, SLF map  vs model snow depth -- only if the model file
  snowline_m             gipfelbuch                              has a depth layer (--hs-key, default: auto)

Points whose height differs from the model cell by more than --max-dz m
are left out of the height-sensitive comparisons (stations, resorts);
the remaining height difference is reported.

    MODEL_SITE=https://… python validate_obs.py --data data --out out_val [--since 2026-11-01]
    python validate_obs.py --sm-file snowdata-123.json.gz --data data
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_newsnow import load_model, stats
from snowval import store

UTC = dt.timezone.utc


def ts(s):
    if not s:
        return None
    if len(s) == 10:
        return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


class Model:
    def __init__(self, sm, hs_key="auto", hs_scale=None):
        self.sm = sm
        self.t = np.array([t.timestamp() for t in sm["times"]])
        self.la, self.lo = sm["lat"], sm["lon"]
        self.dlat = abs(self.la[1] - self.la[0])
        self.dlon = abs(self.lo[1] - self.lo[0])
        self.cum = np.concatenate([np.zeros((1,) + sm["snow"].shape[1:], np.float32), np.cumsum(sm["snow"], 0)])
        self.hs = None
        b, m = sm["blob"], sm["meta"]
        keys = [hs_key] if hs_key != "auto" else [k for k in ("HS", "SD", "HSNOW", "DEPTH", "SNOWDEPTH") if k in b]
        for k in keys:
            if k in b:
                H, W = m["height"], m["width"]
                raw = np.frombuffer(base64.b64decode(b[k]), np.uint8)
                scale = hs_scale or m.get(k.lower() + "_scale") or m.get("hs_scale") or 1.0
                n = len(raw) // (H * W)
                self.hs = raw[: n * H * W].reshape(n, H, W).astype(np.float32) * scale
                self.hs_key, self.hs_T = k, n
                break
        print(f"model {sm['version']}: {len(self.t)} h, {len(self.la)}x{len(self.lo)} cells, "
              f"depth layer: {self.hs_key if self.hs is not None else 'none'}  (blob keys: {sorted(b)})")

    def cell(self, lat, lon):
        i = int(round((self.la[0] - lat) / self.dlat))
        j = int(round((lon - self.lo[0]) / self.dlon))
        if 0 <= i < len(self.la) and 0 <= j < len(self.lo):
            return i, j
        return None

    def hn(self, ij, t0, t1):
        """New snow t0..t1 (cm) or None if the model run does not cover it."""
        a = np.searchsorted(self.t, t0.timestamp())
        b = np.searchsorted(self.t, t1.timestamp())
        if b - a < (t1 - t0).total_seconds() / 3600 * .9:
            return None
        return float(self.cum[b, ij[0], ij[1]] - self.cum[a, ij[0], ij[1]])

    def depth(self, ij, t):
        if self.hs is None:
            return None
        k = int(np.argmin(np.abs(self.t[: self.hs_T] - t.timestamp())))
        if abs(self.t[k] - t.timestamp()) > 3 * 3600:
            return None
        return float(self.hs[k, ij[0], ij[1]])

    def snowline(self, lat, lon, km=3.0, thresh=5.0, t=None):
        """Lowest cell height within km that has >= thresh cm depth."""
        if self.hs is None:
            return None
        k = int(np.argmin(np.abs(self.t[: self.hs_T] - t.timestamp())))
        di = int(km / 111.32 / self.dlat) + 1
        dj = int(km / (111.32 * np.cos(np.radians(lat))) / self.dlon) + 1
        ij = self.cell(lat, lon)
        if not ij:
            return None
        sl = (slice(max(0, ij[0] - di), ij[0] + di + 1), slice(max(0, ij[1] - dj), ij[1] + dj + 1))
        e, h = self.sm["elev"][sl], self.hs[k][sl]
        e = e[h >= thresh]
        return float(e.min()) if e.size else None


def hs_gain(rows):
    """IMIS HS at 06 UTC -> positive 24 h change per station."""
    at6 = defaultdict(dict)
    meta = {}
    for r in rows:
        if r["valid_to"].endswith("T06:00:00Z") and isinstance(r["value"], float):
            at6[r["point_id"]][r["valid_to"][:10]] = r["value"]
            meta[r["point_id"]] = r
    out = []
    for sid, d in at6.items():
        for day, v in d.items():
            prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
            if prev in d:
                r = meta[sid]
                out.append({**r, "var": "hs_gain24", "value": max(0.0, v - d[prev]),
                            "valid_from": prev + "T06:00:00Z", "valid_to": day + "T06:00:00Z"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="validate_out")
    ap.add_argument("--site", default=os.environ.get("MODEL_SITE", ""))
    ap.add_argument("--sm-file")
    ap.add_argument("--since", help="only observations from this day (YYYY-MM-DD)")
    ap.add_argument("--max-dz", type=float, default=300, help="max height difference point vs model cell (m)")
    ap.add_argument("--hs-key", default="auto", help="name of the model's snow-depth layer in the blob")
    ap.add_argument("--hs-scale", type=float, help="cm per stored unit of that layer")
    a = ap.parse_args()
    if not (a.site or a.sm_file):
        raise SystemExit("set MODEL_SITE (or pass --site / --sm-file)")
    M = Model(load_model(a.site.rstrip("/") + "/" if a.site else a.site, a.sm_file), a.hs_key, a.hs_scale)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    obs = store.load(a.data, since=a.since)
    obs += hs_gain([r for r in obs if r["source"] == "imis" and r["var"] == "hs"])
    matched = []
    for r in obs:
        if r["lat"] is None or not isinstance(r["value"], float):
            continue
        ij = M.cell(r["lat"], r["lon"])
        if not ij:
            continue
        ez = float(M.sm["elev"][ij])
        dz = None if r["elev"] is None else r["elev"] - ez
        t0, t1 = ts(r["valid_from"]), ts(r["valid_to"])
        var, mv = r["var"], None
        height_sensitive = r["source"] in ("imis", "snowforecast")
        if height_sensitive and dz is not None and abs(dz) > a.max_dz:
            continue
        if var in ("hn24", "hn48", "hn72", "hs_gain24", "snow_period"):
            mv = M.hn(ij, t0, t1)
        elif var == "snow_cm_new":
            mv = M.hn(ij, t1 - dt.timedelta(hours=72), t1)
        elif var in ("hs", "hs_reported"):
            mv = M.depth(ij, t1)
        elif var == "snowline_m":
            mv = M.snowline(r["lat"], r["lon"], t=t1)
        if mv is None:
            continue
        matched.append({"source": r["source"], "var": var, "kind": r["kind"], "lead_h": r["lead_h"], "point_id": r["point_id"],
                        "lat": r["lat"], "lon": r["lon"], "elev": r["elev"], "cell_elev": ez, "dz": dz,
                        "valid_to": r["valid_to"], "obs": r["value"], "model": round(mv, 2)})

    with open(out / "matched.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matched[0]) if matched else ["source"])
        w.writeheader()
        w.writerows(matched)

    groups = defaultdict(list)
    for m in matched:
        groups[(m["source"], m["var"], m["kind"])].append(m)
    L = ["# Modell gegen Punktbeobachtungen", "",
         f"Erstellt {dt.datetime.now(UTC):%Y-%m-%d %H:%M} UTC · Modell {M.sm['version']} · "
         f"{len(matched)} Paare aus {len(obs)} Beobachtungen" + (f" seit {a.since}" if a.since else ""), "",
         "Bias > 0: Modell höher als die Quelle. Nur Werte im Zeitraum, den der Modelllauf abdeckt.", "",
         "| Quelle | Grösse | Art | n | Mittel Quelle | Mittel Modell | Bias | MAE | RMSE | r | POD ≥5 | FAR ≥5 | |Δz| Median |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for (src, var, kind), ms in sorted(groups.items()):
        o = np.array([m["obs"] for m in ms])
        p = np.array([m["model"] for m in ms])
        s = stats(o, p)
        summary[f"{src}/{var}/{kind}"] = s
        dz = [abs(m["dz"]) for m in ms if m["dz"] is not None]
        if s["n"] < 10:
            L.append(f"| {src} | {var} | {kind} | {s['n']} | {o.mean():.1f} | {p.mean():.1f} | {(p - o).mean():+.1f} | – | – | – | – | – | – |")
            continue
        d5 = s["detect"].get(5, {})
        f = lambda v, k=1: "–" if v is None else f"{v:.{k}f}"   # noqa: E731
        L.append(f"| {src} | {var} | {kind} | {s['n']} | {s['mean_ref']:.1f} | {s['mean_sm']:.1f} | {s['bias']:+.1f} | {s['mae']:.1f} | "
                 f"{s['rmse']:.1f} | {f(s['pearson'], 2)} | {f(d5.get('pod'), 2)} | {f(d5.get('far'), 2)} | "
                 f"{f(float(np.median(dz)) if dz else None, 0)} |")
    L += ["", "Paare einzeln: `matched.csv` · Kennzahlen: `summary.json`", ""]
    if M.hs is None:
        L += ["Die Modelldatei hat keine Schneehöhen-Ebene (gesucht: HS, SD, HSNOW, DEPTH) – Schneehöhe und "
              "Schneegrenze wurden nicht verglichen. Mit `--hs-key NAME --hs-scale CM_PRO_EINHEIT` angeben, falls vorhanden.", ""]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
