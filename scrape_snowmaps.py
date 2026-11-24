#!/usr/bin/env python3
"""Collect snow maps and forecasts -- past and future -- as point values.

Every source is read at the same points: a regular grid over Switzerland
(--grid-km, kept where the SLF map has data) plus every SLF IMIS station.
Values go to the observation store (snowval/store.py), one row per
source / variable / time / coordinate.

Sources (choose with --sources; default slf_hn,slf_hs,imis):
  slf_hn       SLF/WhiteRisk new-snow map HN1D (+HN3D), measured, past days
  slf_hs       SLF snow-depth map -- only if SLF_HS_URL is set (template with {date})
  imis         SLF IMIS stations: snow depth HS + air temperature, last 24 h (hourly)

    python scrape_snowmaps.py --data data [--days-back 3] [--grid-km 5]
        [--sources slf_hn,imis] [--cache .cache]

--days-back N re-reads the SLF maps of the last N days (backfill: e.g. 60).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from snowval import store
from snowval.net import Polite

UTC = dt.timezone.utc
ZURICH = ZoneInfo("Europe/Zurich")
R_EARTH = 6378137.0
CH_BBOX = (45.80, 5.90, 47.85, 10.55)       # lat0, lon0, lat1, lon1
SLF = "https://snow-maps-hs.slf.ch/public"
IMIS = "https://measurement-api.slf.ch/public/api/imis"
SLF_REF = "https://whiterisk.ch/"


def iso(t: dt.datetime | None):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def merc(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return np.radians(lon) * R_EARTH, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_EARTH


# ── points ──────────────────────────────────────────────────────────────────
def grid_points(km):
    la0, lo0, la1, lo1 = CH_BBOX
    dlat = km / 111.32
    dlon = km / (111.32 * math.cos(math.radians((la0 + la1) / 2)))
    lat = np.arange(la0 + dlat / 2, la1, dlat)
    lon = np.arange(lo0 + dlon / 2, lo1, dlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    LA, LO = LA.ravel(), LO.ravel()
    ids = [f"g{a:.3f}_{o:.3f}" for a, o in zip(LA, LO)]
    return {"lat": LA, "lon": LO, "elev": np.full(len(LA), np.nan), "id": ids}


def imis_stations(net):
    st = net.get(IMIS + "/stations", light=True).json()
    st = [s for s in st if s.get("lat") and s.get("lon")]
    return st


def all_points(net, km, mask_fn=None):
    g = grid_points(km)
    if mask_fn is not None:
        keep = mask_fn(g["lat"], g["lon"])
        g = {k: (np.asarray(v)[keep] if not isinstance(v, list) else [x for x, m in zip(v, keep) if m]) for k, v in g.items()}
    st = imis_stations(net)
    return {"lat": np.r_[g["lat"], [s["lat"] for s in st]],
            "lon": np.r_[g["lon"], [s["lon"] for s in st]],
            "elev": np.r_[g["elev"], [s.get("elevation") or np.nan for s in st]],
            "id": list(g["id"]) + [s["code"] for s in st]}, st


def rows_for(points, vals, **fixed):
    out = []
    for i, v in enumerate(vals):
        if v is None or not np.isfinite(v):
            continue
        e = points["elev"][i]
        out.append({**fixed, "lat": float(points["lat"][i]), "lon": float(points["lon"][i]),
                    "elev": None if not np.isfinite(e) else float(e), "value": round(float(v), 2),
                    "point_id": points["id"][i]})
    return out


# ── SLF value-encoded maps (as the WhiteRisk shader reads them) ──────────────
def slf_encoded(net, url, colorscale_url):
    r = net.get(url, light=True, referer=SLF_REF)
    ext = json.loads(r.headers["ch.slf.snow-maps.extent"])
    valid = dt.datetime.fromisoformat(r.headers["ch.slf.snow-maps.valid-at"].replace("Z", "+00:00"))
    img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.float32)
    cs = net.get(colorscale_url, light=True, referer=SLF_REF).json()
    lo, hi = cs["steps"][0]["start"], cs["steps"][-1]["start"]
    present = img[..., 2] >= 128
    val = np.where(img[..., 0] > 0, lo + img[..., 0] / 255 * (hi - lo), 0.0)
    return {"val": np.where(present, val, np.nan), "extent": ext, "valid": valid}


def slf_sample(m, lat, lon):
    x0, y0, x1, y1 = m["extent"]
    h, w = m["val"].shape
    X, Y = merc(lon, lat)
    c = np.floor((X - x0) / (x1 - x0) * w).astype(int)
    r = np.floor((y1 - Y) / (y1 - y0) * h).astype(int)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    out = np.full(len(lat), np.nan, np.float32)
    out[ok] = m["val"][r[ok], c[ok]]
    return out


def do_slf_hn(net, pts, data, days, periods=("HN1D", "HN3D")):
    n = 0
    for d in days:
        for per in periods:
            url = f"{SLF}/hn/map/{per}/{d}/encoded"
            try:
                m = slf_encoded(net, url, f"{SLF}/hn/map/colorscale")
            except Exception as e:                      # noqa: BLE001 -- a missing day is normal
                print(f"  slf_hn {per} {d}: {e}", file=sys.stderr)
                continue
            hours = {"HN1D": 24, "HN2D": 48, "HN3D": 72}[per]
            t1 = m["valid"]
            rows = rows_for(pts, slf_sample(m, pts["lat"], pts["lon"]), kind="obs", unit="cm",
                            valid_from=iso(t1 - dt.timedelta(hours=hours)), valid_to=iso(t1), ref=url)
            store.write(data, "slf_hn", per.lower().replace("hn1d", "hn24").replace("hn3d", "hn72").replace("hn2d", "hn48"),
                        d, rows)
            n += len(rows)
    return n


def do_slf_hs(net, pts, data, days):
    tpl = os.environ.get("SLF_HS_URL")
    if not tpl:
        print("  slf_hs: SLF_HS_URL not set (e.g. the snow-depth map's encoded image URL with {date}) -- skipped")
        return 0
    n = 0
    for d in days:
        try:
            m = slf_encoded(net, tpl.format(date=d), f"{SLF}/hs/map/colorscale")
        except Exception as e:                          # noqa: BLE001
            print(f"  slf_hs {d}: {e}", file=sys.stderr)
            continue
        rows = rows_for(pts, slf_sample(m, pts["lat"], pts["lon"]), kind="obs", unit="cm",
                        valid_from=iso(m["valid"]), valid_to=iso(m["valid"]), ref=tpl.format(date=d))
        store.write(data, "slf_hs", "hs", d, rows)
        n += len(rows)
    return n


# ── SLF IMIS stations ───────────────────────────────────────────────────────
def do_imis(net, stations, data):
    meas = net.get(IMIS + "/measurements", light=True, cache=False).json()
    st = {s["code"]: s for s in stations}
    by = {"hs": [], "ta": []}
    for m in meas:
        t = dt.datetime.fromisoformat(m["measure_date"].replace("Z", "+00:00"))
        if t.minute != 0:
            continue                                   # hourly is enough
        s = st.get(m["station_code"])
        if not s:
            continue
        base = {"kind": "obs", "lat": s["lat"], "lon": s["lon"], "elev": s.get("elevation"),
                "valid_from": iso(t), "valid_to": iso(t), "point_id": s["code"],
                "ref": f"{IMIS}/station/{s['code']}/measurements"}
        if m.get("HS") is not None:
            by["hs"].append({**base, "value": m["HS"], "unit": "cm"})
        if m.get("TA_30MIN_MEAN") is not None:
            by["ta"].append({**base, "value": m["TA_30MIN_MEAN"], "unit": "degC"})
    n = 0
    for var, rows in by.items():
        for day in sorted({r["valid_to"][:10] for r in rows}):
            # the API gives a rolling 24 h: merge with what is already stored for that day
            old = [r for r in store.load(data, "imis", var, since=day, until=day)]
            keyed = {(r["point_id"], r["valid_to"]): r for r in old}
            todays = [r for r in rows if r["valid_to"][:10] == day]
            keyed.update({(r["point_id"], r["valid_to"]): r for r in todays})
            store.write(data, "imis", var, day, keyed.values())
            n += len(todays)
    return n


# ── main ────────────────────────────────────────────────────────────────────
ALL = ["slf_hn", "slf_hs", "imis"]
DEFAULT = ["slf_hn", "slf_hs", "imis"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--sources", default=",".join(DEFAULT), help=f"comma list from {ALL}")
    ap.add_argument("--days-back", type=int, default=2, help="SLF maps: today and this many days before")
    ap.add_argument("--grid-km", type=float, default=5.0)
    ap.add_argument("--cache", default=None, help="folder for a local response cache")
    ap.add_argument("--delay", default="2,6", help="min,max seconds between page requests per host")
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.delay.split(","))
    net = Polite(delay=(lo, hi), cache_dir=a.cache)
    src = [s.strip() for s in a.sources.split(",") if s.strip()]
    today = dt.datetime.now(UTC).date()
    days = [(today - dt.timedelta(days=i)).isoformat() for i in range(a.days_back, -1, -1)]

    # Swiss mask from the newest SLF HN map that answers
    mask_fn = None
    for d in reversed(days):
        try:
            m = slf_encoded(net, f"{SLF}/hn/map/HN1D/{d}/encoded", f"{SLF}/hn/map/colorscale")
            mask_fn = lambda la, lo_, m=m: np.isfinite(slf_sample(m, la, lo_))  # noqa: E731
            break
        except Exception:                               # noqa: BLE001
            continue
    pts, stations = all_points(net, a.grid_km, mask_fn)
    print(f"{len(pts['id'])} points ({len(stations)} IMIS stations)")

    jobs = {
        "slf_hn": lambda: do_slf_hn(net, pts, a.data, days),
        "slf_hs": lambda: do_slf_hs(net, pts, a.data, days),
        "imis": lambda: do_imis(net, stations, a.data),
    }
    failed = []
    for s in src:
        try:
            print(f"{s}: {jobs[s]()} values")
        except Exception as e:                          # noqa: BLE001 -- one broken source must not stop the rest
            failed.append(s)
            print(f"{s}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    print(f"{net.n_requests} requests")
    if failed and len(failed) == len(src):
        sys.exit(1)


if __name__ == "__main__":
    main()
