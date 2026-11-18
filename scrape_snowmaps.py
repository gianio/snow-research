#!/usr/bin/env python3
"""Collect snow maps and forecasts -- past and future -- as point values.

Every source is read at the same points: a regular grid over Switzerland
(--grid-km, kept where the SLF map has data) plus every SLF IMIS station.
Values go to the observation store (snowval/store.py), one row per
source / variable / time / coordinate.

Sources (choose with --sources; default slf_hn,slf_hs,imis,snowforecast):
  slf_hn       SLF/WhiteRisk new-snow map HN1D (+HN3D), measured, past days
  slf_hs       SLF snow-depth map -- only if SLF_HS_URL is set (template with {date})
  imis         SLF IMIS stations: snow depth HS + air temperature, last 24 h (hourly)
  powderguide  PowderGuide ICON-D2 / ICON-EU forecast: new snow 24 h (DAY048..120)
               and snow depth (06 UTC of each day), newest run
  exolabs      PowderGuide's ExoLabs snow-depth layer (satellite + model analysis)
               -- powderguide / exolabs: their tile server's robots.txt currently
               disallows all robots, so these two stop with a PermissionError.
  snowforecast snow-forecast.com resorts in config/snowforecast_resorts.txt:
               forecast snowfall per period at top / mid / bottom and the
               reported top / bottom snow depth

    python scrape_snowmaps.py --data data [--days-back 3] [--grid-km 5]
        [--sources slf_hn,imis,powderguide] [--cache .cache]

--days-back N re-reads the SLF maps of the last N days (backfill: e.g. 60).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import math
import os
import re
import sys
from pathlib import Path
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
PG = "https://v2.tiles.powderguide.com/?p="
PG_REF = "https://map.powderguide.com/"
SF = "https://www.snow-forecast.com"
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


# ── PowderGuide colour tiles ─────────────────────────────────────────────────
_cbars = {}


def pg_cbar(net, var):
    if not _cbars:
        _cbars.update(net.get(PG + "cbars_en.json", light=True, referer=PG_REF).json())
    pts = [(c["r"], c["g"], c["b"], float(c["v"])) for c in _cbars[var] if isinstance(c.get("v"), (int, float))]
    pts = [p for p in pts if p[:3] != (0, 0, 0)]     # black = outside the scale at either end
    return np.array([p[:3] for p in pts], np.float32), np.array([p[3] for p in pts], np.float32)


def pg_sample(net, tile_path, var, lat, lon, z=7):
    cols, vals = pg_cbar(net, var)
    X, Y = merc(lon, lat)
    n = 2 ** z
    size = 2 * math.pi * R_EARTH
    fx = (X + size / 2) / size * n
    fy = (size / 2 - Y) / size * n
    out = np.full(len(lat), np.nan, np.float32)
    for tx, ty in sorted({(int(a), int(b)) for a, b in zip(fx, fy)}):
        url = PG + tile_path.format(z=z, x=tx, y=ty) + ".png"
        try:
            im = np.asarray(Image.open(io.BytesIO(net.get(url, light=True, referer=PG_REF).content)).convert("RGBA")).astype(np.float32)
        except Exception:                               # noqa: BLE001 -- tile outside the domain
            continue
        sel = (fx.astype(int) == tx) & (fy.astype(int) == ty)
        px = np.clip(((fx[sel] - tx) * 256).astype(int), 0, 255)
        py = np.clip(((fy[sel] - ty) * 256).astype(int), 0, 255)
        rgba = im[py, px]
        v = np.zeros(len(px), np.float32)               # transparent = below the scale
        a = rgba[:, 3] > 127
        if a.any():
            d = ((rgba[a, None, :3] - cols[None]) ** 2).sum(-1)
            k = d.argmin(1)
            vv = vals[k].copy()
            vv[np.sqrt(d[np.arange(len(k)), k]) > 40] = np.nan   # border / label pixels
            v[a] = vv
        out[sel] = v
    return out


def pg_plots(net, var):
    return net.get(PG + f"plots_{var}.json", light=True, referer=PG_REF, cache=False).json()["plots"]


def pg_pick(plots, model="icon-eu"):
    e = [p for p in plots if p["region"] == "eur" and p.get("maskStr", "nomask") == "nomask" and p["model"] == model]
    if not e:
        return None, []
    run = max(p["date"] for p in e)
    return run, [p for p in e if p["date"] == run]


def do_powderguide(net, pts, data, z):
    n = 0
    # new snow, 24 h sums: DAY048 = the 24 h ending run + 48 h
    run, plots = pg_pick(pg_plots(net, "snow_gsp__lsfwe"))
    r0 = dt.datetime.strptime(run, "%Y%m%d").replace(tzinfo=UTC)
    rows = []
    for day in sorted({p["day"] for p in plots if re.fullmatch(r"DAY\d{3}", p["day"])}):
        p = next(p for p in plots if p["day"] == day)
        lead = int(day[3:])
        src = p.get("patches") or []
        path, model = (src[0]["tileUrlPath"], src[0]["model"]) if src else (p["tileUrlPath"], p["model"])
        v = pg_sample(net, path, "snow_gsp__lsfwe", pts["lat"], pts["lon"], z)
        t1 = r0 + dt.timedelta(hours=lead)
        rows += rows_for(pts, v, kind="forecast", unit="cm", issued=iso(r0), lead_h=lead,
                         valid_from=iso(t1 - dt.timedelta(hours=24)), valid_to=iso(t1),
                         ref=f"powderguide {model} snow_gsp__lsfwe {run} {day} (1 mm = 1 cm)")
    store.write(data, "powderguide", "hn24", r0.date().isoformat(), rows)
    n += len(rows)
    # snow depth, 06 UTC of each day (+ the analysis at 00)
    run, plots = pg_pick(pg_plots(net, "h_snow__sde"))
    r0 = dt.datetime.strptime(run, "%Y%m%d").replace(tzinfo=UTC)
    rows = []
    for p in sorted(plots, key=lambda p: p["day"]):
        m = re.fullmatch(r"E?DAY(\d{3})", p["day"])
        if not m:
            continue
        lead = int(m.group(1))
        if lead != 0 and lead % 24 != 6:
            continue
        src = p.get("patches") or []
        path, model = (src[0]["tileUrlPath"], src[0]["model"]) if src else (p["tileUrlPath"], p["model"])
        v = pg_sample(net, path, "h_snow__sde", pts["lat"], pts["lon"], z) * 100   # m -> cm
        t = r0 + dt.timedelta(hours=lead)
        rows += rows_for(pts, v, kind="forecast" if lead else "analysis", unit="cm", issued=iso(r0), lead_h=lead,
                         valid_from=iso(t), valid_to=iso(t), ref=f"powderguide {model} h_snow__sde {run} {p['day']}")
    store.write(data, "powderguide", "hs", r0.date().isoformat(), rows)
    return n + len(rows)


def do_exolabs(net, pts, data, z):
    plots = [p for p in pg_plots(net, "h_snow__sde") if p["model"] == "exolabs"]
    if not plots:
        print("  exolabs: no layer offered today")
        return 0
    run = max(p["date"] for p in plots)
    p = next(p for p in plots if p["date"] == run)
    lead = int(re.sub(r"\D", "", p["day"]) or 0)
    t = dt.datetime.strptime(run, "%Y%m%d").replace(tzinfo=UTC) + dt.timedelta(hours=lead)
    src = p.get("patches") or []
    path = src[0]["tileUrlPath"] if src else p["tileUrlPath"]
    v = pg_sample(net, path, "h_snow__sde", pts["lat"], pts["lon"], z) * 100
    rows = rows_for(pts, v, kind="analysis", unit="cm", valid_from=iso(t), valid_to=iso(t),
                    ref=f"powderguide exolabs h_snow__sde {run} {p['day']}")
    store.write(data, "exolabs", "hs", run[:4] + "-" + run[4:6] + "-" + run[6:], rows)
    return len(rows)


# ── snow-forecast.com ───────────────────────────────────────────────────────
def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def sf_page(net, resort, level):
    url = f"{SF}/resorts/{resort}/6day/{level}"
    s = net.get(url, referer=f"{SF}/resorts/{resort}", cache=False).text
    out = {"url": url}
    m = re.search(r'"latitude":\s*([\d.]+)', s), re.search(r'"longitude":\s*([\d.]+)', s)
    out["lat"], out["lon"] = (_num(m[0].group(1)), _num(m[1].group(1))) if all(m) else (None, None)
    out["elev"] = {lv: _num(h) for lv, h in re.findall(r'data-forecast-elevation="(\w+)"[^>]*>\s*<span class="height">(\d+)', s)}
    iss = re.search(r'location-subheader-issued__value"><b>([^<]+)</b>', s)
    out["issued"] = None
    if iss:
        txt = html.unescape(iss.group(1)).replace(" ", " ").strip()
        for fmt in ("%I %p %d %b %Y", "%I:%M %p %d %b %Y"):
            try:
                # issue time is local (Europe/Zurich)
                out["issued"] = dt.datetime.strptime(txt.replace("pm", "PM").replace("am", "AM"), fmt)
                break
            except ValueError:
                pass
    # column dates: the day row spans 3 cells each; time row gives the hour of each cell
    days = re.findall(r'<td class="forecast-table-days__cell[^"]*" colspan="(\d+)"[^>]*data-date="([\d-]+)"', s)
    cols = [d for span, d in days for _ in range(int(span))]
    trow = s[s.find('data-row="time"'):]
    trow = trow[:trow.find("</tr>")]
    hours = [int(h) for h in re.findall(r'data-hour="(\d+)"', trow)]
    srow = s[s.find('data-row="snow"'):]
    srow = srow[:srow.find("</tr>")]
    cells = re.findall(r'<td class="forecast-table__cell[^"]*">(.*?)</td>', srow, re.S)
    snow = [_num((re.search(r'data-value="([\d.]+)"', c) or [None, None])[1]) for c in cells]
    out["snow"] = [(cols[i], hours[i], snow[i]) for i in range(min(len(cols), len(hours), len(snow)))]
    dep = re.search(r'snow-depths-table__table(.*?)</table>', s, re.S)
    out["depth"] = {}
    if dep:
        for lab, val in re.findall(r'<th>([^<]+)</th><td>(.*?)</td>', dep.group(1)):
            v = re.search(r'class="snowht">([\d.]+)<', val)
            out["depth"][lab.strip(" :")] = _num(v.group(1)) if v else None
    return out


def do_snowforecast(net, data, resorts, today):
    fc, dep = [], []
    for res in resorts:
        for level in ("top", "mid", "bot"):
            try:
                p = sf_page(net, res, level)
            except Exception as e:                      # noqa: BLE001
                print(f"  snowforecast {res}/{level}: {e}", file=sys.stderr)
                continue
            if p["lat"] is None:
                continue
            elev = p["elev"].get(level)
            issued = iso(p["issued"].replace(tzinfo=ZURICH)) if p["issued"] else None
            for d, h, v in p["snow"]:
                if v is None:
                    continue
                t = dt.datetime.fromisoformat(d).replace(hour=h, tzinfo=ZURICH)
                # each column = the ~6 h around its label hour (AM ~07-13, PM ~13-19, night ~19-07 local)
                fc.append({"kind": "forecast", "lat": p["lat"], "lon": p["lon"], "elev": elev, "unit": "cm",
                           "valid_from": iso(t - dt.timedelta(hours=3)), "valid_to": iso(t + dt.timedelta(hours=3)),
                           "issued": issued, "value": v, "point_id": f"{res}/{level}", "ref": p["url"]})
            if level == "mid":
                for lab, lv in (("Top snow depth", "top"), ("Bottom snow depth", "bot")):
                    v = p["depth"].get(lab)
                    if v is not None:
                        dep.append({"kind": "obs", "lat": p["lat"], "lon": p["lon"], "elev": p["elev"].get(lv),
                                    "unit": "cm", "valid_from": today, "valid_to": today, "value": v,
                                    "point_id": f"{res}/{lv}", "ref": p["url"] + " (resort report)"})
    store.write(data, "snowforecast", "snow_period", today, fc)
    store.write(data, "snowforecast", "hs_reported", today, dep)
    return len(fc) + len(dep)


# ── main ────────────────────────────────────────────────────────────────────
ALL = ["slf_hn", "slf_hs", "imis", "powderguide", "exolabs", "snowforecast"]
# PowderGuide's tile server says "Disallow: /" in robots.txt, so powderguide and
# exolabs are not run by default (and refuse to run while that stays so)
DEFAULT = ["slf_hn", "slf_hs", "imis", "snowforecast"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--sources", default=",".join(DEFAULT), help=f"comma list from {ALL}")
    ap.add_argument("--days-back", type=int, default=2, help="SLF maps: today and this many days before")
    ap.add_argument("--grid-km", type=float, default=5.0)
    ap.add_argument("--zoom", type=int, default=7, help="PowderGuide tile zoom (7 ≈ 0.9 km/pixel here)")
    ap.add_argument("--resorts", default="config/snowforecast_resorts.txt")
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
        "powderguide": lambda: do_powderguide(net, pts, a.data, a.zoom),
        "exolabs": lambda: do_exolabs(net, pts, a.data, a.zoom),
        "snowforecast": lambda: do_snowforecast(
            net, a.data, [r.strip() for r in Path(a.resorts).read_text().splitlines() if r.strip() and not r.startswith("#")],
            today.isoformat()),
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
