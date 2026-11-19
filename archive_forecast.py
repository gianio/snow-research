#!/usr/bin/env python3
"""Archive every Snowmapper forecast run in a compact form.

The site only ever serves the newest run (~50 MB), so past forecasts are
lost. This keeps, per run, what verification and training need:

  snowmapper_<version>.npz   (1 km grid of the run, ~264 h)
    daily   per 24 h window 06->06 UTC: new snow (settled), precipitation,
            ablation, T mean/min/max, precipitation-weighted T, wind mean/max,
            sunshine, cloud -- int16 in tenths (value = d_x / 10, -32768 = none)
    hourly  only for the first run of each UTC day ("day run"): SNOW, PREC, ABL
            uint8 exactly as in the blob, TEMP every 3 h
    static  elevation, lat, lon, times, window ends, meta (scales, nsp, settle)

  variant_a_<generated>.npz  (Variant A, once per cycle)
    class rasters (uint8, legend index) of ski6, ski18, wind, powder at
    12 UTC of every day in the cycle + the manifest

Files go to --out; the workflow uploads them as release assets
(one release per month, tag fc-archive-YYYY-MM) and keeps an index.

    MODEL_SITE=https://… python archive_forecast.py --out archive --index archive/index.json
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from compare_newsnow import load_model
from snowval.net import Polite

UTC = dt.timezone.utc
HOURLY = ["SNOW", "PREC", "TEMP", "ABL"]


def cube(m, b, key):
    T, H, W = m["T"], m["height"], m["width"]
    return np.frombuffer(base64.b64decode(b[key]), np.uint8)[:T * H * W].reshape(T, H, W)


def q10(a):
    a = np.asarray(a, np.float32) * 10
    return np.where(np.isfinite(a), np.clip(np.round(a), -32767, 32767), -32768).astype(np.int16)


def archive_model(sm, out: Path, hourly=True):
    m, b = sm["meta"], sm["blob"]
    times = sm["times"]
    raw = {k: cube(m, b, k) for k in HOURLY + ["SUN", "WINDG", "CLOUD"] if k in b}
    snow = raw["SNOW"].astype(np.float32) * m["snow_scale"] * m.get("ns_settle", 1.0)
    prec = raw["PREC"].astype(np.float32) / m.get("prec_mul", 5.0)
    temp = raw["TEMP"].astype(np.float32) / m.get("temp_mul", 2.0) - m.get("temp_off", 60.0)
    abl = raw["ABL"].astype(np.float32) * m.get("abl_scale", 0.05)
    sun = raw["SUN"].astype(np.float32) / m.get("sun_mul", 100.0) if "SUN" in raw else None
    wind = raw["WINDG"].astype(np.float32) / m.get("spd_mul", 5.0) if "WINDG" in raw else None
    cloud = raw["CLOUD"].astype(np.float32) / m.get("cloud_mul", 2.5) if "CLOUD" in raw else None
    # 24 h windows ending 06 UTC, fully inside the run
    ends = []
    d = times[0].replace(hour=6, minute=0) + dt.timedelta(days=1)
    while d <= times[-1] + dt.timedelta(hours=1):
        ends.append(d)
        d += dt.timedelta(days=1)
    ts = np.array([t.timestamp() for t in times])
    daily = {k: [] for k in ("snow", "prec", "abl", "t_mean", "t_min", "t_max", "t_prec", "wind_mean", "wind_max", "sun", "cloud")}
    kept = []
    for e in ends:
        sel = (ts >= (e - dt.timedelta(hours=24)).timestamp()) & (ts < e.timestamp())
        if sel.sum() < 23:
            continue
        kept.append(e)
        p = prec[sel]
        daily["snow"].append(snow[sel].sum(0))
        daily["prec"].append(p.sum(0))
        daily["abl"].append(abl[sel].sum(0))
        daily["t_mean"].append(temp[sel].mean(0))
        daily["t_min"].append(temp[sel].min(0))
        daily["t_max"].append(temp[sel].max(0))
        with np.errstate(all="ignore"):
            daily["t_prec"].append(np.where(p.sum(0) > 0.2, (temp[sel] * p).sum(0) / p.sum(0), np.nan))
        daily["wind_mean"].append(wind[sel].mean(0) if wind is not None else np.full(snow.shape[1:], np.nan))
        daily["wind_max"].append(wind[sel].max(0) if wind is not None else np.full(snow.shape[1:], np.nan))
        daily["sun"].append(sun[sel].sum(0) if sun is not None else np.full(snow.shape[1:], np.nan))
        daily["cloud"].append(cloud[sel].mean(0) if cloud is not None else np.full(snow.shape[1:], np.nan))
    meta = {k: v for k, v in m.items() if not isinstance(v, (list, dict)) or k in ("bounds", "nsp")}
    meta.update({"version": sm["version"], "generated": sm["generated"], "settle": sm["settle"]})
    f = out / f"snowmapper_{sm['version']}.npz"
    np.savez_compressed(
        f, meta=json.dumps(meta), times=np.array([t.isoformat() for t in times]),
        window_end=np.array([e.isoformat() for e in kept]), lat=sm["lat"].astype(np.float32), lon=sm["lon"].astype(np.float32),
        elev=sm["elev"].astype(np.float32),
        hourly=hourly,
        **({f"h_{k}": raw[k] for k in ("SNOW", "PREC", "ABL")} if hourly else {}),
        **({"h3_TEMP": raw["TEMP"][::3]} if hourly else {}),
        **{f"d_{k}": q10(np.array(v)) for k, v in daily.items()})
    return f


def png_classes(img, legend):
    """RGBA image -> uint8 legend index (0 = transparent / unknown)."""
    rgb = img[..., :3].astype(np.int16)
    out = np.zeros(img.shape[:2], np.uint8)
    best = np.full(img.shape[:2], 1 << 30, np.int64)
    for k, (_, col, _) in legend.items():
        d = ((rgb - np.array(col, np.int16)) ** 2).sum(-1)
        sel = d < best
        best[sel] = d[sel]
        out[sel] = int(k)
    out[(img[..., 3] < 128) | (best > 30 ** 2)] = 0
    return out


def archive_va(net, site, out: Path, index):
    base = site + "data/variant_a_live/"
    man = net.get(base + "manifest.json", light=True, cache=False).json()
    gen = man.get("generated", "")
    key = "va_" + gen.replace(":", "").replace("-", "")
    if key in index["files"]:
        return None
    arrays = {}
    for tag in man["tags"]:
        if not tag.endswith("T1200"):
            continue
        for layer in ("ski6", "ski18", "wind", "powder"):
            if layer not in man["layers"]:
                continue
            url = base + man["layers"][layer]["file"].format(tag=tag)
            try:
                img = np.asarray(Image.open(io.BytesIO(net.get(url, light=True).content)).convert("RGBA"))
            except Exception as e:                       # noqa: BLE001
                print(f"  VA {layer} {tag}: {e}", file=sys.stderr)
                continue
            arrays[f"{layer}_{tag[:10]}"] = png_classes(img, man["layers"][layer]["legend"])
    f = out / f"variant_a_{gen.replace(':', '').replace('-', '')}.npz"
    np.savez_compressed(f, manifest=json.dumps(man), **arrays)
    return key, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("MODEL_SITE", ""))
    ap.add_argument("--out", default="archive")
    ap.add_argument("--index", default="archive/index.json", help="what is archived already")
    ap.add_argument("--no-va", action="store_true")
    a = ap.parse_args()
    if not a.site:
        raise SystemExit("set MODEL_SITE")
    site = a.site.rstrip("/") + "/"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ip = Path(a.index)
    index = json.loads(ip.read_text()) if ip.exists() else {"files": {}}
    net = Polite(delay=(.3, .8))
    new = []
    lj = net.get(site + "data/latest.json", light=True, cache=False).json()
    key = "sm_" + lj["version"]
    if key not in index["files"]:
        sm = load_model(site)
        day = (sm["generated"] or "")[:10]
        hourly = not any(v.get("hourly") and (v.get("generated") or "")[:10] == day for v in index["files"].values())
        f = archive_model(sm, out, hourly)
        index["files"][key] = {"file": f.name, "version": sm["version"], "generated": sm["generated"], "hourly": hourly,
                               "bytes": f.stat().st_size, "archived": dt.datetime.now(UTC).isoformat(timespec="minutes")}
        new.append(f)
        print(f"model run {sm['version']}: {f.name} {f.stat().st_size / 1e6:.1f} MB{' (with hourly)' if hourly else ''}")
    else:
        print(f"model run {lj['version']} already archived")
    if not a.no_va:
        r = archive_va(net, site, out, index)
        if r:
            k, f = r
            index["files"][k] = {"file": f.name, "bytes": f.stat().st_size,
                                 "archived": dt.datetime.now(UTC).isoformat(timespec="minutes")}
            new.append(f)
            print(f"Variant A: {f.name} {f.stat().st_size / 1e6:.1f} MB")
        else:
            print("Variant A cycle already archived")
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text(json.dumps(index, indent=1))
    # for the workflow: which files to upload
    Path(out, "new_files.txt").write_text("".join(f"{p}\n" for p in new))


if __name__ == "__main__":
    main()
