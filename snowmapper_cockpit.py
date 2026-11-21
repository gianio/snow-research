#!/usr/bin/env python3
"""Snowmapper validation cockpit -- one self-contained HTML page per day.

Pulls together, for the current model run:
  1. new snow (main app layer SNOW) vs SLF HN1D, measured, for every past day
     the model file covers: maps, height bands, regions, history
  2. what-if calibration (calibrate_newsnow.py): which rain/snow threshold,
     snow-to-liquid ratio, terrain factor and precipitation factor would have
     matched the SLF -- chosen leave-one-day-out
  3. snow depth of Variant A (SNOWPACK) vs IMIS, as written in its manifest:
     bias per station, drift status, how often the precipitation factor sits
     at its bounds
  4. snow type: Variant A surface classes (ski6 + wind layer) at the summit of
     each gipfelbuch report vs the reported surface ("Oben: Pulver" …)
  5. the point validation of validate_obs.py, if it ran before
and appends the day's key numbers to a history CSV that the next day plots.

    MODEL_SITE=https://…/ python snowmapper_cockpit.py --out cockpit --data data --history cockpit_history.csv

Writes cockpit/index.html, cockpit/cockpit.json and cockpit/summary.md.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import calibrate_newsnow as cal
from compare_newsnow import compare, load_model, load_slf, sample_slf
from snowval import store
from snowval.net import Polite

UTC = dt.timezone.utc
IMIS = "https://measurement-api.slf.ch/public/api/imis/stations"

# reference palette (dataviz skill): measured = blue, model = orange, best variant = aqua
C = {"meas": "#2a78d6", "model": "#eb6834", "best": "#1baf7a",
     "meas_d": "#3987e5", "model_d": "#d95926", "best_d": "#199e70"}
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#1c5cab", "#6da7ec", "#cde2fb", "#f0efec", "#f5c3c2", "#e66767", "#b22a2a"]
HIST_COLS = ["date", "model_version", "slf_day", "slf_mean", "model_mean", "bias", "mae", "mae_10km", "ratio", "csi10",
             "best_variant", "best_mae_10km", "loo_mae", "loo_mae_current", "va_bias", "va_mae", "va_drift",
             "va_pf_floor_share", "type_n", "type_hit", "type_hit_wind"]


def esc(x):
    return html.escape(str(x), quote=True)


def f(v, p=1, sign=False, pct=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if pct:
        return f"{100 * v:.0f} %"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


# ── maps (PNG) ──────────────────────────────────────────────────────────────
def _cmap(colors):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("c", colors)


def maps_png(slf, mod, lat, lon, title_day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = np.isfinite(slf)
    if ok.any():
        r = np.where(ok.any(1))[0]
        q = np.where(ok.any(0))[0]
        rs, cs = slice(max(0, r[0] - 2), r[-1] + 3), slice(max(0, q[0] - 2), q[-1] + 3)
        slf, mod, lat, lon = slf[rs, cs], mod[rs, cs], lat[rs], lon[cs]
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    asp = 1 / math.cos(math.radians(float(np.mean(lat))))
    vmax = max(5.0, float(np.nanpercentile(np.r_[slf[np.isfinite(slf)], mod[np.isfinite(mod)]], 98)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4), dpi=120)
    for a, arr, t in ((ax[0], slf, f"SLF HN1D (gemessen) · {title_day}"), (ax[1], mod, "Snowmapper Neuschnee"),):
        im = a.imshow(arr, extent=ext, cmap=_cmap(SEQ), vmin=0, vmax=vmax, aspect=asp, interpolation="nearest")
        a.set_title(t, fontsize=10, color="#0b0b0b", loc="left")
    fig.colorbar(im, ax=ax[:2], shrink=.8, label="cm", pad=.01)
    d = mod - slf
    lim = max(3.0, float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 97))) if np.isfinite(d).any() else 5
    im2 = ax[2].imshow(d, extent=ext, cmap=_cmap(DIV), vmin=-lim, vmax=lim, aspect=asp, interpolation="nearest")
    ax[2].set_title("Modell − SLF (rot = Modell mehr)", fontsize=10, color="#0b0b0b", loc="left")
    fig.colorbar(im2, ax=ax[2], shrink=.8, label="cm", pad=.01)
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#c3c2b7")
    return _png_b64(fig)


def station_map_png(stations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [x for x in stations if x.get("lat") and x.get("bias") is not None]
    if not s:
        return None
    fig, a = plt.subplots(figsize=(7.5, 3.8), dpi=110)
    b = np.array([x["bias"] for x in s])
    lim = max(5.0, float(np.percentile(np.abs(b), 95)))
    sc = a.scatter([x["lon"] for x in s], [x["lat"] for x in s], c=b, cmap=_cmap(DIV), vmin=-lim, vmax=lim, s=46,
                   edgecolors="#fcfcfb", linewidths=1.5)
    a.set_aspect(1 / math.cos(math.radians(46.6)))
    a.set_xticks([]), a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#c3c2b7")
    a.set_title("Variant A − IMIS Schneehöhe je Station (rot = Modell mehr)", fontsize=10, loc="left")
    fig.colorbar(sc, ax=a, shrink=.85, label="cm", pad=.01)
    return _png_b64(fig)


def _png_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def nice_ticks(lo, hi, n=4):
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t0 = math.floor(lo / step) * step
    ticks = []
    v = t0
    while v <= hi + step * 1e-6:
        ticks.append(round(v, 10))
        v += step
    if ticks[-1] < hi:
        ticks.append(ticks[-1] + step)
    return ticks


def tick_fmt(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) >= 1 else f"{v:.2f}".rstrip("0")


# ── small inline SVG charts with native hover tooltips ──────────────────────
def svg_bars(groups, series, values, unit="cm", w=640, h=240):
    """groups: x labels; series: [(name, css var)]; values[s][g]."""
    pad_l, pad_b, pad_t = 44, 28, 10
    ticks = nice_ticks(0, max([v for row in values for v in row if v is not None] + [1.0]))
    vmax = ticks[-1]
    gw = (w - pad_l - 10) / len(groups)
    bw = min(26, (gw - 14) / len(series))
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for tv in ticks:
        y = pad_t + (h - pad_t - pad_b) * (1 - tv / vmax)
        out.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick_fmt(tv)}</text>')
    for gi, g in enumerate(groups):
        x0 = pad_l + gi * gw + (gw - bw * len(series) - 2 * (len(series) - 1)) / 2
        for si, (name, var) in enumerate(series):
            v = values[si][gi]
            if v is None:
                continue
            bh = (h - pad_t - pad_b) * v / vmax
            x = x0 + si * (bw + 2)
            y = h - pad_b - bh
            r = min(4, bw / 2, bh)
            out.append(f'<path d="M{x:.1f},{h - pad_b} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} '
                       f'Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{h - pad_b}Z" style="fill:var({var})">'
                       f'<title>{esc(g)} · {esc(name)}: {v:.1f} {unit}</title></path>')
        out.append(f'<text x="{pad_l + gi * gw + gw / 2:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{esc(g)}</text>')
    out.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{h - pad_b}" y2="{h - pad_b}" class="axis"/></svg>')
    return "".join(out)


def svg_lines(xs, series, unit="", w=440, h=230, zero=False):
    """xs: labels (dates); series: [(name, css var, [values])]."""
    pad_l, pad_b, pad_t = 44, 26, 10
    vals = [v for _, _, vs in series for v in vs if v is not None]
    if not vals:
        return '<p class="muted">Noch keine Werte.</p>'
    if len(xs) < 2:
        now = " · ".join(f"{n}: {tick_fmt(vs[-1])}{unit}" for n, _, vs in series if vs and vs[-1] is not None)
        return f'<p class="muted">Heute {esc(now)}. Die Kurve entsteht ab dem zweiten Tag.</p>'
    ticks = nice_ticks(min(vals + ([0] if zero else [])), max(vals + ([0] if zero else [])))
    lo, hi = ticks[0], ticks[-1]
    X = lambda i: pad_l + (w - pad_l - 12) * (i / max(1, len(xs) - 1))  # noqa: E731
    Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for v in ticks:
        out.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{tick_fmt(v)}</text>')
    if zero and lo < 0 < hi:
        out.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="axis"/>')
    step = max(1, len(xs) // 6)
    for i, x in enumerate(xs):
        if i % step == 0 or i == len(xs) - 1:
            anc = "end" if i == len(xs) - 1 else "start" if i == 0 else "middle"
            out.append(f'<text x="{X(i):.1f}" y="{h - 8}" class="tick" text-anchor="{anc}">{esc(x[5:])}</text>')
    for name, var, vs in series:
        pts = [(X(i), Y(v), v, xs[i]) for i, v in enumerate(vs) if v is not None]
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke-width="2" style="stroke:var(' + var + ')" points="'
                       + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts) + '"/>')
        for x, y, v, d in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot" style="fill:var({var})">'
                       f'<title>{esc(d)} · {esc(name)}: {v:.2f}{unit}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:var({v})"></i>{esc(n)}</span>' for n, v in items) + "</div>"


def table(head, rows, num_from=1):
    h = "".join(f'<th{" class=num" if i >= num_from else ""}>{esc(c)}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td{" class=num" if i >= num_from else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
                for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


# ── Variant A (SNOWPACK) ────────────────────────────────────────────────────
GB_CLASS = [  # gipfelbuch surface text -> coarse class (first match wins)
    ("kein", "thin"), ("aper", "thin"), ("wenig", "thin"),
    ("trieb", "wind"), ("press", "wind"), ("windgepresst", "wind"), ("verblasen", "wind"), ("sastrugi", "wind"),
    ("bruchharsch", "hard_crust"), ("harsch", "hard_crust"), ("hart", "hard_crust"), ("eis", "hard_crust"),
    ("gefroren", "hard_crust"), ("piste", "hard_crust"),
    ("sulz", "corn_wet"), ("firn", "corn_wet"), ("nass", "corn_wet"), ("faul", "corn_wet"), ("schwer", "corn_wet"),
    ("pulver", "powder"), ("weich", "powder"),
]
VA_CLASS = {"hard": "hard_crust", "crust": "hard_crust", "corn": "corn_wet", "wet": "corn_wet", "thin_cover": "thin",
            "loose_faceted": "powder"}
CLASS_DE = {"powder": "Pulver", "hard_crust": "hart / Harsch", "corn_wet": "Sulz / nass", "wind": "windbeeinflusst",
            "thin": "wenig / kein Schnee", "none": "keine Daten"}


def gb_class(txt):
    t = (txt or "").lower()
    return next((c for k, c in GB_CLASS if k in t), None)


class VariantA:
    def __init__(self, net, site):
        self.base = site.rstrip("/") + "/data/variant_a_live/"
        self.net = net
        self.m = net.get(self.base + "manifest.json", light=True, cache=False).json()
        (la0, lo0), (la1, lo1) = self.m["bounds"]
        self.b = (la0, lo0, la1, lo1)
        self.ts = [dt.datetime.fromisoformat(t).replace(tzinfo=UTC) for t in self.m["timestamps"]]
        self._img = {}

    def _layer(self, layer, tag):
        k = (layer, tag)
        if k not in self._img:
            url = self.base + self.m["layers"][layer]["file"].format(tag=tag)
            try:
                self._img[k] = np.asarray(Image.open(io.BytesIO(self.net.get(url, light=True).content)).convert("RGBA"))
            except Exception:                            # noqa: BLE001
                self._img[k] = None
        return self._img[k]

    def sample(self, layer, t, lat, lon):
        """Class name of `layer` at (lat, lon) for the frame nearest t (same day), or None / 'none'."""
        if not self.ts:
            return None
        i = int(np.argmin([abs((x - t).total_seconds()) for x in self.ts]))
        if abs((self.ts[i] - t).total_seconds()) > 6 * 3600:
            return None
        im = self._layer(layer, self.m["tags"][i])
        if im is None:
            return None
        la0, lo0, la1, lo1 = self.b
        h, w = im.shape[:2]
        c, r = int((lon - lo0) / (lo1 - lo0) * w), int((la1 - lat) / (la1 - la0) * h)
        if not (0 <= c < w and 0 <= r < h):
            return None
        px = im[r, c]
        if px[3] < 128:
            return "none"
        leg = self.m["layers"][layer]["legend"]
        best, name = 1e9, None
        for _, (nm, rgb, _de) in leg.items():
            d = sum((int(px[j]) - rgb[j]) ** 2 for j in range(3))
            if d < best:
                best, name = d, nm
        return name if best < 30 ** 2 else None


def snowtype_check(va, data_dir, days=6):
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = store.load(data_dir, "gipfelbuch", "surface_top", since=since) if data_dir else []
    out = []
    for r in rows:
        rep = gb_class(r["value"] if isinstance(r["value"], str) else "")
        if not rep or r["lat"] is None:
            continue
        t = dt.datetime.fromisoformat(r["valid_to"][:10]).replace(hour=10, tzinfo=UTC)
        s6 = va.sample("ski6", t, r["lat"], r["lon"])
        if s6 is None:
            continue
        wind = va.sample("wind", t, r["lat"], r["lon"]) if "wind" in va.m["layers"] else None
        mod = "none" if s6 == "none" else ("powder" if s6.startswith("powder") else VA_CLASS.get(s6, s6))
        hit = rep == mod
        hit_w = hit or (rep == "wind" and wind not in (None, "none")) or (rep == "hard_crust" and wind == "scoured")
        out.append({"date": r["valid_to"][:10], "url": r["ref"].split(" | ")[0], "id": r["point_id"], "elev": r["elev"],
                    "reported": r["value"], "rep": rep, "ski6": s6, "wind": wind, "mod": mod, "hit": hit, "hit_wind": hit_w})
    return out


def imis_coords(net):
    try:
        return {s["code"]: s for s in net.get(IMIS, light=True).json()}
    except Exception:                                    # noqa: BLE001
        return {}


# ── history ─────────────────────────────────────────────────────────────────
def hist_update(path, row):
    rows = []
    if path and Path(path).exists():
        with open(path) as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] != row["date"]]
    rows.append({k: row.get(k, "") for k in HIST_COLS})
    rows.sort(key=lambda r: r["date"])
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HIST_COLS)
            w.writeheader()
            w.writerows(rows)
    return rows


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ── page ────────────────────────────────────────────────────────────────────
CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--meas:%(meas)s;--model:%(model)s;--best:%(best)s;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--goodtxt:#006300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--meas:%(meas_d)s;
--model:%(model_d)s;--best:%(best_d)s;--goodtxt:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--meas:%(meas_d)s;--model:%(model_d)s;--best:%(best_d)s;--goodtxt:#0ca30c}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 10px}h3{font-size:15px;margin:18px 0 6px;color:var(--ink2)}
.sub,.muted{color:var(--ink2)}.muted{font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:26px;font-variant-numeric:tabular-nums;line-height:1.2}.kpi span{font-size:13px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin:12px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
img.map{width:100%%;height:auto;border-radius:8px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%%;font-size:13.5px}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.chart{width:100%%;height:auto}.chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:12px}td.wrap{white-space:normal;min-width:220px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.chart .dot{stroke:var(--surface);stroke-width:2}
.chart path:hover,.chart circle:hover{opacity:.8}
.wrapme{white-space:normal;display:inline-block;max-width:360px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:4px 0 8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.st{display:inline-flex;align-items:center;gap:6px;font-weight:600}.st i{width:10px;height:10px;border-radius:50%%;display:inline-block}
.ok i{background:var(--good)}.watch i{background:var(--warn)}.bad i{background:var(--crit)}
.note{border-left:3px solid var(--axis);padding:6px 12px;color:var(--ink2);margin:10px 0}
a{color:var(--meas)}code{font-size:12.5px}
""" % C


def status(s):
    s = (s or "").upper()
    cls, sym = ("ok", "✓") if s == "OK" else ("watch", "!") if s in ("WATCH", "NODATA") else ("bad", "✕")
    return f'<span class="st {cls}"><i></i>{sym} {esc(s or "–")}</span>'


def build(a):
    if not (a.site or a.sm_file):
        raise SystemExit("set MODEL_SITE (or pass --site / --sm-file)")
    net = Polite(delay=(.3, .8), cache_dir=a.cache)
    site = a.site.rstrip("/") + "/"
    sm = load_model(site if not a.sm_file else "", a.sm_file)
    lat, lon = sm["lat"], sm["lon"]
    J = {"generated": dt.datetime.now(UTC).isoformat(timespec="minutes"), "model_version": sm["version"],
         "model_generated": sm["generated"]}
    log = lambda *x: print(*x, file=sys.stderr)  # noqa: E731

    # 1+2: new snow, all past days + calibration
    calib = cal.run(sm, a.days, log=log)
    days = calib["days"]
    last = days[-1]
    slf = load_slf(last)
    t1 = slf["valid"]
    from compare_newsnow import sm_window
    slf_g = sample_slf(slf, lat, lon)
    mask = np.isfinite(slf_g)
    mod_g = np.where(mask, sm_window(sm, t1 - dt.timedelta(hours=24), t1), np.nan)
    res = compare(slf_g, mod_g, sm["elev"], lat, lon)
    cur = calib["scores"]["aktuell"]
    best = calib["best_pooled"]
    J["newsnow"] = {"day": last, "window_end": t1.isoformat(), "cells": res["cells"], "blocks_10km": res["blocks_10km"],
                    "elev": res["elev"], "region": res["region"]}
    J["calibration"] = {k: calib[k] for k in ("days", "current_params", "best_pooled", "leave_one_day_out",
                                               "global_factor", "by_elev", "terrain_factor_median")}
    J["calibration"]["variant_params"] = {best: calib["variants"][best], "aktuell": calib["variants"]["aktuell"]}
    maps = maps_png(slf_g, mod_g, lat, lon, f"24 h bis {t1:%d.%m. %H} UTC")

    # 3: Variant A
    va, va_err = None, None
    try:
        va = VariantA(net, site)
    except Exception as e:                               # noqa: BLE001
        va_err = f"{type(e).__name__}: {e}"
    vm = va.m if va else {}
    v_imis = (vm.get("validation") or {}).get("imis") or {}
    coords = imis_coords(net)
    st = []
    for s in (vm.get("validation") or {}).get("stations", []):
        c = coords.get(s["station"], {})
        st.append({**s, "lat": c.get("lat"), "lon": c.get("lon")})
    pf = list((vm.get("precip_factor") or {}).values())
    pf_floor = (sum(1 for x in pf if x <= 0.5001) / len(pf)) if pf else None
    pf_cap = (sum(1 for x in pf if x >= 1.999) / len(pf)) if pf else None
    drift = (vm.get("drift") or {})
    J["variant_a"] = {"generated": vm.get("generated"), "imis": v_imis, "drift": drift, "precip_factor_floor_share": pf_floor,
                      "precip_factor_cap_share": pf_cap, "n_precip_factors": len(pf), "error": va_err,
                      "community": (vm.get("validation") or {}).get("community")}
    smap = station_map_png(st) if st else None

    # 4: snow type
    types = snowtype_check(va, a.data) if va else []
    n_t = len(types)
    hit = sum(x["hit"] for x in types) / n_t if n_t else None
    hit_w = sum(x["hit_wind"] for x in types) / n_t if n_t else None
    J["snowtype"] = {"n": n_t, "hit": hit, "hit_wind": hit_w, "rows": types}

    # 5: point validation (validate_obs.py output, optional)
    pv = None
    if a.val and Path(a.val, "summary.json").exists():
        pv = json.loads(Path(a.val, "summary.json").read_text())
    J["points"] = pv

    # history
    loo = [x for x in calib["leave_one_day_out"] if x["held_out"] and x["current"]]
    row = {"date": dt.date.today().isoformat(), "model_version": sm["version"], "slf_day": last,
           "slf_mean": f(res["cells"].get("mean_ref"), 2), "model_mean": f(res["cells"].get("mean_sm"), 2),
           "bias": f(res["cells"].get("bias"), 2), "mae": f(res["cells"].get("mae"), 2),
           "mae_10km": f(res["blocks_10km"].get("mae"), 2), "ratio": f(cur[last]["ratio"] if cur.get(last) else None, 3),
           "csi10": f(cur[last]["csi10"] if cur.get(last) else None, 3), "best_variant": best,
           "best_mae_10km": f(np.mean([calib["scores"][best][d]["mae_10km"] for d in days if calib["scores"][best].get(d)]), 2),
           "loo_mae": f(np.mean([x["held_out"]["mae"] for x in loo]) if loo else None, 2),
           "loo_mae_current": f(np.mean([x["current"]["mae"] for x in loo]) if loo else None, 2),
           "va_bias": f(v_imis.get("bias_cm"), 1), "va_mae": f(v_imis.get("mae_cm"), 1),
           "va_drift": (drift.get("all") or {}).get("status", ""), "va_pf_floor_share": f(pf_floor, 3),
           "type_n": n_t, "type_hit": f(hit, 3), "type_hit_wind": f(hit_w, 3)}
    hist = hist_update(a.history, row)
    J["history_rows"] = len(hist)

    # ── HTML ──
    P = []
    c, b10 = res["cells"], res["blocks_10km"]
    ratio = cur[last]["ratio"] if cur.get(last) else None
    loo_now = np.mean([x["current"]["mae"] for x in loo]) if loo else None
    loo_new = np.mean([x["held_out"]["mae"] for x in loo]) if loo else None
    P.append(f"""<header><h1>Snowmapper-Validierung</h1>
<p class="sub">Modell-Lauf <b>{esc(sm['version'])}</b> (erzeugt {esc(sm['generated'])}) · Cockpit erstellt {esc(J['generated'])} UTC ·
Variant A {esc(vm.get('generated', '–'))}</p></header>
<section class="kpis">
<div class="kpi"><span>Neuschnee Modell / SLF ({esc(last[5:])})</span><b>{f(ratio, 2)}×</b><span>1.00 = richtige Menge</span></div>
<div class="kpi"><span>Neuschnee MAE, 10-km-Blöcke</span><b>{f(b10.get('mae'))} cm</b><span>Bias {f(c.get('bias'), 1, True)} cm (1 km)</span></div>
<div class="kpi"><span>Beste Variante, am weggelassenen Tag geprüft</span><b>{f(loo_new)} cm</b><span>MAE statt {f(loo_now)} cm heute</span></div>
<div class="kpi"><span>Schneehöhe Variant A vs IMIS</span><b>{f(v_imis.get('bias_cm'), 1, True)} cm</b><span>MAE {f(v_imis.get('mae_cm'))} cm · {v_imis.get('stations', '–')} Stationen</span></div>
<div class="kpi"><span>Drift (Saison)</span><b>{status((drift.get('all') or {}).get('status'))}</b><span>unabhängig: {esc((drift.get('independent') or {}).get('status', '–'))}</span></div>
<div class="kpi"><span>Schneetyp-Treffer (Gipfelbuch)</span><b>{f(hit_w, pct=True)}</b><span>{n_t} Berichte · ohne Windlayer {f(hit, pct=True)}</span></div>
</section>""")

    # new snow section
    eb = [k for k in res["elev"] if res["elev"][k].get("n", 0) >= 10]
    P.append(f"""<h2>1 · Neuschnee-Layer gegen SLF-Messung</h2>
<p class="muted">SLF HN1D (24 h bis {t1:%d.%m. %H:%M} UTC, aus dem Messnetz interpoliert) gegen den Neuschnee von Snowmapper
im selben Zeitraum (nach 24 h Setzung, Faktor {sm['settle']:.3f}). Nur Zellen, für die das SLF Daten hat.</p>
<div class="card"><img class="map" alt="Karten SLF, Modell, Differenz" src="data:image/png;base64,{maps}"></div>
<div class="grid2"><div class="card"><h3>Mittel nach Höhe (letzte {len(days)} Tage)</h3>
{legend([("SLF gemessen", "--meas"), ("Snowmapper heute", "--model"), ("beste Variante", "--best")])}
{svg_bars([x["band"] for x in calib["by_elev"]["aktuell"]],
          [("SLF", "--meas"), ("Snowmapper", "--model"), ("beste Variante", "--best")],
          [[x["mean_ref"] for x in calib["by_elev"]["aktuell"]], [x["mean_mod"] for x in calib["by_elev"]["aktuell"]],
           [x["mean_mod"] for x in calib["by_elev"][best]]])}
</div><div class="card"><h3>Je Tag (aktueller Layer)</h3>
{table(["Tag", "SLF Ø cm", "Modell Ø cm", "Bias", "MAE", "MAE 10 km", "Modell/SLF", "CSI ≥10"],
       [[d, f(cur[d]["mean_ref"]), f(cur[d]["mean_mod"]), f(cur[d]["bias"], 1, True), f(cur[d]["mae"]), f(cur[d]["mae_10km"]),
         f(cur[d]["ratio"], 2), f(cur[d]["csi10"], 2)] for d in days if cur.get(d)])}</div></div>
<div class="grid2"><div class="card"><h3>Nach Höhe ({esc(last)})</h3>
{table(["Höhe", "n", "SLF", "Modell", "Bias", "MAE", "r"],
       [[k, res["elev"][k]["n"], f(res["elev"][k].get("mean_ref")), f(res["elev"][k].get("mean_sm")), f(res["elev"][k].get("bias"), 1, True),
         f(res["elev"][k].get("mae")), f(res["elev"][k].get("pearson"), 2)] for k in eb])}</div>
<div class="card"><h3>Nach Region ({esc(last)})</h3>
{table(["Region", "n", "SLF", "Modell", "Bias", "MAE", "r"],
       [[k, v["n"], f(v.get("mean_ref")), f(v.get("mean_sm")), f(v.get("bias"), 1, True), f(v.get("mae")), f(v.get("pearson"), 2)]
        for k, v in res["region"].items() if v.get("n", 0) >= 10])}</div></div>""")
    det = c.get("detect", {})
    P.append('<div class="card"><h3>Erkennung von Schwellen (1-km-Zellen)</h3>' + table(
        ["Schwelle", "SLF", "Modell", "POD", "FAR", "CSI", "Frequenz-Bias"],
        [[f"≥ {k} cm", v["obs"], v["fc"], f(v["pod"], pct=True), f(v["far"], pct=True), f(v["csi"], 2), f(v["fbias"], 2)]
         for k, v in det.items()]) + "</div>")

    # calibration section
    tbl = cal.table(calib)
    hdr = [x.strip() for x in tbl[0].strip("|").split("|")]
    rows_ = [[x.strip().replace("**", "") for x in ln.strip("|").split("|")] for ln in tbl[2:]]
    pb = calib["variants"][best]
    hint = []
    if pb[3] < 0.9 and pf_floor and pf_floor > .2:
        hint.append(f"Die beste Variante senkt den Niederschlag auf ×{pb[3]:g}, und Variant A hält den IMIS-Niederschlagsfaktor "
                    f"bei {pf_floor:.0%} der Wetterpunkte am Minimum 0.5 – zwei unabhängige Hinweise, dass der "
                    "Niederschlags-Input zu hoch ist (Plan, Schritt 1).")
    if pb[1] < calib["current_params"]["slr_max"]:
        hint.append(f"SLR-Obergrenze {pb[1]:g} statt {calib['current_params']['slr_max']:g}: Neuschnee ist in den Alpen "
                    "meist dichter (Wind), als das Modell annimmt.")
    hint.append("Niederschlagsfaktor und SLR lassen sich gegen die SLF-Karte allein nicht trennen (beide skalieren cm pro mm). "
                "Das trennt erst ein Vergleich mit gemessenem Niederschlag (SwissMetNet).")
    P.append(f"""<h2>2 · Was hätte besser gepasst? (Kalibrierung)</h2>
<p class="muted">Aus der Modelldatei zurückgerechnet: Niederschlag (PREC) × Schneeanteil(T) × SLR(T)/10 × Geländefaktor.
{len(calib['variants'])} Varianten über {len(days)} Tage gegen SLF HN1D. Sortiert nach MAE auf 10-km-Blöcken.</p>
<div class="card">{table(hdr, rows_)}</div>
<div class="card"><h3>Ehrlich geprüft: Parameter gewählt ohne den Tag, an dem gemessen wird</h3>
{table(["Tag", "gewählte Variante", "MAE gewählt", "MAE aktuell", "Bias gewählt", "Bias aktuell"],
       [[x["day"], esc(x["picked"]), f(x["held_out"]["mae"]), f(x["current"]["mae"]), f(x["held_out"]["bias"], 1, True),
         f(x["current"]["bias"], 1, True)] for x in loo])}
<p class="muted">Globaler Faktor, der das aktuelle Modell im Mittel richtig skalieren würde: <b>{calib['global_factor']:.2f}</b>.
Median des Geländefaktors (Höhe × Luv × Wind): {calib['terrain_factor_median']:.2f}.</p></div>
""" + "".join(f'<p class="note">{esc(h)}</p>' for h in hint))

    # history
    xs = [r["date"] for r in hist]
    P.append(f"""<h2>3 · Verlauf</h2><div class="grid3"><div class="card"><h3>Neuschnee: Modell / SLF (1.0 = richtig)</h3>
{legend([("Snowmapper", "--model")])}{svg_lines(xs, [("Modell/SLF", "--model", [num(r["ratio"]) for r in hist])], "×")}</div>
<div class="card"><h3>Neuschnee MAE 10 km: heute vs beste Variante</h3>
{legend([("Snowmapper", "--model"), ("beste Variante", "--best")])}
{svg_lines(xs, [("Snowmapper", "--model", [num(r["mae_10km"]) for r in hist]), ("beste Variante", "--best", [num(r["best_mae_10km"]) for r in hist])], " cm")}</div>
<div class="card"><h3>Variant A Schneehöhe: Bias gegen IMIS</h3>{legend([("Bias", "--model")])}
{svg_lines(xs, [("Bias cm", "--model", [num(r["va_bias"]) for r in hist])], " cm", zero=True)}</div>
<div class="card"><h3>Schneetyp-Trefferquote</h3>{legend([("mit Windlayer", "--best"), ("nur ski6", "--meas")])}
{svg_lines(xs, [("mit Windlayer", "--best", [num(r["type_hit_wind"]) for r in hist]), ("nur ski6", "--meas", [num(r["type_hit"]) for r in hist])])}</div></div>""")

    # variant A
    worst = sorted([s for s in st if s.get("bias") is not None], key=lambda s: -abs(s["bias"]))[:12]
    dr = drift.get("all") or {}
    P.append(f"""<h2>4 · Schneedecke (Variant A / SNOWPACK) gegen IMIS</h2>
{('<p class="note">Variant A nicht lesbar: ' + esc(va_err) + '</p>') if va_err else ''}
<div class="grid2"><div class="card">{('<img class="map" alt="Bias je Station" src="data:image/png;base64,' + smap + '">') if smap else '<p class="muted">keine Stationen</p>'}</div>
<div class="card"><h3>Drift und Korrektur</h3>{table(["", "Wert"], [
        ["Status (alle / unabhängig)", f"{status(dr.get('status'))} / {status((drift.get('independent') or {}).get('status'))}"],
        ["Bias 7 Tage", f"{f(dr.get('bias_cm'), 1, True)} cm ({f(dr.get('rel_bias'), pct=True)} der Schneehöhe)"],
        ["Trend", f"{f(dr.get('trend_cm_per_day'), 2, True)} cm/Tag"],
        ["Gründe", '<span class="wrapme">' + esc('; '.join(dr.get('reasons') or []) or '–') + '</span>'],
        ["Niederschlagsfaktor am Minimum 0.5", f"{f(pf_floor, pct=True)} von {len(pf)} Wetterpunkten"],
        ["… am Maximum 2.0", f(pf_cap, pct=True)],
        ["Community-Berichte (72 h)", esc(((vm.get('validation') or {}).get('community') or {}).get('reports', '–'))]], num_from=9)}</div></div>
<div class="card"><h3>Grösste Abweichungen</h3>{table(["Station", "Name", "Höhe", "n", "Bias", "MAE", "zuletzt gemessen", "zuletzt Modell", "Holdout"],
        [[esc(s['station']), esc(s.get('label', '')), f(s.get('elev'), 0), s.get('n', ''), f(s['bias'], 1, True), f(s.get('mae')),
          f(s.get('last_meas')), f(s.get('last_model')), "ja" if s.get('holdout') else ""] for s in worst], num_from=2)}</div>""")

    # snow type
    conf = {}
    for x in types:
        conf.setdefault(x["rep"], {}).setdefault(x["mod"], 0)
        conf[x["rep"]][x["mod"]] += 1
    cols = sorted({m for v in conf.values() for m in v})
    P.append("""<h2>5 · Schneetyp gegen Gipfelbuch-Berichte</h2>
<p class="muted">Gemeldete Oberfläche «Oben» (Gipfelbuch) gegen die Variant-A-Klasse (ski6, 10 UTC des Tourtags) an der
Routenkoordinate. «mit Windlayer» zählt auch «windbeeinflusst» als Treffer, wenn der Windlayer Triebschnee oder abgeblasen zeigt.</p>""")
    if types:
        P.append('<div class="grid2"><div class="card"><h3>Verwechslungstabelle (Zeilen = gemeldet)</h3>' + table(
            ["gemeldet \\ Modell"] + [CLASS_DE.get(m, m) for m in cols],
            [[CLASS_DE.get(r, r)] + [conf[r].get(m, 0) for m in cols] for r in sorted(conf)]) + "</div>")
        P.append('<div class="card"><h3>Berichte</h3>' + table(
            ["Datum", "Bericht", "Höhe", "gemeldet", "Modell ski6", "Wind", "Treffer"],
            [[x["date"], f'<a href="{esc(x["url"])}">{esc(x["id"])}</a>', f(x["elev"], 0), esc(x["reported"]),
              esc("keine Daten" if x["ski6"] == "none" else x["ski6"]),
              esc({"none": "kein", None: "–"}.get(x["wind"], x["wind"])), "✓" if x["hit_wind"] else "✕"] for x in sorted(types, key=lambda x: x["date"], reverse=True)[:40]],
            num_from=2) + "</div></div>")
    else:
        P.append('<p class="note">Noch keine passenden Berichte. Sie kommen aus dem Branch <code>data</code> '
                 '(Workflow «Collect validation data»); ohne Gipfelbuch-Login sind es höchstens 3 pro Lauf, '
                 'und nur Berichte im Zeitfenster von Variant A (letzte ~5 Tage) zählen.</p>')

    # points
    if pv:
        P.append("<h2>6 · Punktvalidierung (alle Quellen)</h2><div class='card'>" + table(
            ["Quelle / Grösse / Art", "n", "Ø Quelle", "Ø Modell", "Bias", "MAE", "r"],
            [[esc(k), v.get("n"), f(v.get("mean_ref")), f(v.get("mean_sm")), f(v.get("bias"), 1, True), f(v.get("mae")),
              f(v.get("pearson"), 2)] for k, v in sorted(pv.items())]) + "</div>")

    P.append("""<h2>Quellen und Grenzen</h2><ul class="muted">
<li>SLF/WhiteRisk HN1D: interpolierte Messung (Stationen meist 1500–2500 m, flach, windgeschützt). Über 2500 m und in
Lee-/Luvlagen ist sie selbst unsicher – dort zählt die Höhenverteilung mehr als die einzelne Zelle.</li>
<li>Kalibrierung: wenige Tage = wenige Ereignisse. Erst nach 3–4 Wochen mit mehreren Schneefällen sind die Parameter belastbar.</li>
<li>Variant A: Zahlen aus dem Manifest des Laufs (IMIS-Vergleich, Drift, Community), nicht hier neu gerechnet.</li>
<li>Gipfelbuch: Selbstauskunft, Koordinate = Routenpunkt (meist Gipfel); «Oben» passt am besten zur Gipfelzelle.</li></ul>""")

    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snowmapper-Validierung</title><style>{CSS}</style></head><body><main>{''.join(P)}</main></body></html>"""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    (out / "cockpit.json").write_text(json.dumps(J, indent=1, default=str))
    md = [f"### Snowmapper-Validierung {row['date']} · Lauf {sm['version']}", "",
          "| Kennzahl | Wert |", "|---|---|",
          f"| Neuschnee Modell/SLF ({last}) | {f(ratio, 2)}× |",
          f"| Neuschnee MAE 10 km / Bias | {f(b10.get('mae'))} cm / {f(c.get('bias'), 1, True)} cm |",
          f"| Beste Variante (Tag weggelassen) | MAE {f(loo_new)} statt {f(loo_now)} cm · `{best}` |",
          f"| Variant A vs IMIS | Bias {f(v_imis.get('bias_cm'), 1, True)} cm, MAE {f(v_imis.get('mae_cm'))} cm |",
          f"| Drift | {(drift.get('all') or {}).get('status', '–')} |",
          f"| Niederschlagsfaktor am Minimum | {f(pf_floor, pct=True)} |",
          f"| Schneetyp-Treffer | {f(hit_w, pct=True)} ({n_t} Berichte) |", "",
          "Cockpit: Artefakt `snowmapper-cockpit-…` → `index.html`"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("MODEL_SITE", ""), help="model site (or env MODEL_SITE)")
    ap.add_argument("--sm-file", help="local snowdata-*.json.gz instead of the site's newest")
    ap.add_argument("--data", default="data", help="observation store (branch data), for gipfelbuch")
    ap.add_argument("--val", default=None, help="folder with validate_obs.py output (summary.json)")
    ap.add_argument("--history", default="cockpit_history.csv")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--out", default="cockpit")
    ap.add_argument("--cache", default=None)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
