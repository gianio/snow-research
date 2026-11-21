#!/usr/bin/env python3
"""Check Snowmapper against screenshots of SLF / WhiteRisk maps.

You take screenshots (WhiteRisk app/web, SLF Pro), list them in a manifest
with what they show and when, and this tool

  1. georeferences each image -- from places on the map (city markers,
     refined to the marker ring automatically) or, for the Swiss-projection
     SLF Pro maps, from the extreme points of the national border;
  2. reads the values off the image with the map's own colour scale:
       stations    WhiteRisk station circles at every SLF IMIS station position
       raster      WhiteRisk area maps (new snow, snow depth), every 1 km cell
       obs_points  SLF Pro observer circles (new snow observed)
       obs_symbols SLF Pro snow-surface symbols (detected; classes via the legend)
  3. reads the model at the same place and time:
       new snow    Snowmapper SNOW layer, summed over the same 24 h
       snow depth  Variant A (SNOWPACK) total snow height, flat slope, at the
                   station's / cell's height (from its metric packs)
       surface     Variant A ski6 / ski18 / wind class
  4. writes a report with agreement per colour class, bias, and an overlay
     image per screenshot (Swiss border + sampled points drawn on it) so you
     can see that the georeferencing is right before trusting any number.

The screenshot classes are bins (e.g. 20-50 cm), so model values are binned
the same way: "same class" and "within one class" are the fair measures.

Where things go
    screencheck/inbox/<set>/     your screenshots + manifest.json   (git-ignored)
    screencheck/results/<set>/   report.md, results.json, overlays (git-ignored)
A <set> is any folder name, e.g. a date or "januar-sturm".

    python screencheck/screencheck.py init <set>    # manifest skeleton for new images, with marker rings found
    python screencheck/screencheck.py run  <set>    # check (MODEL_SITE must be set)
    python screencheck/screencheck.py run  <set> --archive archive   # also older days, from the forecast archive

Images of any resolution work: marker and symbol sizes scale with the image.
Screenshots and results stay local: they show SLF products, partly from a
personal SLF Pro login.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                   # repo root: compare_newsnow, snowval, scrape_snowmaps
INBOX, RESULTS = HERE / "inbox", HERE / "results"

UTC = dt.timezone.utc
ZRH = ZoneInfo("Europe/Zurich")
R = 6378137.0

CITIES = {  # lat, lon of the town centre / map marker
    "Basel": (47.5596, 7.5886), "Zürich": (47.3769, 8.5417), "Bern": (46.9480, 7.4474), "Luzern": (47.0502, 8.3093),
    "Lausanne": (46.5197, 6.6323), "Genève": (46.2044, 6.1432), "St. Gallen": (47.4245, 9.3767), "Biel": (47.1368, 7.2468),
    "Winterthur": (47.4988, 8.7237), "Milano": (45.4642, 9.1900), "Torino": (45.0703, 7.6869), "Strasbourg": (48.5734, 7.7521),
    "Stuttgart": (48.7758, 9.1829), "Mulhouse": (47.7508, 7.3359), "Freiburg": (47.9990, 7.8421), "Ulm": (48.4011, 9.9876),
    "Parma": (44.8015, 10.3279), "Genova": (44.4056, 8.9463), "Grenoble": (45.1885, 5.7245), "Chur": (46.8508, 9.5320),
}
_B = json.loads((HERE / "ch_border.json").read_text())
BORDER = [np.array(r)[:, ::-1] for r in _B["rings"]]   # (lat, lon) rings, © swisstopo
_all = np.vstack(BORDER)
CH_EXTREMES = {"west": tuple(_all[_all[:, 1].argmin()]), "east": tuple(_all[_all[:, 1].argmax()]),
               "north": tuple(_all[_all[:, 0].argmax()]), "south": tuple(_all[_all[:, 0].argmin()])}


def in_switzerland(lat, lon):
    from matplotlib.path import Path as MPath
    pts = np.column_stack([lat, lon])
    inside = np.zeros(len(pts), bool)
    for r in BORDER:
        inside ^= MPath(r).contains_points(pts)
    return inside

WR_COLORS = ["#CDFFCD", "#99F0B2", "#53BD9F", "#3296B4", "#0670B0", "#054F8C", "#610432", "#4D020F"]
LEGENDS = {  # lower bounds of each colour class, cm
    "wr_hn": [1, 10, 20, 30, 40, 60, 80, 100],
    "wr_hs": [1, 20, 50, 80, 120, 200, 300, 400],
}


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


PAL = np.array([hexrgb(h) for h in WR_COLORS])


def bin_of(v, edges):
    """cm -> class index (-1 = below the first class, i.e. 0 / no snow)."""
    if v is None or not np.isfinite(v):
        return None
    return int(np.searchsorted(edges, v, side="right") - 1)


def bin_label(i, edges):
    if i is None:
        return "–"
    if i < 0:
        return f"< {edges[0]}"
    return f"{edges[i]}–{edges[i + 1]}" if i + 1 < len(edges) else f"≥ {edges[i]}"


# ── projections ─────────────────────────────────────────────────────────────
def merc(lat, lon):
    return np.radians(lon) * R, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R


def lv95(lat, lon):
    """swisstopo approximation WGS84 -> LV95 (metres, ~1 m)."""
    p = (np.asarray(lat) * 3600 - 169028.66) / 10000
    l = (np.asarray(lon) * 3600 - 26782.5) / 10000
    e = 2600072.37 + 211455.93 * l - 10938.51 * l * p - 0.36 * l * p ** 2 - 44.54 * l ** 3
    n = 1200147.07 + 308807.95 * p + 3745.25 * l ** 2 + 76.63 * p ** 2 - 194.56 * l ** 2 * p + 119.79 * p ** 3
    return e, n


PROJ = {"merc": merc, "lv95": lv95}


class Georef:
    """Affine map from projected coordinates to image pixels."""

    def __init__(self, proj, A, resid=None, how=""):
        self.proj, self.A, self.resid, self.how = proj, A, resid, how

    @classmethod
    def fit(cls, proj, pts, how):
        X = np.array([[*PROJ[proj](la, lo), 1.0] for _, _, la, lo in pts])
        P = np.array([[px, py] for px, py, _, _ in pts], float)
        A, *_ = np.linalg.lstsq(X, P, rcond=None)
        res = np.sqrt(((X @ A - P) ** 2).sum(1))
        return cls(proj, A, res, how)

    def px(self, lat, lon):
        x, y = PROJ[self.proj](np.asarray(lat, float), np.asarray(lon, float))
        return x * self.A[0, 0] + y * self.A[1, 0] + self.A[2, 0], x * self.A[0, 1] + y * self.A[1, 1] + self.A[2, 1]

    def latlon(self, px, py):
        """Inverse by Newton steps on the forward map (projections are smooth)."""
        lat, lon = 46.8, 8.2
        for _ in range(30):
            x0, y0 = self.px(lat, lon)
            xa, ya = self.px(lat + 1e-3, lon)
            xb, yb = self.px(lat, lon + 1e-3)
            J = np.array([[xa - x0, xb - x0], [ya - y0, yb - y0]]) / 1e-3
            d = np.linalg.solve(J, [px - x0, py - y0])
            lat, lon = lat + d[0], lon + d[1]
            if abs(d).max() < 1e-7:
                break
        return float(lat), float(lon)


def scale_of(img):
    return max(0.5, min(img.shape[:2]) / 1290)


def find_rings(img):
    """All small dark rings with a light centre (city markers) -> [(x, y)]."""
    k = scale_of(img)
    lum = img.mean(-1)
    lab, n = ndi.label(lum < 80)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 8 * k <= h <= 24 * k and 8 * k <= w <= 24 * k and abs(h - w) <= 4 * k:
            cy, cx = (sl[0].start + sl[0].stop - 1) / 2, (sl[1].start + sl[1].stop - 1) / 2
            if (lab[sl] == i).mean() < 0.6 and lum[int(cy), int(cx)] > 180:
                out.append((round(cx), round(cy)))
    return out


def refine_ring(img, x, y, r=None):
    """Move a guessed city marker position onto the centre of the nearby dark ring."""
    k = scale_of(img)
    r = r or 14 * k
    lum = img.mean(-1)
    pad = int(r + 12 * k)
    y0, y1, x0, x1 = max(0, int(y) - pad), int(y) + pad, max(0, int(x) - pad), int(x) + pad
    sub = lum[y0:y1, x0:x1] < 80
    lab, n = ndi.label(sub)
    best = None
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 8 * k <= h <= 24 * k and 8 * k <= w <= 24 * k and abs(h - w) <= 4 * k:
            cy, cx = y0 + (sl[0].start + sl[0].stop - 1) / 2, x0 + (sl[1].start + sl[1].stop - 1) / 2
            d = math.hypot(cx - x, cy - y)
            if d <= r and (best is None or d < best[0]):
                best = (d, cx, cy)
    return (best[1], best[2], True) if best else (x, y, False)


def border_mask(img, gray=(138, 166), max_sat=14, x_range=None):
    a = img.astype(int)
    g = a.mean(-1)
    sat = a.max(-1) - a.min(-1)
    m = (g >= gray[0]) & (g <= gray[1]) & (sat < max_sat)
    if x_range:
        m[:, :x_range[0]] = False
        m[:, x_range[1]:] = False
    return m


def refine_on_border(G, img, spec):
    """Fit the official border line onto the border drawn in the image (all 6 affine parameters).
    Objective: mean distance (capped) from the projected border vertices to the nearest border pixel."""
    from scipy.optimize import minimize
    m = border_mask(img, tuple(spec.get("gray", (138, 166))), spec.get("max_sat", 14), spec.get("x_range"))
    if m.sum() < 500:
        return G, None
    dist = ndi.distance_transform_edt(~m)
    cap = 25 * scale_of(img)
    pts = np.vstack(BORDER)
    X, Y = PROJ[G.proj](pts[:, 0], pts[:, 1])
    P = np.column_stack([X, Y, np.ones_like(X)])
    h, w = m.shape

    q0 = P @ G.A                                          # current pixel positions
    cx, cy = w / 2, h / 2

    def warp(v):
        sx, sy, rot, shear, tx, ty = v
        M = np.array([[sx, shear - rot], [rot, sy]])
        return (q0 - [cx, cy]) @ M.T + [cx + tx, cy + ty]

    def cost(v):
        q = warp(v)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        if ok.mean() < 0.5:
            return cap
        d = dist[q[ok, 1].astype(int), q[ok, 0].astype(int)]
        return float(np.minimum(d, cap).mean() + cap * (1 - ok.mean()))
    v0 = np.array([1, 1, 0, 0, 0, 0], float)
    c0 = cost(v0)
    step = np.array([.03, .03, .02, .02, 8 * scale_of(img), 8 * scale_of(img)])
    best = (c0, v0)
    for start in (v0,) + tuple(v0 + np.array([0, 0, 0, 0, dx, dy]) * scale_of(img) for dx, dy in ((10, 0), (-10, 0), (0, 10), (0, -10))):
        simplex = np.vstack([start] + [start + np.eye(6)[k] * step[k] for k in range(6)])
        r = minimize(cost, start, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-3, "maxiter": 3000})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    c1, v = best
    if c1 < c0 - 0.2:
        # fold the pixel-space correction into the affine: q' = (q - c) M^T + c + t
        sx, sy, rot, shear, tx, ty = v
        M = np.array([[sx, shear - rot], [rot, sy]])
        A = G.A.copy()
        A[:2] = G.A[:2] @ M.T
        A[2] = (G.A[2] - [cx, cy]) @ M.T + [cx + tx, cy + ty]
        return Georef(G.proj, A, G.resid, G.how + f" + an Landesgrenze angepasst (mittl. Abstand {c0:.1f} → {c1:.1f} px)"), c1
    return Georef(G.proj, G.A, G.resid, G.how + f" (Grenze passt: {c0:.1f} px)"), c0


def georef_border(img, spec):
    """SLF Pro maps (LV95, north up): the national border's outermost pixels = Switzerland's extreme points."""
    x0, x1 = spec.get("x_range", [0, img.shape[1]])
    a = img[:, x0:x1].astype(int)
    g = a.mean(-1)
    sat = a.max(-1) - a.min(-1)
    lo, hi = spec.get("gray", [138, 166])
    m = (g >= lo) & (g <= hi) & (sat < 14)
    # the border is the long connected band; keep the largest component
    lab, n = ndi.label(m)
    if n == 0:
        raise SystemExit("border not found")
    big = 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    ys, xs = np.where(lab == big)
    t = spec.get("half_width", 4 * scale_of(img))
    W, E = xs.min() + x0 + t, xs.max() + x0 - t
    N, S = ys.min() + t, ys.max() - t
    e = {k: lv95(*v) for k, v in CH_EXTREMES.items()}
    ax = (E - W) / (e["east"][0] - e["west"][0])
    ay = (S - N) / (e["south"][1] - e["north"][1])
    A = np.array([[ax, 0], [0, ay], [W - ax * e["west"][0], N - ay * e["north"][1]]])
    return Georef("lv95", A, None, f"border extremes, band {lo}-{hi}, inset {t}px")


# ── colour reading ─────────────────────────────────────────────────────────
def classify(rgb, blended=False, tol=45):
    """RGB (n,3) -> palette index, -1 for grey/white (no colour), None if no class fits.
    blended: the map colours are drawn semi-transparent over a grey relief."""
    rgb = np.atleast_2d(np.asarray(rgb, float))
    sat = rgb.max(1) - rgb.min(1)
    out = np.full(len(rgb), -2, int)                # -2 = unknown
    grey = sat < 16
    out[grey] = -1
    idx = np.where(~grey)[0]
    if not len(idx):
        return out
    p = rgb[idx]
    if not blended:
        d = np.sqrt(((p[:, None, :] - PAL[None]) ** 2).sum(-1))
        k = d.argmin(1)
        ok = d[np.arange(len(k)), k] <= tol
        out[idx[ok]] = k[ok]
        return out
    best = np.full(len(p), 1e9)
    bk = np.full(len(p), -2)
    for a in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        for k, c in enumerate(PAL):
            if a == 1.0:
                pred = np.repeat(c[None], len(p), 0)
            else:
                gval = np.clip(((p - a * c) / (1 - a)).mean(1), 40, 255)
                pred = a * c + (1 - a) * gval[:, None]
            # relief shading darkens the whole pixel: compare after removing brightness
            s = (p.sum(1) / np.maximum(pred.sum(1), 1))[:, None]
            d = np.sqrt(((p - pred * s) ** 2).sum(1)) + 25 * np.abs(np.log(np.clip(s[:, 0], .2, 5)))
            better = d < best
            best[better], bk[better] = d[better], k
    ok = best <= tol
    out[idx[ok]] = bk[ok]
    return out


def disk_colour(img, x, y, r=3):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (r <= x < w - r and r <= y < h - r):
        return None
    return np.median(img[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3), 0)


# ── model access ───────────────────────────────────────────────────────────
class Models:
    def __init__(self, site, archive=None):
        from compare_newsnow import load_model
        from snowval.net import Polite
        self.net = Polite(delay=(.2, .5))
        self.site = site.rstrip("/") + "/"
        self.sm = load_model(self.site)
        self.va_base = self.site + "data/variant_a_live/"
        self.man = self.net.get(self.va_base + "manifest.json", light=True, cache=False).json()
        self.pack = self.net.get(self.va_base + "pack/index.json", light=True, cache=False).json()
        self._frames, self._layers = {}, {}
        self.ts = np.array([t.timestamp() for t in self.sm["times"]])
        self.cum = np.concatenate([np.zeros((1,) + self.sm["snow"].shape[1:], np.float32), np.cumsum(self.sm["snow"], 0)])
        wps = self.pack["wps"]
        self.wp_ll = np.array([[w["lat"], w["lon"]] for w in wps])
        # flat runs per weather point: band -> run index
        self.flat = {}
        for ri, (wp, band, sl, asp) in enumerate(self.pack["runs"]):
            if sl == 0 and asp == -1:
                self.flat.setdefault(wp, {})[band] = ri
        try:
            self.stations = self.net.get("https://measurement-api.slf.ch/public/api/imis/stations", light=True).json()
        except Exception:                                # noqa: BLE001
            self.stations = []
        self.used = set()                                # which sources answered (for the report)
        self.arc_runs, self.arc_va, self._npz = [], [], {}
        if archive and Path(archive).is_dir():
            for f in sorted(Path(archive).glob("snowmapper_*.npz")):
                with np.load(f) as z:
                    if "hourly" not in z.files or not bool(z["hourly"]):
                        continue
                    meta = json.loads(str(z["meta"]))
                    tt = [dt.datetime.fromisoformat(x) for x in z["times"][[0, -1]]]
                gen = dt.datetime.fromisoformat(meta["generated"].replace("Z", "+00:00"))
                self.arc_runs.append((gen, tt[0], tt[1], f))
            for f in sorted(Path(archive).glob("variant_a_*.npz")):
                with np.load(f) as z:
                    man = json.loads(str(z["manifest"]))
                    keys = set(z.files)
                gen = dt.datetime.strptime(man["generated"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
                self.arc_va.append((gen, f, man, keys))

    def _load(self, f):
        if f not in self._npz:
            if len(self._npz) > 4:
                self._npz.pop(next(iter(self._npz)))
            self._npz[f] = dict(np.load(f))
        return self._npz[f]

    def _arc_newsnow(self, lat, lon, t0, t1):
        cand = [(g, f) for g, a, b, f in self.arc_runs if a <= t0 and b >= t1]
        if not cand:
            return None
        after = [c for c in cand if c[0] >= t1]           # the first run after the window: past hours = analysis
        g, f = min(after) if after else max(cand)
        z = self._load(f)
        meta = json.loads(str(z["meta"]))
        la, lo = z["lat"], z["lon"]
        i, j = int(round((la[0] - lat) / abs(la[1] - la[0]))), int(round((lon - lo[0]) / abs(lo[1] - lo[0])))
        if not (0 <= i < len(la) and 0 <= j < len(lo)):
            return None
        ts = np.array([dt.datetime.fromisoformat(x).timestamp() for x in z["times"]])
        sel = (ts >= t0.timestamp()) & (ts < t1.timestamp())
        self.used.add(f"Archiv {meta['version']}")
        return float(z["h_SNOW"][sel, i, j].sum()) * meta["snow_scale"] * meta.get("ns_settle", 1.0)

    # Snowmapper new snow
    def cell(self, lat, lon):
        la, lo = self.sm["lat"], self.sm["lon"]
        i = int(round((la[0] - lat) / abs(la[1] - la[0])))
        j = int(round((lon - lo[0]) / abs(lo[1] - lo[0])))
        return (i, j) if 0 <= i < len(la) and 0 <= j < len(lo) else None

    def newsnow(self, lat, lon, t0, t1):
        ij = self.cell(lat, lon)
        if ij is None:
            return None
        a, b = np.searchsorted(self.ts, t0.timestamp()), np.searchsorted(self.ts, t1.timestamp())
        if b - a < 22:
            return self._arc_newsnow(lat, lon, t0, t1)
        self.used.add(f"live {self.sm['version']}")
        return float(self.cum[b][ij] - self.cum[a][ij])

    def elev(self, lat, lon):
        ij = self.cell(lat, lon)
        return float(self.sm["elev"][ij]) if ij else None

    # Variant A snow depth from the metric packs
    def _frame(self, t):
        tags = self.pack["tags"]
        tt = [dt.datetime.strptime(x, "%Y-%m-%dT%H%M").replace(tzinfo=UTC).timestamp() for x in tags]
        k = int(np.argmin(np.abs(np.array(tt) - t.timestamp())))
        if abs(tt[k] - t.timestamp()) > 2 * 3600:
            return None
        tag = tags[k]
        if tag not in self._frames:
            img = np.asarray(Image.open(io.BytesIO(self.net.get(self.va_base + self.pack["file"].format(tag=tag), light=True).content)).convert("RGB"))
            per = self.pack["px_per_run"]
            rows = img.shape[0]
            q = img.reshape(rows * self.pack["runs_per_row"], per * 3)[: len(self.pack["runs"])]
            self._frames[tag] = q
        return self._frames[tag]

    def va_hs(self, lat, lon, elev, t):
        if elev is None:
            return None
        q = self._frame(t)
        if q is not None:
            mi = self.pack["mets"].index("total_hs_cm")
            okc = len(self.pack["mets"])               # the ok flag follows the metrics
            hs = np.where(q[:, okc] > 0, q[:, mi] / self.pack["mul"][mi], np.nan)
            self.used.add(f"Variant A live {self.man.get('generated')}")
            return self._hs_interp(self.pack, self.flat, self.wp_ll, hs, lat, lon, elev)
        # archive: the newest cycle at or after t that has a frame within 6 h (06 / 18 UTC)
        best = None
        for g, f, man, keys in self.arc_va:
            for k in keys:
                if k.startswith("hs_"):
                    tt = dt.datetime.strptime(k[3:], "%Y-%m-%dT%H%M").replace(tzinfo=UTC)
                    dtm = abs((tt - t).total_seconds())
                    if dtm <= 6 * 3600 and (best is None or (dtm, -g.timestamp()) < best[0]):
                        best = ((dtm, -g.timestamp()), f, k)
        if best is None:
            return None
        z = self._load(best[1])
        pm = json.loads(str(z["pack_meta"]))
        mi = pm["mets"].index("total_hs_cm")
        raw = z[best[2]].astype(float)
        hs = np.where(raw == 255, np.nan, raw / pm["mul"][mi])
        flat = {}
        for ri, (wp, band, sl, asp) in enumerate(pm["runs"]):
            if sl == 0 and asp == -1:
                flat.setdefault(wp, {})[band] = ri
        self.used.add(f"Variant A Archiv {best[2][3:]}")
        return self._hs_interp(pm, flat, np.array([[w["lat"], w["lon"]] for w in pm["wps"]]), hs, lat, lon, elev)

    @staticmethod
    def _hs_interp(pack, flat_runs, wp_ll, hs, lat, lon, elev):
        """Flat-slope snow height at (lat, lon, elev): bands interpolated in height, 3 nearest weather points."""
        d = np.array([haversine(lat, lon, a, b) for a, b in wp_ll])
        near = np.argsort(d)[: pack.get("neighbours", 3)]
        d0 = pack.get("d0_m", 7500.0)
        vals, ws = [], []
        for wi in near:
            w = pack["wps"][wi]
            bands = flat_runs.get(int(wi), {})
            if not bands:
                continue
            el = np.array([w["b0"] + b * pack["elev_step"] for b in sorted(bands)])
            hv = np.array([hs[bands[b]] for b in sorted(bands)])
            okm = np.isfinite(hv)
            if not okm.any():
                continue
            v = float(np.interp(elev, el[okm], hv[okm]))
            vals.append(v)
            ws.append(1.0 / (d[wi] ** 2 + d0 ** 2))
        return float(np.average(vals, weights=ws)) if vals else None

    def va_class(self, layer, lat, lon, day):
        tag = f"{day.isoformat()}T1200"
        man = self.man
        if tag in self.man["tags"]:
            key = (layer, tag)
            if key not in self._layers:
                url = self.va_base + self.man["layers"][layer]["file"].format(tag=tag)
                self._layers[key] = np.asarray(Image.open(io.BytesIO(self.net.get(url, light=True).content)).convert("RGBA"))
            im = self._layers[key]
        else:                                            # archive: class rasters (legend index)
            k = f"{layer}_{day.isoformat()}"
            hit = [(g, f, m) for g, f, m, keys in self.arc_va if k in keys]
            if not hit:
                return None
            g, f, man = max(hit, key=lambda x: x[0])
            idx = self._load(f)[k]
            v = self._mode(idx, man["bounds"], lat, lon)
            return None if v is None else "none" if v == 0 else man["layers"][layer]["legend"].get(str(v), [None])[0]
        leg = man["layers"][layer]["legend"]
        # RGBA -> legend index once per raster, then the same neighbourhood vote as for the archive
        ikey = (layer, tag, "idx")
        if ikey not in self._layers:
            from archive_forecast import png_classes
            self._layers[ikey] = png_classes(im, leg)
        v = self._mode(self._layers[ikey], self.man["bounds"], lat, lon)
        return None if v is None else "none" if v == 0 else leg.get(str(v), [None])[0]

    @staticmethod
    def _mode(idx, bounds, lat, lon, radius_km=1.0):
        """Most frequent class within ~1 km (a point observation is not a pixel)."""
        (la0, lo0), (la1, lo1) = bounds
        h, w = idx.shape
        c, r = int((lon - lo0) / (lo1 - lo0) * w), int((la1 - lat) / (la1 - la0) * h)
        if not (0 <= r < h and 0 <= c < w):
            return None
        ry = max(1, int(round(radius_km / 111.32 / ((la1 - la0) / h))))
        rx = max(1, int(round(radius_km / (111.32 * math.cos(math.radians(lat))) / ((lo1 - lo0) / w))))
        win = idx[max(0, r - ry):r + ry + 1, max(0, c - rx):c + rx + 1].ravel()
        vals, cnt = np.unique(win, return_counts=True)
        return int(vals[cnt.argmax()])


def haversine(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


# ── per image ──────────────────────────────────────────────────────────────
def local(ts):
    t = dt.datetime.fromisoformat(ts)
    return (t.replace(tzinfo=ZRH) if t.tzinfo is None else t).astimezone(UTC)


def run_item(it, base: Path, M: Models, border, out: Path):
    img = np.asarray(Image.open(base / it["file"]).convert("RGB"))
    g = it["georef"]
    if "points" in g:
        pts, notes = [], []
        known = [q for q in g["points"] if q[2] in CITIES]
        if len(known) < 3:
            raise ValueError(f"mindestens 3 bekannte Orte nötig (bekannt: {', '.join(sorted(CITIES))}); "
                             f"im Manifest: {[q[2] for q in g['points']]}")
        for x, y, name in known:
            if g.get("refine", True):
                x, y, hit = refine_ring(img, x, y)
                notes.append(f"{name}{'' if hit else ' (nicht verfeinert)'}")
            else:
                notes.append(f"{name} (von Auge)")
            pts.append((x, y, *CITIES[name]))
        G = Georef.fit(it.get("proj", "merc"), pts, "places: " + ", ".join(notes))
    else:
        G = georef_border(img, g["border"])
    fit = g.get("fit_border", "border" in g)               # SLF Pro: always; others when the border is visible
    if fit:
        spec = g["border"] if "border" in g else (fit if isinstance(fit, dict) else {"gray": [60, 130], "max_sat": 20})
        G, _ = refine_on_border(G, img, spec)
    t1 = local(it["valid_local"])
    t0 = t1 - dt.timedelta(hours=it.get("window_h", 24))
    edges = LEGENDS.get(it.get("legend", ""), [])
    rows = []
    kind, var = it["kind"], it["var"]

    def model_value(lat, lon, elev=None):
        if var == "hn24":
            return M.newsnow(lat, lon, t0, t1)
        if var == "hs":
            return M.va_hs(lat, lon, elev if elev is not None else M.elev(lat, lon), t1)
        return None

    if kind == "stations":
        for s in M.stations:
            if s.get("type") and s["type"] != "SNOW_FLAT":
                continue                                  # wind stations measure no snow depth / new snow
            x, y = G.px(s["lat"], s["lon"])
            col = disk_colour(img, x, y, max(2, int(3 * scale_of(img))))
            if col is None:
                continue
            k = int(classify(col)[0])
            # WhiteRisk grey squares = no value at that station (they also sit on high
            # summits with snow), so they are skipped like background pixels
            if k < 0:
                continue
            mv = model_value(s["lat"], s["lon"], s.get("elevation"))
            rows.append({"id": s["code"], "label": s["label"], "lat": s["lat"], "lon": s["lon"], "elev": s.get("elevation"),
                         "px": round(x), "py": round(y), "shot_bin": k,
                         "model": mv, "model_bin": bin_of(mv, edges)})
    elif kind == "raster":
        ref = None
        if it.get("reference") == "slf_hn1d":
            from scrape_snowmaps import slf_encoded, slf_sample, SLF
            ref = slf_encoded(M.net, f"{SLF}/hn/map/HN1D/{t1:%Y-%m-%d}/encoded", f"{SLF}/hn/map/colorscale")
        la, lo, el = M.sm["lat"], M.sm["lon"], M.sm["elev"]
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        X, Y = G.px(LA.ravel(), LO.ravel())
        h, w = img.shape[:2]
        sel = np.where((X >= 4) & (X < w - 4) & (Y >= 4) & (Y < h - 4) & np.isfinite(el.ravel())
                       & in_switzerland(LA.ravel(), LO.ravel()))[0]
        cols = np.array([disk_colour(img, X[i], Y[i], 2) for i in sel])
        ks = classify(cols, blended=True)
        for n_, i in enumerate(sel):
            if ks[n_] == -2:
                continue
            lat, lon = float(LA.ravel()[i]), float(LO.ravel()[i])
            mv = model_value(lat, lon, float(el.ravel()[i]))
            r = {"id": f"c{i}", "lat": lat, "lon": lon, "elev": float(el.ravel()[i]), "px": round(X[i]), "py": round(Y[i]),
                 "shot_bin": int(ks[n_]), "model": mv, "model_bin": bin_of(mv, edges)}
            if ref is not None:
                rv = float(slf_sample(ref, np.array([lat]), np.array([lon]))[0])
                r["ref"], r["ref_bin"] = rv, bin_of(rv, edges)
            rows.append(r)
    elif kind in ("obs_points", "obs_symbols"):
        x0, x1 = it.get("x_range", [0, img.shape[1]])
        a = img[:, x0:x1].astype(int)
        sat = a.max(-1) - a.min(-1)
        if kind == "obs_points":
            # only the legend's colours: map lines (violet-blue) and buttons are not in the palette
            colored = (classify(a.reshape(-1, 3), tol=40) >= 0).reshape(sat.shape) & (sat > 40)
            hollow = (np.abs(a.mean(-1) - 125) < 25) & (sat < 12)  # grey rings
            masks = [("colour", colored), ("grey", hollow)]
        else:
            masks = [("blue", (a[..., 2] > 180) & (a[..., 0] < 60) & (a[..., 1] < 60)),
                     ("cyan", (a[..., 2] > 200) & (a[..., 1] > 150) & (a[..., 0] < 120)),
                     ("red", (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)),
                     ("black", a.max(-1) < 50)]
        for name, m in masks:
            lab, n = ndi.label(ndi.binary_closing(m, iterations=2))
            for i, sl in enumerate(ndi.find_objects(lab), 1):
                hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                k_ = scale_of(img)
                if not (14 * k_ <= hh <= 80 * k_ and 14 * k_ <= ww <= 80 * k_):
                    continue
                fill = (lab[sl] == i).mean()
                if kind == "obs_points" and (abs(hh - ww) > 0.3 * max(hh, ww)
                                             or (name == "colour" and fill < 0.5) or (name == "grey" and not 0.1 < fill < 0.8)):
                    continue                                 # not a round marker (map lines, labels)
                cy, cx = (sl[0].start + sl[0].stop) / 2, x0 + (sl[1].start + sl[1].stop) / 2
                lat, lon = G.latlon(cx, cy)
                if not (45.7 < lat < 47.9 and 5.9 < lon < 10.6):
                    continue
                r = {"id": f"{name}{i}", "lat": lat, "lon": lon, "px": round(cx), "py": round(cy), "symbol": name}
                if kind == "obs_points":
                    col = np.median(img[int(cy) - 3:int(cy) + 4, int(cx) - 3:int(cx) + 4].reshape(-1, 3), 0)
                    k = -1 if name == "grey" else int(classify(col, tol=60)[0])
                    if k == -2:
                        continue
                    mv = model_value(lat, lon)
                    r.update({"shot_bin": k, "model": mv, "model_bin": bin_of(mv, edges)})
                else:
                    day = t1.astimezone(ZRH).date()
                    cls = (it.get("symbol_legend") or {}).get(name)
                    r.update({"reported": cls or f"? ({name})", "va_ski6": M.va_class("ski6", lat, lon, day),
                              "va_ski18": M.va_class("ski18", lat, lon, day), "va_wind": M.va_class("wind", lat, lon, day)})
                rows.append(r)
    # overlay for checking the georeferencing
    ov = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(ov)
    for ring in BORDER:
        X, Y = G.px(ring[:, 0], ring[:, 1])
        d.line(list(zip(X.tolist(), Y.tolist())), fill=(255, 0, 255), width=max(2, int(2 * scale_of(img))))
    for r in rows[:: max(1, len(rows) // 3000)]:
        d.ellipse((r["px"] - 4, r["py"] - 4, r["px"] + 4, r["py"] + 4), outline=(255, 0, 255), width=2)
    ov.thumbnail((1400, 1400))
    ov_path = out / f"overlay_{Path(it['file']).stem}.png"
    ov.save(ov_path)
    return G, rows, ov_path, (t0, t1), edges


def summarize(rows, edges):
    pairs = [(r["shot_bin"], r["model_bin"]) for r in rows if r.get("model_bin") is not None and r.get("shot_bin") is not None]
    if not pairs:
        return None
    s = np.array([p[0] for p in pairs])
    m = np.array([p[1] for p in pairs])
    n = len(pairs)
    mids = [(edges[i] + edges[i + 1]) / 2 if i + 1 < len(edges) else edges[i] * 1.2 for i in range(len(edges))]
    mid = lambda k: 0.0 if k < 0 else mids[k]   # noqa: E731
    return {"n": n, "same_class": float((s == m).mean()), "within_one": float((np.abs(s - m) <= 1).mean()),
            "model_higher": float((m > s).mean()), "model_lower": float((m < s).mean()),
            "bias_mid_cm": float(np.mean([mid(b) - mid(a) for a, b in pairs])),
            "confusion": {f"{bin_label(a, edges)} → {bin_label(b, edges)}": int(((s == a) & (m == b)).sum())
                          for a in sorted(set(s)) for b in sorted(set(m)) if ((s == a) & (m == b)).any()}}


KIND_HINTS = [  # filename words -> manifest defaults (rename files to get these for free)
    (("hn24_stations", "hn_stations", "neuschnee_stationen"), {"kind": "stations", "var": "hn24", "legend": "wr_hn", "window_h": 24}),
    (("hs_stations", "schneehoehe_stationen"), {"kind": "stations", "var": "hs", "legend": "wr_hs"}),
    (("hn24_map", "hn_map", "neuschnee_karte"), {"kind": "raster", "var": "hn24", "legend": "wr_hn", "window_h": 24,
                                                  "reference": "slf_hn1d"}),
    (("hs_map", "schneehoehe_karte"), {"kind": "raster", "var": "hs", "legend": "wr_hs"}),
    (("hn_observed", "neuschnee_beobachtet"), {"kind": "obs_points", "var": "hn24", "legend": "wr_hn", "window_h": 24, "proj": "lv95"}),
    (("surface", "oberflaeche"), {"kind": "obs_symbols", "var": "surface", "proj": "lv95", "symbol_legend": {}}),
]


def set_dir(name):
    p = Path(name)
    return p if p.is_dir() else INBOX / name


def cmd_init(a):
    d = set_dir(a.set)
    d.mkdir(parents=True, exist_ok=True)
    mf = d / "manifest.json"
    items = json.loads(mf.read_text()) if mf.exists() else []
    have = {it["file"] for it in items}
    added = 0
    for f in sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))):
        if f.name in have:
            continue
        img = np.asarray(Image.open(f).convert("RGB"))
        it = {"file": f.name, "title": "", "kind": "TODO stations|raster|obs_points|obs_symbols", "var": "TODO hn24|hs|surface",
              "legend": "wr_hn|wr_hs", "valid_local": "TODO 2026-12-31T18:30 (Schweizer Zeit) oder ...Z für UTC", "proj": "merc"}
        for words, defaults in KIND_HINTS:
            if any(w in f.stem.lower() for w in words):
                it.update(defaults)
                break
        if it.get("proj") == "lv95":
            w = img.shape[1]
            it["x_range"] = [int(w * .37), int(w * .89)]
            it["georef"] = {"border": {"x_range": it["x_range"], "gray": [138, 166]}}
        else:
            rings = find_rings(img)
            it["georef"] = {"points": [[x, y, "TODO Ortsname"] for x, y in rings]}
            it["_hint"] = (f"{len(rings)} Ortsmarker gefunden: Namen eintragen (bekannt: {', '.join(sorted(CITIES))}); "
                           "falsche Einträge löschen. Ohne Marker: 3+ Orte von Auge mit \"refine\": false.")
        items.append(it)
        added += 1
    mf.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{mf}: {added} neue Bilder, {len(items)} insgesamt. TODO-Felder ausfüllen, dann: run {a.set}")


def cmd_run(a):
    if not a.site:
        sys.exit("set MODEL_SITE (or --site)")
    d = set_dir(a.set)
    mf = d / "manifest.json"
    items = [it for it in json.loads(mf.read_text()) if not any("TODO" in str(it.get(k, "")) for k in ("kind", "var", "valid_local"))]
    out = Path(a.out) if a.out else RESULTS / d.name
    out.mkdir(parents=True, exist_ok=True)
    M = Models(a.site, a.archive)
    L = ["# Screenshots gegen Snowmapper", "",
         f"Modell-Lauf {M.sm['version']} (App-Layer: {M.sm['times'][0]:%d.%m.} – {M.sm['times'][-1]:%d.%m.%Y}) · "
         f"Variant A {M.man.get('generated')} (Packs {M.pack['tags'][0]} – {M.pack['tags'][-1]})", "",
         "Klassen = Farbstufen der Karte. «gleich» = Modell in derselben Stufe, «±1» = höchstens eine Stufe daneben. "
         "Bias = Differenz der Stufenmitten (Modell − Screenshot).", ""]
    allres = {}
    for it in items:
        try:
            G, rows, ov, (t0, t1), edges = run_item(it, d, M, None, out)
        except Exception as e:                           # noqa: BLE001
            L += [f"## {it['file']}", "", f"Fehler: `{type(e).__name__}: {e}`", ""]
            continue
        s = summarize(rows, edges) if edges else None
        allres[it["file"]] = {"summary": s, "georef": G.how, "residual_px": None if G.resid is None else G.resid.round(1).tolist(),
                              "rows": rows}
        L += [f"## {it.get('title', it['file'])}", "",
              f"`{it['file']}` · {it['kind']} · {it['var']} · Zeitraum {t0.astimezone(ZRH):%d.%m. %H:%M} – {t1.astimezone(ZRH):%d.%m. %H:%M} "
              f"(Schweizer Zeit) · Georeferenz: {G.how}"
              + (f", Restfehler {', '.join(f'{v:.1f}' for v in G.resid)} px" if G.resid is not None else ""), "",
              f"Kontrollbild: `{ov.name}` (magenta: Landesgrenze und abgelesene Punkte – sie müssen auf der Karte liegen)", ""]
        refp = [(r["shot_bin"], r["ref_bin"]) for r in rows if r.get("ref_bin") is not None]
        if refp:
            a_ = np.array(refp)
            snow = a_[:, 1] >= 0
            L += [f"**Selbsttest Ablesen:** Screenshot-Stufe gegen die exakten SLF-Werte derselben Karte: "
                  f"{(a_[:, 0] == a_[:, 1]).mean():.0%} gleiche Stufe, {(np.abs(a_[:, 0] - a_[:, 1]) <= 1).mean():.0%} ±1 "
                  f"(nur Zellen mit Schnee: {(a_[snow, 0] == a_[snow, 1]).mean():.0%} gleich, n={int(snow.sum())}).", ""]
            allres[it["file"]]["reference_check"] = {"same": float((a_[:, 0] == a_[:, 1]).mean()), "n": len(refp)}
        if s:
            L += ["| n | gleiche Stufe | ±1 Stufe | Modell höher | Modell tiefer | Bias (Stufenmitten) |", "|---:|---:|---:|---:|---:|---:|",
                  f"| {s['n']} | {s['same_class']:.0%} | {s['within_one']:.0%} | {s['model_higher']:.0%} | {s['model_lower']:.0%} | "
                  f"{s['bias_mid_cm']:+.0f} cm |", "",
                  "Häufigste Paare (Screenshot → Modell): " + ", ".join(
                      f"{k}: {v}" for k, v in sorted(s["confusion"].items(), key=lambda kv: -kv[1])[:8]), ""]
            if it["kind"] == "stations":
                worst = sorted([r for r in rows if r.get("model_bin") is not None],
                               key=lambda r: -abs(r["model_bin"] - r["shot_bin"]))[:10]
                L += ["| Station | Höhe | Screenshot | Modell |", "|---|---:|---|---:|"]
                L += [f"| {r['id']} {r['label']} | {r.get('elev') or '–'} | {bin_label(r['shot_bin'], edges)} | {r['model']:.0f} cm |" for r in worst]
                L.append("")
        elif it["kind"] == "obs_symbols":
            def coarse6(c):
                if not c or c == "none":
                    return c
                return "powder" if c.startswith("powder") else {"hard": "hard", "crust": "crust", "corn": "corn_wet", "wet": "corn_wet",
                                                                "thin_cover": "thin_none", "loose_faceted": "powder"}.get(c, c)
            leg = it.get("symbol_legend") or {}
            L += ["| Symbol | gemeldet (grob) | Ort | Variant A ski6 | ski18 | Wind | passt (ski6) |", "|---|---|---|---|---|---|---|"]
            hits = []
            for r in rows:
                rep_c = (leg.get(r["symbol"]) or {}).get("coarse") if isinstance(leg.get(r["symbol"]), dict) else None
                ok = None if not rep_c else (coarse6(r["va_ski6"]) == rep_c)
                if ok is not None:
                    hits.append(ok)
                L.append(f"| {r['symbol']} | {rep_c or '?'} | {r['lat']:.3f}, {r['lon']:.3f} | {r['va_ski6']} | {r['va_ski18']} | "
                         f"{r['va_wind']} | {'–' if ok is None else '✓' if ok else '✕'} |")
            L.append("")
            if hits:
                L += [f"Treffer ski6: {sum(hits)}/{len(hits)}.", ""]
            else:
                L += ["Die Symbole werden erkannt, aber ihre Bedeutung fehlt: Legende im Manifest unter `symbol_legend` eintragen, "
                      "z. B. `\"cyan\": {\"name\": \"Pulver\", \"coarse\": \"powder\"}` (grob: powder, wind, crust, hard, corn_wet, thin_none).", ""]
            dis = sum(1 for r in rows if r["va_ski6"] and r["va_ski18"] and coarse6(r["va_ski6"]) == "powder"
                      and ("crust" in str(r["va_ski18"]) or "hard" in str(r["va_ski18"])))
            if dis:
                L += [f"Hinweis: an {dis} Orten zeigt Variant A in ski6 Pulver, in ski18 aber Kruste/hart – die beiden Layer widersprechen sich.", ""]
        else:
            L += ["Keine vergleichbaren Werte (Modell deckt den Zeitraum nicht ab oder nichts abgelesen).", ""]
    L += ["", "Modellquellen: " + ", ".join(sorted(M.used)) if M.used else "Modellquellen: –",
          "Landesgrenze: © swisstopo (swissBOUNDARIES3D)."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "results.json").write_text(json.dumps(allres, indent=1, default=str))
    print("\n".join(L))




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("init", help="manifest skeleton for the images in screencheck/inbox/<set>")
    i.add_argument("set")
    r = sp.add_parser("run", help="check the images of a set against the model")
    r.add_argument("set")
    r.add_argument("--out", help="default screencheck/results/<set>")
    r.add_argument("--site", default=os.environ.get("MODEL_SITE", ""))
    r.add_argument("--archive", help="folder with archive_forecast.py files, for days the live model no longer covers")
    a = ap.parse_args()
    {"init": cmd_init, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
