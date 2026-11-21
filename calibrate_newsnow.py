#!/usr/bin/env python3
"""What-if calibration of the Snowmapper new-snow layer against SLF HN1D.

The model file carries, per 1 km cell and hour, the precipitation at the
weather reference height (PREC), the cell temperature (TEMP) and the
resulting new snow (SNOW). Snowmapper computes

    SNOW = PREC * phase(T) * SLR(T) / 10 * M          M = altitude x orographic x wind factor

so M can be recovered per cell and hour (M = SNOW / (PREC*phase*SLR/10)).
That lets us recompute the last days' new snow with other parameters --
rain/snow threshold, snow-to-liquid ratio, a damped terrain factor M**alpha
-- and score each variant against what the SLF actually measured (HN1D,
24 h to 06 UTC) for every day the model file covers in the past.

To keep it honest, the "best" variant is chosen leave-one-day-out: the
parameters picked on the other days are scored on the day left out.

    MODEL_SITE=https://… python calibrate_newsnow.py --out calib
    python calibrate_newsnow.py --sm-file snowdata.json.gz --days 5

Writes calib.json (all variants, per day and pooled) and prints a table.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import itertools
import json
import os
from pathlib import Path

import numpy as np

from compare_newsnow import blocks, load_model, load_slf, sample_slf

UTC = dt.timezone.utc

# the grid that is searched; the current parameters are always added
GRID = {
    "t_center": [-0.5, 0.0, 0.5, 1.0, 1.5],
    "slr_max": [7.0, 8.0, 9.0, 11.0, 13.0, 15.0],
    "alpha": [0.0, 1.0],                       # terrain factor M**alpha (1 = as now, 0 = none)
    "pscale": [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],  # factor on the precipitation input
}


def cube(sm, key, mul, off=0.0):
    m, b = sm["meta"], sm["blob"]
    T, H, W = m["T"], m["height"], m["width"]
    raw = np.frombuffer(base64.b64decode(b[key]), np.uint8)[:T * H * W].reshape(T, H, W)
    return raw.astype(np.float32) / mul - off


def phase(T, t_center, k):
    return 1.0 / (1.0 + np.exp(np.clip(k * (T - t_center), -50, 50)))


def slr(T, i, s, lo, hi):
    return np.clip(i + s * T, lo, hi)


class Decomp:
    """Hourly PREC, TEMP and the terrain multiplier M for the model's past hours."""

    def __init__(self, sm):
        m = sm["meta"]
        self.sm = sm
        self.nsp = m.get("nsp") or {}
        self.settle = sm["settle"]
        self.times = sm["times"]
        now = m.get("now_index", m.get("today_index", len(self.times)))
        self.past = slice(0, now)
        self.P = cube(sm, "PREC", m.get("prec_mul", 5.0))[self.past]
        self.T = cube(sm, "TEMP", m.get("temp_mul", 2.0), m.get("temp_off", 60.0))[self.past]
        S = sm["snow"][self.past] / self.settle               # load_model applied the settling
        base = self.base(self.cur["t_center"], self.cur["k"], self.cur["slr_i"], self.cur["slr_s"],
                         self.cur["slr_min"], self.cur["slr_max"])
        with np.errstate(all="ignore"):
            M = np.where(base > 0.05, S / base, np.nan)
        # hours with almost no snow: use the cell's typical factor
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            typ = np.nanmedian(M, axis=0)
        typ = np.where(np.isfinite(typ), typ, 1.0)
        self.M = np.where(np.isfinite(M), np.clip(M, 0.05, 6.0), typ[None]).astype(np.float32)
        self.S = S
        self.M_typ = typ

    @property
    def cur(self):
        n = self.nsp
        return {"t_center": n.get("t_center", 1.0), "k": n.get("k", 1.4), "slr_i": n.get("slr_i", 10.0),
                "slr_s": n.get("slr_s", -0.5), "slr_min": n.get("slr_min", 6.0), "slr_max": n.get("slr_max", 15.0)}

    def base(self, tc, k, si, ss, smin, smax, sl=slice(None)):
        P, T = self.P[sl], self.T[sl]
        return P * phase(T, tc, k) * slr(T, si, ss, smin, smax) / 10.0

    def window(self, t0, t1):
        idx = [i for i, t in enumerate(self.times[self.past]) if t0 <= t < t1]
        if len(idx) < 22:
            return None
        return slice(idx[0], idx[-1] + 1)

    def newsnow(self, sl, t_center, slr_max, alpha, scale=1.0):
        c = self.cur
        b = self.base(t_center, c["k"], c["slr_i"], c["slr_s"], c["slr_min"], slr_max, sl)
        return (b * self.M[sl] ** alpha).sum(0) * self.settle * scale


def score(ref, mod, k=10):
    ok = np.isfinite(ref) & np.isfinite(mod)
    if ok.sum() < 50:
        return None
    rb, mb = blocks(np.where(ok, ref, np.nan), k), blocks(np.where(ok, mod, np.nan), k)
    okb = np.isfinite(rb) & np.isfinite(mb)
    d = mod[ok] - ref[ok]
    hit = lambda th: ((ref[ok] >= th) & (mod[ok] >= th)).sum()   # noqa: E731
    csi10 = hit(10) / max(1, ((ref[ok] >= 10) | (mod[ok] >= 10)).sum())
    return {"n": int(ok.sum()), "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "mae_10km": float(np.abs(mb[okb] - rb[okb]).mean()), "ratio": float(mod[ok].sum() / max(1e-6, ref[ok].sum())),
            "csi10": float(csi10), "mean_ref": float(ref[ok].mean()), "mean_mod": float(mod[ok].mean())}


def run(sm, days=5, log=print):
    D = Decomp(sm)
    lat, lon = sm["lat"], sm["lon"]
    last = max(t for t in D.times[D.past])
    day_list = []
    d = last.date()
    while len(day_list) < days and d > (last - dt.timedelta(days=days + 2)).date():
        day_list.append(d)
        d -= dt.timedelta(days=1)
    obs = {}
    for d in sorted(day_list):
        try:
            slf = load_slf(d.isoformat())
        except Exception as e:                           # noqa: BLE001
            log(f"  SLF HN1D {d}: {e}")
            continue
        t1 = slf["valid"]
        sl = D.window(t1 - dt.timedelta(hours=24), t1)
        if sl is None:
            continue
        obs[d.isoformat()] = (sample_slf(slf, lat, lon), sl)
    if not obs:
        raise SystemExit("no SLF day inside the model's past hours")
    c = D.cur
    variants = {"aktuell": (c["t_center"], c["slr_max"], 1.0, 1.0)}
    for tc, sm_, a, ps in itertools.product(GRID["t_center"], GRID["slr_max"], GRID["alpha"], GRID["pscale"]):
        variants[f"tc{tc:g}_slr{sm_:g}_a{a:g}_p{ps:g}"] = (tc, sm_, a, ps)
    res = {name: {} for name in variants}
    for day, (ref, sl) in obs.items():
        for name, (tc, sm_, a, ps) in variants.items():
            res[name][day] = score(ref, D.newsnow(sl, tc, sm_, a, ps))
    # pooled score (mean MAE over days) and leave-one-day-out choice
    def pooled(name, days_):
        s = [res[name][d] for d in days_ if res[name].get(d)]
        return np.mean([x["mae_10km"] for x in s]) if s else np.inf
    days_ = sorted(obs)
    best = min(variants, key=lambda n: pooled(n, days_))
    loo = []
    for d in days_:
        others = [x for x in days_ if x != d] or days_
        pick = min(variants, key=lambda n: pooled(n, others))
        loo.append({"day": d, "picked": pick, "held_out": res[pick][d], "current": res["aktuell"][d]})
    # a single global factor on the current model, for scale
    refs = np.concatenate([o[0].ravel() for o in obs.values()])
    mods = np.concatenate([D.newsnow(o[1], c["t_center"], c["slr_max"], 1.0, 1.0).ravel() for o in obs.values()])
    ok = np.isfinite(refs) & np.isfinite(mods)
    k_scale = float((refs[ok] * mods[ok]).sum() / max(1e-6, (mods[ok] ** 2).sum()))
    # how the error depends on height, current vs best
    elev = sm["elev"]
    bands = [(0, 1500), (1500, 2000), (2000, 2500), (2500, 5000)]
    by_elev = {}
    for name in ("aktuell", best):
        tc, sm_, a, ps = variants[name]
        rows = []
        for lo_, hi_ in bands:
            r_all, m_all = [], []
            for ref, sl in obs.values():
                mm = D.newsnow(sl, tc, sm_, a, ps)
                sel = (elev >= lo_) & (elev < hi_) & np.isfinite(ref)
                r_all.append(ref[sel]); m_all.append(mm[sel])
            r_all, m_all = np.concatenate(r_all), np.concatenate(m_all)
            rows.append({"band": f"{lo_}–{hi_} m", "n": int(r_all.size),
                         "mean_ref": float(r_all.mean()) if r_all.size else None,
                         "mean_mod": float(m_all.mean()) if m_all.size else None})
        by_elev[name] = rows
    return {"days": days_, "current_params": c, "variants": {n: list(v) for n, v in variants.items()},
            "scores": res, "best_pooled": best, "leave_one_day_out": loo, "global_factor": k_scale,
            "by_elev": by_elev, "terrain_factor_median": float(np.nanmedian(D.M_typ)),
            "model_version": sm["version"]}


def table(r):
    days = r["days"]
    def pooled(n):
        s = [r["scores"][n][d] for d in days if r["scores"][n].get(d)]
        return {k: float(np.mean([x[k] for x in s])) for k in ("bias", "mae", "mae_10km", "ratio", "csi10")} if s else None
    ranked = sorted((n for n in r["scores"] if pooled(n)), key=lambda n: pooled(n)["mae_10km"])
    keep = ["aktuell"] + [n for n in ranked if n != "aktuell"][:8]
    L = ["| Variante | Regen/Schnee-Mitte °C | SLR max | Geländefaktor^α | Niederschlag × | Bias cm | MAE cm | MAE 10 km | Modell/SLF | CSI ≥10 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in keep:
        p = pooled(n)
        tc, sm_, a, ps = r["variants"][n]
        L.append(f"| {'**' + n + '**' if n == 'aktuell' else n} | {tc:g} | {sm_:g} | {a:g} | {ps:g} | {p['bias']:+.1f} | {p['mae']:.1f} | "
                 f"{p['mae_10km']:.1f} | {p['ratio']:.2f} | {p['csi10']:.2f} |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("MODEL_SITE", ""))
    ap.add_argument("--sm-file")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--out", default="calib")
    a = ap.parse_args()
    if not (a.site or a.sm_file):
        raise SystemExit("set MODEL_SITE (or pass --site / --sm-file)")
    sm = load_model(a.site.rstrip("/") + "/" if a.site else a.site, a.sm_file)
    r = run(sm, a.days)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps(r, indent=1, default=str))
    print(f"Tage: {', '.join(r['days'])}")
    print("\n".join(table(r)))
    print(f"\nbeste Variante (alle Tage): {r['best_pooled']} · globaler Faktor auf das aktuelle Modell: {r['global_factor']:.2f}")
    for x in r["leave_one_day_out"]:
        if x["held_out"] and x["current"]:
            print(f"  {x['day']}: gewählt {x['picked']:<22} MAE {x['held_out']['mae']:.1f} (aktuell {x['current']['mae']:.1f})")


if __name__ == "__main__":
    main()
