#!/usr/bin/env python3
"""Snow-surface model trained on LLM labels from gipfelbuch reports.

  labels   data/labels/gipfelbuch_surfaces.csv   (scrape_gipfelbuch.py --llm / label_reports.py)
  features the forecast archive (archive_forecast.py): Snowmapper daily fields of
           the days before the tour at the report's 1 km cell, temperatures moved
           to the label's height, and Variant A's surface / wind / powder class there
  target   coarse surface: powder · wind · crust · hard · corn_wet · thin_none

    python surface_model.py dataset --data data --archive archive     -> data/training/surface.csv
    python surface_model.py train   --data data --out models/surface   -> model.joblib + report.json/.md
    python surface_model.py predict --archive archive --model models/surface --date 2026-12-20 --out pred

Honest by construction:
  * test days come strictly after training days (no leakage between neighbouring
    reports of the same storm), and the report says how many there were;
  * the model is compared with the rule it should beat: Variant A's own class
    mapped to the same six classes;
  * with fewer than --min-rows labelled surfaces it refuses to train (--force
    to try anyway) -- a model on 40 reports would just memorise them.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

CLASSES = ["powder", "wind", "crust", "hard", "corn_wet", "thin_none"]
VA_SKI6 = {"hard": "hard", "crust": "crust", "corn": "corn_wet", "wet": "corn_wet", "thin_cover": "thin_none",
           "loose_faceted": "powder"}                   # powder_* -> powder
LAPSE = -0.0065
ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
FEATURES = ["elev_m", "cell_elev", "dz", "aspect_sin", "aspect_cos", "aspect_known", "lat", "lon", "doy_sin", "doy_cos",
            "hn24", "hn72", "prec24", "prec72", "days_since_snow", "abl24", "abl72",
            "tmax_prev", "tmin_prev", "tmean_prev", "tprec_prev", "tmax_day", "tmin_day",
            "wind_max_prev", "wind_mean_prev", "wind_max_day", "sun_prev", "sun_day", "cloud_day",
            "va_ski6", "va_ski18", "va_wind", "va_powder"]
CATEGORICAL = ["va_ski6", "va_ski18", "va_wind", "va_powder"]


# ── archive access ──────────────────────────────────────────────────────────
class Archive:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.runs, self.va = [], []
        for f in sorted(self.dir.glob("snowmapper_*.npz")):
            with np.load(f) as z:
                meta = json.loads(str(z["meta"]))
                ends = [dt.datetime.fromisoformat(x) for x in z["window_end"]]
            gen = dt.datetime.fromisoformat((meta.get("generated") or "").replace("Z", "+00:00"))
            self.runs.append((gen, f, ends))
        for f in sorted(self.dir.glob("variant_a_*.npz")):
            with np.load(f) as z:
                man = json.loads(str(z["manifest"]))
                keys = list(z.files)
            gen = dt.datetime.strptime(man["generated"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
            self.va.append((gen, f, man, keys))
        self._cache = {}

    def _load(self, f):
        if f not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            self._cache[f] = dict(np.load(f))
        return self._cache[f]

    def run_for(self, day: dt.date):
        """Newest run generated before noon of the tour day that covers the 3 nights before and the day itself."""
        noon = dt.datetime.combine(day, dt.time(12), dt.timezone.utc)
        need = [dt.datetime.combine(day, dt.time(6), dt.timezone.utc) + dt.timedelta(days=k) for k in (-2, -1, 0, 1)]
        ok = [(g, f) for g, f, ends in self.runs if g <= noon and all(n in ends for n in need)]
        return self._load(max(ok)[1]) if ok else None

    def va_for(self, day: dt.date):
        key = f"ski6_{day.isoformat()}"
        lim = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0), dt.timezone.utc)
        ok = [(g, f, man) for g, f, man, keys in self.va if key in keys and g <= lim]
        if not ok:
            return None, None
        g, f, man = max(ok, key=lambda x: x[0])
        return self._load(f), man


def _cell(z, lat, lon):
    la, lo = z["lat"], z["lon"]
    i = int(round((la[0] - lat) / abs(la[1] - la[0])))
    j = int(round((lon - lo[0]) / abs(lo[1] - lo[0])))
    return (i, j) if 0 <= i < len(la) and 0 <= j < len(lo) else None


def _va_class(va, man, layer, day, lat, lon):
    k = f"{layer}_{day.isoformat()}"
    if va is None or k not in va:
        return -1
    (la0, lo0), (la1, lo1) = man["bounds"]
    a = va[k]
    h, w = a.shape
    c, r = int((lon - lo0) / (lo1 - lo0) * w), int((la1 - lat) / (la1 - la0) * h)
    return int(a[r, c]) if 0 <= r < h and 0 <= c < w else -1


def features_at(arc, day, lat, lon, elev_m=None, aspect=None):
    z = arc.run_for(day)
    if z is None:
        return None
    ij = _cell(z, lat, lon)
    if ij is None:
        return None
    ends = [dt.datetime.fromisoformat(x) for x in z["window_end"]]
    d0 = dt.datetime.combine(day, dt.time(6), dt.timezone.utc)
    w = {k: ends.index(d0 + dt.timedelta(days=k)) for k in (-2, -1, 0, 1)}

    def g(name, k):
        v = z["d_" + name][w[k]][ij]
        return None if v == -32768 else float(v) / 10

    cell_elev = float(z["elev"][ij])
    elev = float(elev_m) if elev_m is not None else cell_elev
    dz = elev - cell_elev
    t = lambda name, k: (None if g(name, k) is None else g(name, k) + LAPSE * dz)  # noqa: E731
    snow = [g("snow", k) or 0.0 for k in (0, -1, -2)]
    # days since the last 24 h with >= 1 cm, looking back as far as the run reaches
    dss, k = 0, 0
    while True:
        e = d0 - dt.timedelta(days=k)
        if e not in ends:
            dss = 7 if dss == k else dss
            break
        v = z["d_snow"][ends.index(e)][ij] / 10
        if v >= 1:
            break
        k += 1
        dss = k
    asp = ASPECT_DEG.get(aspect) if aspect else None
    doy = day.timetuple().tm_yday
    va, man = arc.va_for(day)
    return {"elev_m": elev, "cell_elev": cell_elev, "dz": dz,
            "aspect_sin": math.sin(math.radians(asp)) if asp is not None else 0.0,
            "aspect_cos": math.cos(math.radians(asp)) if asp is not None else 0.0,
            "aspect_known": 1.0 if asp is not None else 0.0, "lat": lat, "lon": lon,
            "doy_sin": math.sin(2 * math.pi * doy / 365), "doy_cos": math.cos(2 * math.pi * doy / 365),
            "hn24": snow[0], "hn72": sum(snow), "prec24": g("prec", 0), "prec72": sum((g("prec", k) or 0) for k in (0, -1, -2)),
            "days_since_snow": dss, "abl24": g("abl", 0), "abl72": sum((g("abl", k) or 0) for k in (0, -1, -2)),
            "tmax_prev": t("t_max", 0), "tmin_prev": t("t_min", 0), "tmean_prev": t("t_mean", 0), "tprec_prev": t("t_prec", 0),
            "tmax_day": t("t_max", 1), "tmin_day": t("t_min", 1),
            "wind_max_prev": g("wind_max", 0), "wind_mean_prev": g("wind_mean", 0), "wind_max_day": g("wind_max", 1),
            "sun_prev": g("sun", 0), "sun_day": g("sun", 1), "cloud_day": g("cloud", 1),
            "va_ski6": _va_class(va, man, "ski6", day, lat, lon), "va_ski18": _va_class(va, man, "ski18", day, lat, lon),
            "va_wind": _va_class(va, man, "wind", day, lat, lon), "va_powder": _va_class(va, man, "powder", day, lat, lon),
            "_va_ski6_name": (man["layers"]["ski6"]["legend"].get(str(_va_class(va, man, "ski6", day, lat, lon)), [None])[0]
                              if man else None)}


def va_rule(name):
    """Variant A's ski6 class mapped to the coarse classes (the baseline to beat)."""
    if not name:
        return None
    return "powder" if name.startswith("powder") else VA_SKI6.get(name)


# ── dataset ─────────────────────────────────────────────────────────────────
def cmd_dataset(a):
    arc = Archive(a.archive)
    src = Path(a.data, "labels", "gipfelbuch_surfaces.csv")
    if not src.exists():
        sys.exit(f"no labels at {src}")
    rows, skipped = [], {"no_elevation": 0, "no_class": 0, "low_confidence": 0, "no_archive": 0}
    with open(src, newline="") as fh:
        labels = list(csv.DictReader(fh))
    for r in labels:
        if r["coarse"] not in CLASSES:
            skipped["no_class"] += 1
            continue
        if r["confidence"] == "low" and not a.keep_low:
            skipped["low_confidence"] += 1
            continue
        if not r["elev_m"]:
            skipped["no_elevation"] += 1
            continue
        day = dt.date.fromisoformat(r["date"])
        f = features_at(arc, day, float(r["lat"]), float(r["lon"]), float(r["elev_m"]), r["aspect"] if r["aspect"] in ASPECT_DEG else None)
        if f is None:
            skipped["no_archive"] += 1
            continue
        rows.append({"report_id": r["report_id"], "seg": r["seg"], "date": r["date"], "y": r["coarse"],
                     "confidence": r["confidence"], "va_rule": va_rule(f.pop("_va_ski6_name")) or "", **f})
    out = Path(a.data, "training", "surface.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["report_id", "seg", "date", "y", "confidence", "va_rule"] + FEATURES)
        w.writeheader()
        w.writerows(rows)
    by = {c: sum(1 for x in rows if x["y"] == c) for c in CLASSES}
    print(f"{len(rows)} training rows from {len(labels)} labels -> {out}")
    print(f"  per class: {by}")
    print(f"  skipped: {skipped}")


# ── train ───────────────────────────────────────────────────────────────────
def _xy(rows):
    X = np.array([[float(r[k]) if r[k] not in ("", "None") else np.nan for k in FEATURES] for r in rows])
    y = np.array([r["y"] for r in rows])
    return X, y


def _metrics(y, p, labels):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    ok = np.array([q is not None and q != "" for q in p])
    if ok.sum() == 0:
        return None
    y, p = y[ok], np.array(p)[ok]
    return {"n": int(ok.sum()), "accuracy": float((y == p).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
            "confusion": confusion_matrix(y, p, labels=labels).tolist()}


def cmd_train(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    src = Path(a.data, "training", "surface.csv")
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["y"] for r in rows})
    if (len(rows) < a.min_rows or len(classes) < 2) and not a.force:
        sys.exit(f"only {len(rows)} labelled surfaces in {len(classes)} classes (need {a.min_rows}); "
                 "collect more or pass --force")
    days = sorted({r["date"] for r in rows})
    cut = days[int(len(days) * (1 - a.test_share))] if len(days) > 3 else days[-1]
    tr = [r for r in rows if r["date"] < cut]
    te = [r for r in rows if r["date"] >= cut]
    if not tr or not te:
        sys.exit("not enough distinct days for a train/test split by date")
    Xtr, ytr = _xy(tr)
    Xte, yte = _xy(te)
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    # categorical features must be non-negative ints: shift the "missing" -1 to the top code
    for X in (Xtr, Xte):
        for c in cat:
            X[:, c] = np.where(X[:, c] < 0, 254, X[:, c])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                                         class_weight="balanced", categorical_features=cat, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels = [c for c in CLASSES if c in set(ytr) | set(yte)]
    rep = {"trained": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "n_train": len(tr), "n_test": len(te),
           "train_days": [days[0], tr[-1]["date"] if tr else None], "test_from": cut, "classes": labels,
           "model": _metrics(yte, list(pred), labels),
           "baseline_variant_a": _metrics(yte, [r["va_rule"] for r in te], labels),
           "baseline_majority": _metrics(yte, [max(set(ytr), key=list(ytr).count)] * len(te), labels)}
    # which inputs matter (permutation importance on the test days)
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(clf, Xte, yte, n_repeats=8, random_state=0, scoring="balanced_accuracy")
        rep["importance"] = sorted(((FEATURES[i], float(pi.importances_mean[i])) for i in range(len(FEATURES))),
                                   key=lambda x: -x[1])[:15]
    except Exception as e:                               # noqa: BLE001
        rep["importance_error"] = str(e)
    # final model on all rows for use
    Xall, yall = _xy(rows)
    for c in cat:
        Xall[:, c] = np.where(Xall[:, c] < 0, 254, Xall[:, c])
    clf.fit(Xall, yall)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES, "categorical": CATEGORICAL}, out / "model.joblib")
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    L = [f"# Oberflächen-Modell ({rep['trained']})", "",
         f"Training {len(tr)} Flächen ({days[0]} – {tr[-1]['date']}), Test {len(te)} Flächen ab {cut} (spätere Tage).", "",
         "| | n | Trefferquote | ausgewogen | Makro-F1 |", "|---|---:|---:|---:|---:|"]
    for k, nm in (("model", "Modell"), ("baseline_variant_a", "Variant A (Regel)"), ("baseline_majority", "häufigste Klasse")):
        m = rep[k]
        L.append(f"| {nm} | {m['n'] if m else 0} | {m['accuracy']:.0%} | {m['balanced_accuracy']:.0%} | {m['macro_f1']:.2f} |"
                 if m else f"| {nm} | 0 | – | – | – |")
    L += ["", "Verwechslung Modell (Zeilen = gemeldet): " + ", ".join(labels), ""]
    if rep["model"]:
        L += ["| | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
        L += [f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(rep["model"]["confusion"])]
    if rep.get("importance"):
        L += ["", "Wichtigste Eingaben: " + ", ".join(f"{k} ({v:+.3f})" for k, v in rep["importance"][:8])]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ── predict ─────────────────────────────────────────────────────────────────
def cmd_predict(a):
    import joblib
    arc = Archive(a.archive)
    day = dt.date.fromisoformat(a.date)
    z = arc.run_for(day)
    if z is None:
        sys.exit(f"no archived run covers {day}")
    m = joblib.load(Path(a.model, "model.joblib"))
    la, lo, el = z["lat"], z["lon"], z["elev"]
    asp = z["aspect"] if z["aspect"].size else None
    grid = np.full(el.shape, 255, np.uint8)
    rows, idx = [], []
    for i in range(0, len(la), a.step):
        for j in range(0, len(lo), a.step):
            if not np.isfinite(el[i, j]) or el[i, j] < a.min_elev:
                continue
            aspect = None
            if asp is not None:
                aspect = list(ASPECT_DEG)[int(((asp[i, j] + 22.5) % 360) // 45)]
            f = features_at(arc, day, float(la[i]), float(lo[j]), None, aspect)
            if f is None:
                continue
            f.pop("_va_ski6_name", None)
            rows.append([np.nan if f[k] is None else f[k] for k in m["features"]])
            idx.append((i, j))
    if not rows:
        sys.exit("no cells")
    X = np.array(rows, float)
    for c in m["categorical"]:
        k = m["features"].index(c)
        X[:, k] = np.where(X[:, k] < 0, 254, X[:, k])
    p = m["model"].predict(X)
    for (i, j), c in zip(idx, p):
        grid[i, j] = CLASSES.index(c)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"surface_{day.isoformat()}.npz"
    np.savez_compressed(f, classes=np.array(CLASSES), grid=grid, lat=la, lon=lo, step=a.step)
    counts = {CLASSES[k]: int((grid == k).sum()) for k in range(len(CLASSES))}
    print(f"{f}: {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dataset")
    d.add_argument("--data", default="data")
    d.add_argument("--archive", default="archive")
    d.add_argument("--keep-low", action="store_true", help="also labels the LLM marked low-confidence")
    t = sp.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="models/surface")
    t.add_argument("--min-rows", type=int, default=300)
    t.add_argument("--test-share", type=float, default=0.25)
    t.add_argument("--force", action="store_true")
    p = sp.add_parser("predict")
    p.add_argument("--archive", default="archive")
    p.add_argument("--model", default="models/surface")
    p.add_argument("--date", required=True)
    p.add_argument("--out", default="pred")
    p.add_argument("--step", type=int, default=1, help="every n-th cell (faster previews)")
    p.add_argument("--min-elev", type=float, default=1000)
    a = ap.parse_args()
    {"dataset": cmd_dataset, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
