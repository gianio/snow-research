"""Observation store: one row = one value at one coordinate and time.

Files are append-only, gzip CSV, one per source / variable / run day:
    <data>/obs/<source>/<var>/<YYYY-MM-DD>.csv.gz
That keeps git diffs small and lets a day be re-collected by overwriting
one file. `load()` reads them back (optionally filtered) and `to_sqlite()`
builds one SQLite table for ad-hoc queries.

Columns
  source       slf_hn, slf_hs, imis, gipfelbuch …
  var          hn24 (new snow 24 h, cm), hs (snow depth, cm), snowline_m, …
  kind         obs (measured / reported) | analysis | forecast
  lat, lon     WGS84
  elev         m a.s.l. of the point if the source gives one, else empty
  valid_from   ISO UTC, start of the period (= valid_to for a state like hs)
  valid_to     ISO UTC, end of the period / time of the state
  issued       ISO UTC, run / publication time (forecasts), else empty
  lead_h       hours from issued to valid_to (forecasts), else empty
  value        number (or a short text for categorical vars like surface)
  unit         cm, m, class …
  point_id     station code, grid id, report id …
  ref          URL or short note where the value came from
"""
from __future__ import annotations

import csv
import gzip
import sqlite3
from pathlib import Path

COLS = ["source", "var", "kind", "lat", "lon", "elev", "valid_from", "valid_to", "issued", "lead_h",
        "value", "unit", "point_id", "ref"]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.5f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.0f}"
    return str(v)


def write(data_dir, source, var, day, rows, mode="w"):
    """rows: iterable of dicts with (a subset of) COLS. day: 'YYYY-MM-DD'.
    mode 'w' replaces the day's file, 'a' appends (deduplicated on load)."""
    f = Path(data_dir) / "obs" / source / var / f"{day}.csv.gz"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return None
    new = not f.exists() or mode == "w"
    with gzip.open(f, "wt" if mode == "w" else "at", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        for r in rows:
            r = {"source": source, "var": var, **r}
            w.writerow([_fmt(r.get(c)) for c in COLS])
    return f


def load(data_dir, source=None, var=None, since=None, until=None):
    """All rows as dicts (value as float when numeric). since/until: 'YYYY-MM-DD' on the file day."""
    base = Path(data_dir) / "obs"
    out, seen = [], set()
    for f in sorted(base.glob(f"{source or '*'}/{var or '*'}/*.csv.gz")):
        d = f.name[:10]
        if (since and d < since) or (until and d > until):
            continue
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                k = (r["source"], r["var"], r["point_id"], r["lat"], r["lon"], r["valid_to"], r["issued"])
                if k in seen:
                    continue
                seen.add(k)
                for c in ("lat", "lon", "elev", "lead_h", "value"):
                    try:
                        r[c] = float(r[c]) if r[c] != "" else None
                    except ValueError:
                        pass                       # categorical value stays text
                out.append(r)
    return out


def to_sqlite(data_dir, db_path):
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS obs")
    con.execute("CREATE TABLE obs (source TEXT, var TEXT, kind TEXT, lat REAL, lon REAL, elev REAL, valid_from TEXT, "
                "valid_to TEXT, issued TEXT, lead_h REAL, value, unit TEXT, point_id TEXT, ref TEXT)")
    rows = load(data_dir)
    con.executemany(f"INSERT INTO obs VALUES ({','.join('?' * len(COLS))})", [[r[c] for c in COLS] for r in rows])
    con.execute("CREATE INDEX obs_q ON obs (source, var, valid_to)")
    con.execute("CREATE INDEX obs_ll ON obs (lat, lon)")
    con.commit()
    con.close()
    return len(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build a SQLite file from the collected observations")
    ap.add_argument("--data", default="data")
    ap.add_argument("--db", default="data/obs.sqlite")
    a = ap.parse_args()
    print(f"{to_sqlite(a.data, a.db)} rows -> {a.db}")
