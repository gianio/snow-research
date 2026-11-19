#!/usr/bin/env python3
"""New condition reports from gipfelbuch.ch -> where, when, what the snow was like.

Finds new reports through the site's own sitemaps (sitemap_conditions*.xml
with <lastmod>), so only reports not seen before are opened. For each one:

  report     id, url, date of the tour, activity (Skitour, Hochtour, …),
             summit, summit elevation, route, lat/lon (the route's map point),
             the quick info (snow above / below, avalanche perception,
             overall rating, goal reached), description and outlook text
  observations (to the observation store, source "gipfelbuch"):
             surface_top / surface_bottom   e.g. Pulver, Sulz, Harsch (class)
             snowline_m      "Schnee ab 1800 m", "ab 2000 m durchgehend" …
             aper_below_m    "bis 1600 m aper"
             ski_from_m      "Ski ab 1500 m", "Skidepot auf 2900 m" …
             snow_cm         numbers in cm, with a guess whether new / total
             avalanche_perception, rating (class)

Without login the site shows only 3 reports per visit, then asks to log in.
This tool does not get around that: set GIPFELBUCH_EMAIL / GIPFELBUCH_PASSWORD
(your own free account) to log in, otherwise it reads 3 reports per run and
stops as soon as a page comes back without the report.

Nothing about the author is stored (no name, no profile). Texts stay in
your own data folder -- keep the repository private; they are the
authors' content.

    python scrape_gipfelbuch.py --data data [--since-days 3] [--types snow|all|skitour,hochtour]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path

from snowval import llm_labels, store
from snowval.net import Polite

BASE = "https://www.gipfelbuch.ch"
SITEMAP = BASE + "/data/sitemaps/sitemap.xml"
# the URL slug usually starts with the activity, but not always (then it starts
# with the summit name). Default "snow": read everything except activities that
# rarely say anything about snow -- hikers' "Schnee ab 1800 m" is valuable.
SKIP_FOR_SNOW = {"klettertour", "sportklettern", "mehrseillaengen", "mountainbike", "mtb", "velotour", "bouldern"}
SURFACES = ["Pulver", "Pressschnee", "Triebschnee", "Sulz", "Firn", "Harsch", "Bruchharsch", "Nassschnee", "Faulschnee",
            "Hart", "Eis", "Bruchharst", "Windgepresst", "Sastrugi", "Schwer", "Weich", "Apere Stellen", "Kein Schnee"]


def text(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t]+", " ", html.unescape(s)).strip()


# ── finding new reports ─────────────────────────────────────────────────────
def new_report_urls(net, since: dt.datetime, seen: set[int]):
    idx = net.get(SITEMAP, cache=False).text
    maps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx) if "conditions" in u]
    found = []
    for sm in maps:
        x = net.get(sm, light=True, cache=False).text
        for loc, lastmod in re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", x):
            m = re.search(r"/verhaeltnisse/(\d+)-([a-z0-9-]+)", loc)
            if not m:
                continue
            t = dt.datetime.fromisoformat(lastmod)
            rid = int(m.group(1))
            if t >= since and rid not in seen:
                found.append((t, rid, m.group(2), loc))
    found.sort()
    return found


def activity_of(slug):
    return slug.split("-", 1)[0]


# ── parsing one report ──────────────────────────────────────────────────────
def parse_report(page: str, url: str):
    r = {"url": url, "id": int(re.search(r"/verhaeltnisse/(\d+)", url).group(1))}
    m = re.search(r"Verhältnisse vom (\d{2})\.(\d{2})\.(\d{4})", page)
    r["date"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
    m = re.search(r'<div class="page-title">.*?<h2>(?:<a[^>]*>)?(.*?)(?:</a>)?</h2>', page, re.S)
    title = text(m.group(1)) if m else ""
    r["title"] = title
    m = re.match(r"(.*?)\s*\((\d{3,4})\s*m\)\s*:?\s*(.*)", title)
    r["summit"], r["summit_elev"], r["route"] = (m.group(1), int(m.group(2)), m.group(3)) if m else (title, None, None)
    m = re.search(r'<div class="page-title">.*?<a href="(/routen/\d[^"]+)"', page, re.S)
    r["route_url"] = BASE + m.group(1) if m else None
    tags = re.findall(r'<div class="title-tags[^"]*">(.*?)</div>', page, re.S)
    # the activity tag is the one whose icon is not the rating star
    acts = [text(sp) for sp in re.findall(r"<span>(.*?)</span>", tags[0], re.S) if "#star" not in sp] if tags else []
    r["activity"] = acts[0] if acts else None
    m = re.search(r'id="map-container-inner" data-lat="([\d.]+)" data-lon="([\d.]+)"', page) \
        or re.search(r'data-lat="([\d.]+)" data-lon="([\d.]+)"', page)
    r["lat"], r["lon"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    # quick info bar: icon name -> text
    qi = {}
    bar = re.search(r'<div class="quickinfo-bar">(.*?)</div></div><div class="label-group', page, re.S)
    for icon, val in re.findall(r'#([a-z0-9-]+)"></use></svg></i><span>(.*?)</span>', bar.group(1) if bar else "", re.S):
        qi[icon] = text(val)
    r["quickinfo"] = qi
    snow = qi.get("schnee", "")
    r["surface_top"] = (re.search(r"Oben:\s*([^\n]+)", snow) or [None, None])[1]
    r["surface_bottom"] = (re.search(r"Unten:\s*([^\n]+)", snow) or [None, None])[1]
    r["avalanche_perception"] = qi.get("lawinen")
    r["goal"] = qi.get("ziel-erreicht") or qi.get("ziel-nicht-erreicht")
    r["rating"] = next((v for k, v in qi.items() if k.startswith("smiley")), None)
    r["group_size"] = qi.get("gruppe")
    # labelled text blocks
    for lab, val in re.findall(r"<label>([^<]+)</label>\s*<span>(.*?)</span>", page, re.S):
        key = {"Verhältnis-Beschreibung": "description", "Ausblick": "outlook"}.get(lab.strip(), lab.strip())
        r[key] = text(val)
    r["n_images"] = len(re.findall(r'class="image-box grp-', page))
    return r


# ── what the text says about snow ───────────────────────────────────────────
NUM = r"(\d{3,4})(?:\s*(?:m|m\.ü\.M\.|müM|hm)\b)"
PAT_SNOWLINE = [
    rf"(?:Schnee|schneebedeckt|eingeschneit|durchgehend|zusammenhängend|geschlossene Schneedecke|Schneegrenze)[^.;\n]{{0,40}}?\bab\s*(?:ca\.?\s*|etwa\s*|rund\s*)?{NUM}",
    rf"\bab\s*(?:ca\.?\s*|etwa\s*|rund\s*)?{NUM}[^.;\n]{{0,30}}?(?:Schnee|schneebedeckt|durchgehend|eingeschneit|weiss)",
    rf"Schneegrenze[^.;\n]{{0,20}}?(?:bei|auf|um)?\s*(?:ca\.?\s*)?{NUM}",
]
PAT_APER = [rf"\b(?:bis|unterhalb(?: von)?)\s*(?:ca\.?\s*)?{NUM}[^.;\n]{{0,25}}?\baper", rf"\baper\b[^.;\n]{{0,25}}?\bbis\s*(?:ca\.?\s*)?{NUM}"]
PAT_SKI = [rf"\bSki(?:er)?\s*(?:ab|von)\s*(?:ca\.?\s*)?{NUM}", rf"\b(?:Ski|Skis)\s+(?:angeschnallt|an)\s+(?:ab|auf|bei)\s*(?:ca\.?\s*)?{NUM}",
           rf"\bab\s*(?:ca\.?\s*)?{NUM}[^.;\n]{{0,20}}?\bmit (?:den )?Ski"]
PAT_DEPOT = [rf"\bSkidepot[^.;\n]{{0,20}}?(?:auf|bei|ca\.?)\s*{NUM}"]
PAT_CM = r"(\d{1,3})(?:\s*(?:-|–|bis)\s*(\d{1,3}))?\s*cm\b"
NEW_WORDS = r"Neuschnee|frisch|gefallen|neue[nm]? Schnee|Pulver|über Nacht|geschneit|Zuwachs"
TOTAL_WORDS = r"Schneehöhe|Gesamtschnee|Schneedecke|liegt|liegen|Unterlage|Altschnee|insgesamt"


def _nearest(t, m, groups, before=45, after=30):
    """The keyword group closest to the match (same sentence), else 'unknown'."""
    best, kind = 1e9, "unknown"
    lo = max(t.rfind(".", 0, m.start()) + 1, m.start() - before)
    hi = min([x for x in (t.find(".", m.end()), m.end() + after) if x >= 0])
    for k, pat in groups.items():
        for w in re.finditer(pat, t[lo:hi], re.I):
            a = lo + w.start()
            d = a - m.end() if a >= m.end() else m.start() - (lo + w.end())
            if d < best:
                best, kind = d, k
    return kind


def snow_facts(r):
    t = " ".join(filter(None, [r.get("description"), r.get("outlook")]))
    t = re.sub(r"\b(ca|bzw|evtl|ev|max|min|rund)\.", r"\1", t)     # abbreviations are no sentence end
    facts = []

    def add(var, value, unit, snippet):
        facts.append({"var": var, "value": value, "unit": unit, "snippet": snippet.strip()[:160]})

    for var, pats in (("snowline_m", PAT_SNOWLINE), ("aper_below_m", PAT_APER), ("ski_from_m", PAT_SKI),
                      ("skidepot_m", PAT_DEPOT)):
        got = set()
        for p in pats:
            for m in re.finditer(p, t, re.I):
                v = int(m.group(1))
                if 300 <= v <= 4800 and v not in got:
                    got.add(v)
                    add(var, v, "m", t[max(0, m.start() - 40):m.end() + 20])
    for m in re.finditer(PAT_CM, t):
        a, b = int(m.group(1)), m.group(2) and int(m.group(2))
        v = (a + b) / 2 if b else a
        ctx = t[max(0, m.start() - 70):m.end() + 40]
        kind = _nearest(t, m, {"new": NEW_WORDS, "total": TOTAL_WORDS})
        add(f"snow_cm_{kind}", v, "cm", ctx)
    return facts


def to_obs(r, with_text=True):
    """Rows for the observation store from one parsed report."""
    if r["lat"] is None or not r["date"]:
        return {}
    base = {"kind": "obs", "lat": r["lat"], "lon": r["lon"], "valid_from": r["date"] + "T06:00:00Z",
            "valid_to": r["date"] + "T16:00:00Z", "point_id": str(r["id"]), "ref": r["url"]}
    out: dict[str, list] = {}

    def put(var, value, unit, elev=None, note=None):
        out.setdefault(var, []).append({**base, "elev": elev, "value": value, "unit": unit,
                                        "ref": r["url"] + (f" | {note}" if note else "")})

    if r.get("surface_top"):
        put("surface_top", r["surface_top"], "class", r.get("summit_elev"))
    if r.get("surface_bottom"):
        put("surface_bottom", r["surface_bottom"], "class")
    if r.get("avalanche_perception"):
        put("avalanche_perception", r["avalanche_perception"], "class", r.get("summit_elev"))
    if r.get("rating"):
        put("rating", r["rating"], "class", r.get("summit_elev"))
    for f in r.get("facts") or snow_facts(r):
        # an elevation statement is a value AT that height: elev = value
        elev = f["value"] if f["unit"] == "m" else r.get("summit_elev")
        put(f["var"], f["value"], f["unit"], elev, f["snippet"].replace("\n", " ") if with_text and f.get("snippet") else None)
    return out


# ── access ──────────────────────────────────────────────────────────────────
def gated(page):
    """True if the page is the login prompt instead of the report."""
    return "quickinfo-bar" not in page and "Verhältnis-Beschreibung" not in page


def login(net, email, password):
    form = net.get(BASE + "/meta/login", cache=False).text
    data = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form))
    data.update({"email_login": email, "password_login": password, "dvForm_subm": "1"})
    net.post(BASE + "/meta/login", data, referer=BASE + "/")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--since-days", type=float, default=3, help="reports changed in the last N days (first run)")
    ap.add_argument("--types", default="snow", help="snow | all | comma list of slug prefixes (skitour,hochtour,wanderung,…)")
    ap.add_argument("--max", type=int, default=300, help="at most this many reports per run")
    ap.add_argument("--delay", default="3,8", help="min,max seconds between report pages")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-text", action="store_true",
                    help="store only the extracted facts, not the authors' texts (use this in a public repository)")
    ap.add_argument("--llm", default="auto", choices=["auto", "live", "fake", "off"],
                    help="label each report with Claude (auto = if ANTHROPIC_API_KEY is set); fake = offline rules")
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.delay.split(","))
    net = Polite(delay=(lo, hi), cache_dir=a.cache, cache_ttl_h=24 * 30)
    L = llm_labels.labeler(a.llm, keep_evidence=not a.no_text)

    gdir = Path(a.data) / "gipfelbuch"
    gdir.mkdir(parents=True, exist_ok=True)
    state_f = gdir / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    seen = set(state.get("seen", []))
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=a.since_days)
    if state.get("last_run"):
        since = min(since, dt.datetime.fromisoformat(state["last_run"]) - dt.timedelta(hours=6))

    cand = new_report_urls(net, since, seen)
    if a.types == "all":
        todo = cand
    elif a.types == "snow":
        todo = [c for c in cand if activity_of(c[2]) not in SKIP_FOR_SNOW]
    else:
        want = set(a.types.split(","))
        todo = [c for c in cand if activity_of(c[2]) in want]
    todo = todo[-a.max:]                                # newest first if capped
    print(f"{len(cand)} new/changed reports since {since:%Y-%m-%d %H:%M}, {len(todo)} to read ({a.types})")

    email, pw = os.environ.get("GIPFELBUCH_EMAIL"), os.environ.get("GIPFELBUCH_PASSWORD")
    if email and pw:
        login(net, email, pw)
    else:
        todo = todo[-3:]
        print("  no GIPFELBUCH_EMAIL / GIPFELBUCH_PASSWORD: reading only the 3 newest (the site's limit without login)")

    reports, obs, label_rows, n_err = [], {}, [], 0
    for t, rid, slug, url in todo:
        try:
            page = net.get(url, referer=BASE + "/verhaeltnisse/uebersicht", cache=False).text
            if gated(page):
                print(f"  {rid}: the site asks for a login -- stopping here ({'login failed?' if email else 'set the account variables'})")
                break
            r = parse_report(page, url)
        except Exception as e:                          # noqa: BLE001
            print(f"  {rid}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        r["lastmod"] = t.isoformat()
        r["facts"] = snow_facts(r)
        if L is not None:                               # needs the text: before it is dropped below
            lab = L(r)
            r["llm"] = lab
            if "error" in lab:
                n_err += 1
                print(f"  {rid}: LLM {lab['error']}", file=sys.stderr)
            else:
                label_rows += llm_labels.label_rows(r, lab)
        if a.no_text:
            for k in ("description", "outlook", "Besonderes/Persönliches"):
                r.pop(k, None)
            for f in r["facts"]:
                f.pop("snippet", None)
        reports.append(r)
        seen.add(rid)
        for var, rows in to_obs(r, with_text=not a.no_text).items():
            obs.setdefault((var, r["date"]), []).extend(rows)
        print(f"  {r['date']} {r.get('activity') or activity_of(slug):<14} {r['summit']} ({r['summit_elev']} m)"
              f"  top={r.get('surface_top')}  facts={len(r['facts'])}")

    # reports: one JSON line each, a file per month of the tour date
    for r in reports:
        f = gdir / f"reports_{(r['date'] or 'unknown')[:7]}.jsonl"
        old = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        old = [o for o in old if o["id"] != r["id"]] + [r]
        f.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in sorted(old, key=lambda o: o["id"])))
    # observations: merged into the day's file per variable
    for (var, day), rows in obs.items():
        old = store.load(a.data, "gipfelbuch", var, since=day, until=day)
        ids = {r["point_id"] for r in rows}
        store.write(a.data, "gipfelbuch", var, day, [o for o in old if o["point_id"] not in ids] + rows)

    if label_rows:
        llm_labels.write_label_rows(a.data, label_rows)
    if L is not None:
        print(f"LLM labels: {len(label_rows)} surfaces from {len(reports) - n_err} reports, {n_err} errors, usage {L.usage}")

    state["seen"] = sorted(seen)[-20000:]
    state["last_run"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state_f.write_text(json.dumps(state))
    print(f"{len(reports)} reports, {sum(len(v) for v in obs.values())} observations, {net.n_requests} requests")


if __name__ == "__main__":
    main()
