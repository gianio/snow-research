#!/usr/bin/env python3
"""Label stored gipfelbuch reports with Claude in bulk (Message Batches API).

For the backlog (historic reports, or after a new prompt version). New reports
are labelled live by scrape_gipfelbuch.py. Needs the report texts, so this
only works on reports stored with text (private repository / local run).

    python label_reports.py submit  --data data [--max 5000] [--relabel]
    python label_reports.py collect --data data          # after the batch ended (usually < 1 h)
    python label_reports.py status  --data data
    python label_reports.py review  --data data [--n 200]   # sample to check by hand
    python label_reports.py score   --data data             # accuracy of the checked sample

Batches cost half of live requests. A rough guide for planning: about 1 500
input and 400 output tokens per report; check the actual usage that
`collect` prints before labelling thousands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from snowval import llm_labels


def load_reports(data):
    out = {}
    for f in sorted(Path(data, "gipfelbuch").glob("reports_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[r["id"]] = (r, f)
    return out


def save_reports(reports):
    by_file = {}
    for r, f in reports.values():
        by_file.setdefault(f, []).append(r)
    for f, rs in by_file.items():
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rs, key=lambda r: r["id"])))


def state_file(data):
    return Path(data, "labels", "batches.json")


def submit(a):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reps = load_reports(a.data)
    todo = [r for r, _ in reps.values()
            if r.get("description") and (a.relabel or "error" in (r.get("llm") or {"error": 1})
                                         or (r.get("llm") or {}).get("_prompt") != llm_labels.PROMPT_VERSION)]
    todo = todo[: a.max]
    if not todo:
        print("nothing to label (reports without text cannot be labelled)")
        return
    reqs = [Request(custom_id=str(r["id"]), params=MessageCreateParamsNonStreaming(**llm_labels.request_params(r)))
            for r in todo]
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    sf = state_file(a.data)
    sf.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(sf.read_text()) if sf.exists() else {"batches": []}
    st["batches"].append({"id": batch.id, "n": len(reqs), "prompt": llm_labels.PROMPT_VERSION, "model": llm_labels.MODEL,
                          "collected": False})
    sf.write_text(json.dumps(st, indent=1))
    print(f"batch {batch.id}: {len(reqs)} reports submitted ({batch.processing_status})")


def status(a):
    import anthropic
    sf = state_file(a.data)
    if not sf.exists():
        print("no batches")
        return
    c = anthropic.Anthropic()
    for b in json.loads(sf.read_text())["batches"]:
        x = c.messages.batches.retrieve(b["id"])
        print(f"{b['id']} n={b['n']} {x.processing_status} ok={x.request_counts.succeeded} "
              f"err={x.request_counts.errored} collected={b['collected']}")


def collect(a):
    import anthropic
    sf = state_file(a.data)
    if not sf.exists():
        print("no batches")
        return
    st = json.loads(sf.read_text())
    c = anthropic.Anthropic()
    reps = load_reports(a.data)
    rows, usage, n_ok, n_err = [], {"input": 0, "output": 0}, 0, 0
    for b in st["batches"]:
        if b["collected"]:
            continue
        if c.messages.batches.retrieve(b["id"]).processing_status != "ended":
            print(f"{b['id']} still running")
            continue
        for res in c.messages.batches.results(b["id"]):
            rid = int(res.custom_id)
            if rid not in reps:
                continue
            r, _ = reps[rid]
            if res.result.type != "succeeded":
                r["llm"] = {"error": res.result.type}
                n_err += 1
                continue
            msg = res.result.message
            usage["input"] += msg.usage.input_tokens or 0
            usage["output"] += msg.usage.output_tokens or 0
            lab = llm_labels.parse_message(msg, keep_evidence=not a.no_text)
            r["llm"] = lab
            if "error" in lab:
                n_err += 1
            else:
                n_ok += 1
                rows += llm_labels.label_rows(r, lab)
        b["collected"] = True
    save_reports(reps)
    if rows:
        llm_labels.write_label_rows(a.data, rows)
    sf.write_text(json.dumps(st, indent=1))
    print(f"{n_ok} reports labelled, {len(rows)} surfaces, {n_err} errors, tokens {usage}")


def review(a):
    """Write a random sample of labelled surfaces to check by hand (column 'correct': y / n / the right class)."""
    import csv
    import random
    reps = load_reports(a.data)
    cand = []
    for r, _ in reps.values():
        lab = r.get("llm") or {}
        for i, srf in enumerate(lab.get("surfaces") or []):
            cand.append({"report_id": r["id"], "seg": i, "url": r["url"], "date": r.get("date"), "summit": r.get("summit"),
                         "band": srf.get("band"), "aspect": srf.get("aspect"), "surface": srf.get("surface"),
                         "coarse": srf.get("coarse"), "confidence": srf.get("confidence"),
                         "evidence": srf.get("evidence", ""), "quickinfo_snow": (r.get("quickinfo") or {}).get("schnee", ""),
                         "correct": ""})
    random.Random(7).shuffle(cand)
    f = Path(a.data, "labels", "review_sample.csv")
    f.parent.mkdir(parents=True, exist_ok=True)
    keep = {}
    if f.exists():                                   # keep what was already checked
        with open(f, newline="") as fh:
            keep = {(x["report_id"], x["seg"]): x for x in csv.DictReader(fh) if x["correct"]}
    rows = list(keep.values()) + [c for c in cand if (str(c["report_id"]), str(c["seg"])) not in keep][: max(0, a.n - len(keep))]
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand[0]) if cand else ["report_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} surfaces in {f} ({len(keep)} already checked). Fill 'correct' with y, n or the right class, then run score.")


def score(a):
    import csv
    from collections import Counter
    f = Path(a.data, "labels", "review_sample.csv")
    with open(f, newline="") as fh:
        rows = [x for x in csv.DictReader(fh) if x["correct"].strip()]
    if not rows:
        sys.exit("nothing checked yet")
    ok = [x for x in rows if x["correct"].strip().lower() in ("y", "j", "ja", "yes", x["surface"], x["coarse"])]
    print(f"{len(ok)}/{len(rows)} correct ({len(ok) / len(rows):.0%})")
    for k in ("confidence", "coarse"):
        tot, good = Counter(x[k] for x in rows), Counter(x[k] for x in ok)
        print(f"  by {k}: " + ", ".join(f"{v}: {good[v]}/{n}" for v, n in tot.most_common()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["submit", "collect", "status", "review", "score"])
    ap.add_argument("--n", type=int, default=200, help="review: sample size")
    ap.add_argument("--data", default="data")
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--relabel", action="store_true", help="also reports that already have a label")
    ap.add_argument("--no-text", action="store_true", help="do not keep evidence quotes in the labels")
    a = ap.parse_args()
    try:
        {"submit": submit, "collect": collect, "status": status, "review": review, "score": score}[a.cmd](a)
    except ImportError:
        sys.exit("pip install anthropic")


if __name__ == "__main__":
    main()
