"""LLM labels for gipfelbuch condition reports.

A report says things like "unten noch braun, ab dem Sattel durchgehend Pulver,
in den Nordhängen etwas windgepresst". Regexes miss most of that. Claude turns
each report into a fixed JSON structure: one entry per elevation band / aspect
with a surface class, plus snow line, new snow, crusts, wind effects and
avalanche signs, each with how sure it is.

Two ways to run it:
  live   one request per new report (daily pipeline); refusals fall back
         server-side to another model ("fallbacks": "default")
  batch  the Message Batches API for a backlog (half the price, results
         within ~1 h); see label_reports.py

The labels never contain the author's name. With keep_evidence=False they
contain no text from the report either (for a public repository).
"""
from __future__ import annotations

import json
import os


MODEL = os.environ.get("LLM_MODEL", "claude-opus-5-5")
PROMPT_VERSION = "gb-surface-v1"

SURFACE_CLASSES = [
    "powder",             # Pulver, locker, nicht windbeeinflusst
    "wind_affected",      # Pulver mit Windeinfluss, teils gepresst
    "wind_slab_drift",    # Triebschnee, frische Ablagerungen
    "wind_packed",        # Pressschnee, windgepresst, verblasen, Sastrugi
    "crust_carrying",     # tragender Harsch / Schmelzharsch
    "crust_breaking",     # Bruchharsch
    "corn",               # Sulz, aufgefirnt (gut)
    "wet_heavy",          # Nassschnee, Faulschnee, schwer, sumpfig
    "hard_icy",           # hart, eisig, gefroren
    "groomed",            # Piste
    "thin_rocky",         # wenig Schnee, steinig, apere Stellen
    "no_snow",            # aper
    "other",
]
COARSE = {  # what the snow-type model predicts
    "powder": "powder", "wind_affected": "wind", "wind_slab_drift": "wind", "wind_packed": "wind",
    "crust_carrying": "crust", "crust_breaking": "crust", "hard_icy": "hard", "groomed": "hard",
    "corn": "corn_wet", "wet_heavy": "corn_wet", "thin_rocky": "thin_none", "no_snow": "thin_none", "other": None,
}
BANDS = ["summit_upper", "middle", "lower", "whole_route", "unknown"]
ASPECTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "all", "unknown"]
CONF = ["low", "medium", "high"]


def _nul(t):
    return {"anyOf": [{"type": t}, {"type": "null"}]}


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["snow_relevant", "surfaces", "snowline_m", "continuous_snow_from_m", "ski_from_m", "skidepot_m",
                 "new_snow_cm", "new_snow_days", "total_snow_cm", "total_snow_elev_m", "wind_effects",
                 "avalanche_signs", "crust_thickness_cm", "confidence"],
    "properties": {
        "snow_relevant": {"type": "boolean", "description": "Does the report say anything about snow at all?"},
        "surfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["band", "elev_from_m", "elev_to_m", "aspect", "surface", "ski_quality", "confidence", "evidence"],
                "properties": {
                    "band": {"type": "string", "enum": BANDS},
                    "elev_from_m": _nul("integer"),
                    "elev_to_m": _nul("integer"),
                    "aspect": {"type": "string", "enum": ASPECTS},
                    "surface": {"type": "string", "enum": SURFACE_CLASSES},
                    "ski_quality": {"type": "string", "enum": ["good", "ok", "bad", "unknown"]},
                    "confidence": {"type": "string", "enum": CONF},
                    "evidence": {"type": "string", "description": "short quote (max ~15 words) the label rests on"},
                },
            },
        },
        "snowline_m": _nul("integer"),
        "continuous_snow_from_m": _nul("integer"),
        "ski_from_m": _nul("integer"),
        "skidepot_m": _nul("integer"),
        "new_snow_cm": _nul("number"),
        "new_snow_days": _nul("integer"),
        "total_snow_cm": _nul("number"),
        "total_snow_elev_m": _nul("integer"),
        "wind_effects": {"type": "string", "enum": ["none", "some", "strong", "unknown"]},
        "avalanche_signs": {"type": "array", "items": {"type": "string", "enum": [
            "whumpfs", "shooting_cracks", "fresh_avalanches", "glide_cracks", "wet_slides", "none_observed"]}},
        "crust_thickness_cm": _nul("number"),
        "confidence": {"type": "string", "enum": CONF},
    },
}

SYSTEM = """Du extrahierst Schneeverhältnisse aus Tourenberichten von gipfelbuch.ch (Schweiz, Deutsch/Dialekt, teils FR/IT).
Gib nur an, was der Bericht sagt. Nichts erfinden: Unbekanntes ist null bzw. "unknown". Lieber keine Fläche als eine geratene.

Flächen (surfaces): je Höhenbereich und Exposition, die der Bericht unterscheidet, ein Eintrag.
- band: summit_upper = Gipfelbereich/oben, middle, lower = unten/Zustieg, whole_route = gilt für die ganze Tour.
  Die Kurzinfo "Oben: …" gehört zu summit_upper, "Unten: …" zu lower.
- elev_from_m / elev_to_m nur, wenn Höhen genannt sind oder sich eindeutig aus Ortsangaben ergeben, die im Bericht stehen.
- aspect nur, wenn genannt ("Nordhang", "schattig N-seitig" = N); sonst unknown.
- surface:
  powder = Pulver, locker, nicht vom Wind verändert · wind_affected = Pulver mit Windeinfluss, teils gepresst
  wind_slab_drift = Triebschnee, frische Verfrachtungen · wind_packed = Pressschnee, windgepresst, verblasen, Sastrugi
  crust_carrying = tragender Harsch/Schmelzharsch · crust_breaking = Bruchharsch, bricht ein
  corn = Sulz/aufgefirnt (fester Untergrund, oben weich) · wet_heavy = nass, faul, schwer, sulzig-tief, sumpfig
  hard_icy = hart, eisig, gefroren · groomed = präparierte Piste · thin_rocky = wenig Schnee, steinig, apere Stellen
  no_snow = aper/kein Schnee · other = nichts davon
- ski_quality: wie der Bericht die Abfahrt/Begehung dort bewertet.
- evidence: kurzes wörtliches Zitat (höchstens ~15 Wörter), auf das sich das Label stützt.

Zahlen: snowline_m = Höhe, ab der Schnee liegt; continuous_snow_from_m = ab wann durchgehend; ski_from_m = ab wo mit Ski
(angeschnallt/aufgefellt); skidepot_m = Höhe des Skidepots; new_snow_cm mit new_snow_days = über wie viele Tage;
total_snow_cm mit total_snow_elev_m = wo gemessen/geschätzt. Bereiche "20–30 cm" als Mittelwert.
confidence (gesamt und je Fläche): high = ausdrücklich gesagt, medium = klar ableitbar, low = vage."""


def report_text(r: dict) -> str:
    """The parts of a parsed report that go to the model (never the author)."""
    q = r.get("quickinfo") or {}
    lines = [
        f"Datum der Tour: {r.get('date')}",
        f"Tourenart: {r.get('activity')}",
        f"Gipfel/Ziel: {r.get('summit')} ({r.get('summit_elev')} m)",
        f"Route: {r.get('route')}",
        "Kurzinfo: " + "; ".join(f"{k}: {v}".replace("\n", " ") for k, v in q.items() if k not in ("gruppe", "personenroute")),
        "",
        "Verhältnis-Beschreibung:",
        r.get("description") or "(leer)",
    ]
    if r.get("outlook"):
        lines += ["", "Ausblick:", r["outlook"]]
    return "\n".join(lines)


def request_params(r: dict) -> dict:
    """Messages API parameters for one report (shared by live and batch)."""
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": report_text(r)}],
        # extraction: low effort is enough; the schema guarantees valid JSON
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def _clean(label: dict, keep_evidence: bool) -> dict:
    for s in label.get("surfaces", []):
        s["coarse"] = COARSE.get(s.get("surface"))
        if not keep_evidence:
            s.pop("evidence", None)
    for k in ("snowline_m", "continuous_snow_from_m", "ski_from_m", "skidepot_m", "total_snow_elev_m"):
        v = label.get(k)
        if v is not None and not (300 <= v <= 4800):
            label[k] = None                              # outside the Alps' range: drop, do not trust
    return label


def parse_message(msg, keep_evidence=True) -> dict:
    """Message -> label dict, or {'error': …} (refusal, truncation)."""
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        return {"error": f"refusal ({cat})"}
    if msg.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        label = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"invalid json: {e}"}
    label = _clean(label, keep_evidence)
    label["_model"] = msg.model
    label["_prompt"] = PROMPT_VERSION
    return label


class LiveLabeler:
    def __init__(self, keep_evidence=True):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.keep = keep_evidence
        self.usage = {"input": 0, "output": 0, "cache_read": 0}

    def __call__(self, r: dict) -> dict:
        a = self.anthropic
        try:
            msg = self.client.beta.messages.create(
                **request_params(r),
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.BadRequestError as e:
            return {"error": f"bad request: {e.message}"}
        except a.RateLimitError:
            return {"error": "rate limited"}
        except a.APIStatusError as e:
            return {"error": f"api {e.status_code}"}
        except a.APIConnectionError:
            return {"error": "connection"}
        u = msg.usage
        self.usage["input"] += u.input_tokens or 0
        self.usage["output"] += u.output_tokens or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        return parse_message(msg, self.keep)


class FakeLabeler:
    """Offline stand-in (tests, no API key): rule-based from quick info + regex facts."""
    KEYS = [("bruchharsch", "crust_breaking"), ("harsch", "crust_carrying"), ("trieb", "wind_slab_drift"),
            ("press", "wind_packed"), ("pulver", "powder"), ("sulz", "corn"), ("firn", "corn"), ("nass", "wet_heavy"),
            ("faul", "wet_heavy"), ("hart", "hard_icy"), ("eis", "hard_icy"), ("pist", "groomed"), ("aper", "no_snow")]

    def __init__(self, keep_evidence=True):
        self.keep = keep_evidence
        self.usage = {}

    def __call__(self, r: dict) -> dict:
        surfaces = []
        for band, txt in (("summit_upper", r.get("surface_top")), ("lower", r.get("surface_bottom"))):
            cls = next((c for k, c in self.KEYS if txt and k in txt.lower()), None)
            if cls:
                surfaces.append({"band": band, "elev_from_m": None, "elev_to_m": None, "aspect": "unknown",
                                 "surface": cls, "ski_quality": "unknown", "confidence": "high", "evidence": txt})
        f = {x["var"]: x["value"] for x in r.get("facts", [])}
        lab = {"snow_relevant": bool(surfaces or f), "surfaces": surfaces, "snowline_m": f.get("snowline_m"),
               "continuous_snow_from_m": None, "ski_from_m": f.get("ski_from_m"), "skidepot_m": f.get("skidepot_m"),
               "new_snow_cm": f.get("snow_cm_new"), "new_snow_days": None, "total_snow_cm": f.get("snow_cm_total"),
               "total_snow_elev_m": None, "wind_effects": "unknown", "avalanche_signs": [], "crust_thickness_cm": None,
               "confidence": "low"}
        lab = _clean(lab, self.keep)
        lab["_model"] = "fake-rules"
        lab["_prompt"] = PROMPT_VERSION
        return lab


def labeler(mode: str, keep_evidence=True):
    """mode: auto (live if an API key is set), live, fake, off."""
    if mode == "auto":
        mode = "live" if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) else "off"
    return {"live": LiveLabeler, "fake": FakeLabeler}.get(mode, lambda **_: None)(keep_evidence=keep_evidence)


def label_rows(r: dict, label: dict) -> list[dict]:
    """One training row per labelled surface, with where and when."""
    rows = []
    for i, s in enumerate(label.get("surfaces") or []):
        if not s.get("coarse"):
            continue
        lo, hi = s.get("elev_from_m"), s.get("elev_to_m")
        top = r.get("summit_elev")
        if lo is None and hi is None and s["band"] == "summit_upper" and top:
            # "oben" without heights: the top 300 m. Lower / middle bands without heights stay
            # without an elevation -- the start of a tour is not known, guessing it would be invented.
            lo, hi = top - 300, top
        elev = (lo + hi) / 2 if lo is not None and hi is not None else hi if hi is not None else lo
        rows.append({"report_id": r["id"], "seg": i, "date": r.get("date"), "lat": r.get("lat"), "lon": r.get("lon"),
                     "summit_elev": top, "band": s["band"], "elev_from_m": lo, "elev_to_m": hi, "elev_m": elev,
                     "aspect": s["aspect"], "surface": s["surface"], "coarse": s["coarse"],
                     "ski_quality": s["ski_quality"], "confidence": s["confidence"],
                     "report_confidence": label.get("confidence"), "wind_effects": label.get("wind_effects"),
                     "activity": r.get("activity"), "model": label.get("_model"), "prompt": label.get("_prompt")})
    return rows




LABEL_COLS = ["report_id", "seg", "date", "lat", "lon", "summit_elev", "band", "elev_from_m", "elev_to_m", "elev_m",
              "aspect", "surface", "coarse", "ski_quality", "confidence", "report_confidence", "wind_effects",
              "activity", "model", "prompt"]


def write_label_rows(data_dir, rows, name="gipfelbuch_surfaces.csv"):
    """Merge rows into <data>/labels/<name> (one file, replaced per report id)."""
    import csv
    from pathlib import Path
    f = Path(data_dir) / "labels" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if f.exists():
        with open(f, newline="") as fh:
            old = list(csv.DictReader(fh))
    ids = {str(r["report_id"]) for r in rows}
    allr = [o for o in old if o["report_id"] not in ids] + rows
    allr.sort(key=lambda r: (str(r["date"]), str(r["report_id"]), int(r["seg"])))
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(allr)
    return f
