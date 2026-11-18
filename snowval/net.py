"""A polite, low-profile HTTP client for the scrapers.

What "low profile" means here:
  * the requests look like an ordinary browser visit (one current browser
    User-Agent per run with the matching Accept / Accept-Language headers,
    cookies kept per session, a Referer from the site itself);
  * nothing in them says who runs the collection (no name, mail, repo, URL);
  * they are few and slow: a random pause between requests per host,
    a local cache so nothing is fetched twice, back-off on 429 / 5xx,
    and Retry-After is obeyed.

What it deliberately does NOT do: rotate proxies or IPs, solve captchas,
fake TLS fingerprints, log in, or go where robots.txt says no. If a site
blocks the collection, that is the site's answer -- ask them for access
or use their open data instead.
"""
from __future__ import annotations

import hashlib
import random
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

import requests

# current desktop browsers; one is picked per run and kept for every request
BROWSERS = [
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
     "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:144.0) Gecko/20100101 Firefox/144.0",
     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
]
LANGS = ["de-CH,de;q=0.9,en;q=0.7", "de-CH,de;q=0.9", "de,en-US;q=0.7,en;q=0.3"]


class Polite:
    def __init__(self, delay=(2.0, 6.0), cache_dir: str | Path | None = None, cache_ttl_h=6.0,
                 respect_robots=True, timeout=60, seed=None):
        rnd = random.Random(seed)
        ua, accept = rnd.choice(BROWSERS)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept": accept, "Accept-Language": rnd.choice(LANGS),
                               "Accept-Encoding": "gzip, deflate", "DNT": "1",
                               "Upgrade-Insecure-Requests": "1"})
        self.delay = delay
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.cache = Path(cache_dir) if cache_dir else None
        self.ttl = cache_ttl_h * 3600
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.n_requests = 0

    # ── helpers ──
    def _wait(self, host, factor=1.0):
        lo, hi = self.delay
        gap = random.uniform(lo, hi) * factor
        last = self._last.get(host)
        if last is not None:
            sleep = last + gap - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
        self._last[host] = time.monotonic()

    def allowed(self, url) -> bool:
        if not self.respect_robots:
            return True
        p = urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(base + "/robots.txt", timeout=self.timeout)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp = None                            # unreachable robots.txt: treat as allowed
            self._robots[base] = rp
        rp = self._robots[base]
        return rp is None or rp.can_fetch(self.s.headers["User-Agent"], url)

    def _key(self, url, params):
        return hashlib.sha1((url + "?" + repr(sorted((params or {}).items()))).encode()).hexdigest()

    # ── main entry ──
    def get(self, url, params=None, referer=None, light=False, cache=True, tries=4, **kw) -> requests.Response:
        """light=True: a cheap static file (map tile, JSON) -> shorter pause."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        key = self._key(url, params)
        if cache and self.cache:
            f = self.cache / key[:2] / key
            if f.exists() and time.time() - f.stat().st_mtime < self.ttl:
                r = requests.Response()
                r.status_code, r._content, r.url = 200, f.read_bytes(), url
                hf = f.with_suffix(".h")
                if hf.exists():
                    for line in hf.read_text().splitlines():
                        k, _, v = line.partition(": ")
                        r.headers[k] = v
                return r
        host = urlsplit(url).netloc
        headers = dict(kw.pop("headers", {}))
        if referer:
            headers["Referer"] = referer
        if light:
            headers.setdefault("Accept", "*/*")
        for i in range(tries):
            self._wait(host, .25 if light else 1.0)
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=self.timeout, **kw)
                self.n_requests += 1
            except requests.RequestException:
                if i == tries - 1:
                    raise
                time.sleep(5 * 2 ** i + random.uniform(0, 3))
                continue
            if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
                ra = r.headers.get("Retry-After", "")
                time.sleep(float(ra) if ra.isdigit() else 10 * 2 ** i + random.uniform(0, 5))
                continue
            r.raise_for_status()
            if cache and self.cache:
                f = self.cache / key[:2] / key
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
                f.with_suffix(".h").write_text("\n".join(f"{k}: {v}" for k, v in r.headers.items()
                                                         if k.lower().startswith(("ch.slf", "content-type", "last-modified"))))
            return r
        raise RuntimeError("unreachable")

    def post(self, url, data, referer=None):
        """Form post (used only to log in with your own account). Never cached."""
        host = urlsplit(url).netloc
        self._wait(host)
        headers = {"Referer": referer} if referer else {}
        r = self.s.post(url, data=data, headers=headers, timeout=self.timeout)
        self.n_requests += 1
        r.raise_for_status()
        return r
