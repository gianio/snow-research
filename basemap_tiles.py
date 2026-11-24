#!/usr/bin/env python3
"""One basemap image of Switzerland from swisstopo WMTS tiles (Web Mercator).

Dashboards and artifacts cannot load map tiles from other hosts at view time,
so the tiles are stitched here into one JPEG, cropped to a lat/lon box, plus a
JSON with the image's exact corners. Web Mercator, so it lines up with
d3.geoMercator (and any other Web Mercator map).

    python basemap_tiles.py --out basemap --zoom 10 [--layer ch.swisstopo.pixelkarte-grau]

Source: swisstopo, free geodata (OGD). Attribution: «© swisstopo».
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

from PIL import Image

from snowval.net import Polite

URL = "https://wmts.geo.admin.ch/1.0.0/{layer}/default/current/3857/{z}/{x}/{y}.{ext}"


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="basemap")
    ap.add_argument("--zoom", type=int, default=10)
    ap.add_argument("--layer", default="ch.swisstopo.pixelkarte-grau")
    ap.add_argument("--ext", default="jpeg")
    ap.add_argument("--bbox", default="45.78,5.90,47.86,10.55", help="lat0,lon0,lat1,lon1")
    ap.add_argument("--quality", type=int, default=82)
    a = ap.parse_args()
    la0, lo0, la1, lo1 = map(float, a.bbox.split(","))
    z = a.zoom
    x0f, y0f = tile_xy(la1, lo0, z)          # top-left
    x1f, y1f = tile_xy(la0, lo1, z)          # bottom-right
    tx0, ty0, tx1, ty1 = int(x0f), int(y0f), int(x1f), int(y1f)
    W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    mosaic = Image.new("RGB", (W, H), "white")
    net = Polite(delay=(0.05, 0.15), respect_robots=False)   # tile server for map clients, no robots.txt
    n = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            r = net.get(URL.format(layer=a.layer, z=z, x=tx, y=ty, ext=a.ext), light=True, referer="https://map.geo.admin.ch/")
            t = Image.open(io.BytesIO(r.content)).convert("RGBA")          # PNG layers are transparent outside CH
            bg = Image.new("RGBA", t.size, (255, 255, 255, 255))
            mosaic.paste(Image.alpha_composite(bg, t).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
            n += 1
    # crop to the box exactly
    cx0, cy0 = round((x0f - tx0) * 256), round((y0f - ty0) * 256)
    cx1, cy1 = round((x1f - tx0) * 256), round((y1f - ty0) * 256)
    img = mosaic.crop((cx0, cy0, cx1, cy1))

    def inv(xf, yf):
        nn = 2 ** z
        lon = xf / nn * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yf / nn))))
        return lat, lon
    north, west = inv(tx0 + cx0 / 256, ty0 + cy0 / 256)
    south, east = inv(tx0 + cx1 / 256, ty0 + cy1 / 256)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.layer}_z{z}.jpg"
    img.save(f, "JPEG", quality=a.quality, optimize=True, progressive=True)
    meta = {"file": f.name, "layer": a.layer, "zoom": z, "width": img.width, "height": img.height,
            "north": north, "south": south, "west": west, "east": east, "projection": "EPSG:3857",
            "attribution": "© swisstopo", "tiles": n}
    (out / f"{a.layer}_z{z}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), f.stat().st_size)


if __name__ == "__main__":
    main()
