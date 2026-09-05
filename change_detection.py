#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from scipy import ndimage

AOI = (85.505, 28.270, 85.560, 28.312)
UTM = "EPSG:32645"
PRE_DATES = ["20260711", "20260723", "20260804", "20260816"]
POST_DATE = "20260828"
INC_MIN, INC_MAX = 20.0, 70.0


def layer(zip_path: Path, suffix: str) -> str:
    s = zip_path.name[:-4]
    return f"/vsizip/{zip_path}/{s}/{s}_{suffix}.tif"


def read(path: str, bounds):
    with rasterio.open(path) as src:
        assert src.crs.to_string() == UTM, f"{path}: {src.crs}, expected {UTM}"
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.window_transform(win)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtc-dir", default="data/rtc")
    ap.add_argument("--scar-db", type=float, default=6.0)
    ap.add_argument("--min-cluster", type=int, default=8)
    args = ap.parse_args()

    zips = sorted(Path(args.rtc_dir).glob("*.zip"))
    if not zips:
        raise SystemExit(f"no RTC product zips in {args.rtc_dir}")
    by_date = {z.name[7:15]: z for z in zips}

    lo, la = AOI[::2], AOI[1::2]
    x, y = warp_transform("EPSG:4326", UTM, list(lo), list(la))
    bounds = (min(x), min(y), max(x), max(y))

    vv = {}
    for d in PRE_DATES + [POST_DATE]:
        a, tr = read(layer(by_date[d], "VV"), bounds)
        a[a <= 0] = np.nan
        vv[d] = a
    inc, tr_i = read(layer(by_date[POST_DATE], "inc_map"), bounds)
    dem, tr_d = read(layer(by_date[POST_DATE], "dem"), bounds)
    assert inc.shape == dem.shape == vv[POST_DATE].shape
    assert tr_i == tr_d == tr
    dem[dem <= 0] = np.nan
    inc = np.degrees(inc) if np.nanmax(inc) < 3.2 else inc

    valid = (np.isfinite(inc) & (inc > INC_MIN) & (inc < INC_MAX)
             & np.all(np.dstack([np.isfinite(vv[d]) for d in vv]), axis=2))
    change = 10 * np.log10(vv[POST_DATE] / vv[PRE_DATES[-1]])
    scar = valid & np.isfinite(change) & (change > args.scar_db)

    lab, n = ndimage.label(scar)
    sizes = ndimage.sum(scar, lab, range(1, n + 1))
    keep = np.isin(lab, np.where(sizes >= args.min_cluster)[0] + 1)
    if not keep.any():
        raise SystemExit("nothing passed the threshold")
    px_km2 = abs(tr.a * tr.e) / 1e6
    jump = 10 * np.log10(np.nanmean(vv[POST_DATE][keep])
                         / np.nanmean(vv[PRE_DATES[-1]][keep]))
    print(f"{int(keep.sum())} px = {keep.sum() * px_km2:.2f} km2, "
          f"median elevation {np.nanmedian(dem[keep]):.0f} m, "
          f"{jump:+.2f} dB across {PRE_DATES[-1]} -> {POST_DATE}")

    elev = float(np.nanmedian(dem[keep]))
    control = (valid & np.isfinite(dem) & ~scar
               & (np.abs(dem - elev) < 200.0))
    print(f"{'date':<10}{'scar dB':>9}{'control dB':>12}")
    for d in PRE_DATES + [POST_DATE]:
        print(f"{d:<10}"
              f"{10 * np.log10(np.nanmean(vv[d][keep])):>9.2f}"
              f"{10 * np.log10(np.nanmean(vv[d][control])):>12.2f}")
    ctrl = (10 * np.log10(np.nanmean(vv[POST_DATE][control]))
            - 10 * np.log10(np.nanmean(vv[PRE_DATES[-1]][control])))
    print(f"control ({int(control.sum()):,} px at {elev:.0f} +/- 200 m): "
          f"{ctrl:+.2f} dB across the event")
    for t in (4.0, 5.0, 6.0, 7.0, 8.0):
        m = valid & np.isfinite(change) & (change > t)
        lb, nn = ndimage.label(m)
        sz = ndimage.sum(m, lb, range(1, nn + 1))
        kp = np.isin(lb, np.where(sz >= args.min_cluster)[0] + 1)
        print(f"threshold {t:.0f} dB -> {int(kp.sum())} px")

    feats = []
    for geom, val in shapes(keep.astype("uint8"), mask=keep, transform=tr):
        rings = []
        for ring in geom["coordinates"]:
            xs, ys = zip(*ring)
            lon, lat = warp_transform(UTM, "EPSG:4326", list(xs), list(ys))
            rings.append(list(zip(lon, lat)))
        feats.append(rings)

    z = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    z = ndimage.gaussian_filter(z, 1.0)
    gy, gx = np.gradient(z, abs(tr.a))
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.clip((np.sin(alt) * np.sin(slope)
                  + np.cos(alt) * np.cos(slope) * np.cos(az - aspect) + 1) / 2, 0, 1)
    t = np.clip((z - np.nanpercentile(z, 2))
                / (np.nanpercentile(z, 99.5) - np.nanpercentile(z, 2)), 0, 1)
    rgb = np.dstack([0.52 + 0.44 * t, 0.47 + 0.40 * t, 0.40 + 0.38 * t])
    rgb = np.clip(rgb * (0.30 + 0.70 * hs[..., None]), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype("uint8")).save(buf, "PNG")

    cx = [bounds[0], bounds[2], bounds[2], bounds[0]]
    cy = [bounds[1], bounds[1], bounds[3], bounds[3]]
    clon, clat = warp_transform(UTM, "EPSG:4326", cx, cy)
    west, east = min(clon), max(clon)
    south, north = min(clat), max(clat)

    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           '<name>Lhende 2026 collapse scar</name>',
           '<Style id="scar"><LineStyle><color>ff00d4ff</color><width>2.5</width>'
           '</LineStyle><PolyStyle><color>554848e3</color></PolyStyle></Style>',
           '<GroundOverlay><name>Copernicus DEM (shaded)</name>'
           '<drawOrder>0</drawOrder><Icon><href>dem.png</href></Icon>'
           f'<LatLonBox><north>{north:.6f}</north><south>{south:.6f}</south>'
           f'<east>{east:.6f}</east><west>{west:.6f}</west></LatLonBox>'
           '</GroundOverlay>']
    for i, rings in enumerate(feats, 1):
        outer = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in rings[0])
        kml += [f'<Placemark><name>scar {i}</name><styleUrl>#scar</styleUrl>',
                '<Polygon><outerBoundaryIs><LinearRing><coordinates>',
                outer,
                '</coordinates></LinearRing></outerBoundaryIs>']
        for hole in rings[1:]:
            inner = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in hole)
            kml += ['<innerBoundaryIs><LinearRing><coordinates>',
                    inner,
                    '</coordinates></LinearRing></innerBoundaryIs>']
        kml.append('</Polygon></Placemark>')
    kml.append('</Document></kml>')

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    kmz = out / "source_zone_scar.kmz"
    with zipfile.ZipFile(kmz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", "\n".join(kml))
        zf.writestr("dem.png", buf.getvalue())
    print(f"wrote {kmz}")


if __name__ == "__main__":
    main()
