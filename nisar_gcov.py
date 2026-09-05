#!/usr/bin/env python
"""
NISAR L-band cross-check of the Lhende source-zone scar.

Two steps, one file:

    python nisar_gcov.py download        # search + fetch L2 GCOV over the AOI
    python nisar_gcov.py analyze         # L-band jump test + C-vs-L figure

`download` uses asf_search and needs a (free) NASA Earthdata login. Put the
credentials in ~/.netrc as

    machine urs.earthdata.nasa.gov login <user> password <pass>

or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the environment.

`analyze` reads the geocoded HH backscatter straight out of the GCOV HDF5
(no unpacking, no offset processing - the heavy work is already done in the
product), crops to the source zone, and compares the pre/post pair the same
way scripts did for Sentinel-1.

Deps:  pip install asf_search h5py numpy scipy matplotlib pyproj
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

# source zone (same box the Sentinel-1 script uses)
AOI = (85.505, 28.270, 85.560, 28.312)          # W, S, E, N (lon/lat)
# the two co-event pairs found in Vertex; download grabs whatever GCOV it finds
# over the AOI in this window and you keep the ones you want
START, END = "2026-08-14", "2026-09-02"
NISAR_DIR = Path("data/nisar")
SCAR_GEOJSON = Path("outputs/source_zone_scar.geojson")


# --------------------------------------------------------------- download
def download() -> None:
    import asf_search as asf

    NISAR_DIR.mkdir(parents=True, exist_ok=True)
    session = None
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if user and pw:
        session = asf.ASFSession().auth_with_creds(user, pw)
    else:
        try:
            session = asf.ASFSession().auth_with_creds  # netrc path below
        except Exception:
            pass

    aoi_wkt = (f"POLYGON(({AOI[0]} {AOI[1]},{AOI[2]} {AOI[1]},"
               f"{AOI[2]} {AOI[3]},{AOI[0]} {AOI[3]},{AOI[0]} {AOI[1]}))")
    print("searching ASF for NISAR GCOV over the source zone ...")
    results = asf.search(
        intersectsWith=aoi_wkt, start=START, end=END,
        dataset=getattr(asf.DATASET, "NISAR", "NISAR"),
        processingLevel="GCOV")
    if not results:
        # fall back to a platform search if the dataset enum differs by version
        results = asf.search(intersectsWith=aoi_wkt, start=START, end=END,
                             platform="NISAR")
        results = [r for r in results if "GCOV" in str(
            r.properties.get("fileName", "")).upper()]
    print(f"  {len(results)} GCOV granule(s):")
    for r in results:
        print("   ", r.properties.get("fileName") or r.properties.get("sceneName"))
    if not results:
        raise SystemExit("no GCOV granules found - widen START/END or check login")

    opts = {}
    if user and pw:
        opts["session"] = asf.ASFSession().auth_with_creds(user, pw)
    print(f"downloading to {NISAR_DIR}/ ...")
    asf.download_urls(
        urls=[r.properties["url"] for r in results],
        path=str(NISAR_DIR),
        session=opts.get("session"))
    print("done. files:")
    for p in sorted(NISAR_DIR.glob("*.h5")):
        print(f"   {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")


# --------------------------------------------------------------- analyze
def _find(h5, needle, kind="dataset"):
    """First object whose path contains all substrings in `needle`."""
    import h5py
    hits = []

    def visit(name, obj):
        if all(s in name for s in needle):
            if kind == "dataset" and isinstance(obj, h5py.Dataset):
                hits.append(name)
            elif kind == "group" and isinstance(obj, h5py.Group):
                hits.append(name)
    h5.visititems(visit)
    return hits


def read_gcov(path, pol="HHHH"):
    """(backscatter, x, y, epsg) for one geocoded GCOV, frequency A."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        grid = _find(f, ["GCOV", "frequencyA"], "group")
        base = grid[0] if grid else "science/LSAR/GCOV/grids/frequencyA"
        ds = _find(f, ["frequencyA", pol])
        if not ds:
            raise SystemExit(f"{path}: polarisation {pol} not found")
        arr = f[ds[0]][()].astype("float32")
        xs = f[_find(f, ["frequencyA", "xCoordinates"])[0]][()]
        ys = f[_find(f, ["frequencyA", "yCoordinates"])[0]][()]
        epsg = None
        proj = _find(f, ["frequencyA", "projection"])
        if proj:
            v = f[proj[0]][()]
            epsg = int(v) if np.ndim(v) == 0 else int(v.flat[0])
        if epsg is None:
            for a in ("epsg", "epsgCode"):
                for d in _find(f, ["frequencyA"]):
                    if a in f[d].attrs:
                        epsg = int(f[d].attrs[a]); break
        arr[arr <= 0] = np.nan
    return arr, xs, ys, epsg or 32645


def scar_mask(xs, ys, epsg):
    import json
    import numpy as np
    from matplotlib.path import Path as MplPath
    from pyproj import Transformer
    ring = np.array(json.loads(SCAR_GEOJSON.read_text())
                    ["features"][0]["geometry"]["coordinates"][0])
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    px, py = tr.transform(ring[:, 0], ring[:, 1])
    XX, YY = np.meshgrid(xs, ys)
    inside = MplPath(np.column_stack([px, py])).contains_points(
        np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return inside


def analyze() -> None:
    import numpy as np
    files = sorted(NISAR_DIR.glob("*GCOV*.h5")) or sorted(NISAR_DIR.glob("*.h5"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 GCOV files in {NISAR_DIR}/ (run download first)")
    # order by the acquisition date embedded in the NISAR name (…_YYYYMMDD…)
    import re
    def datekey(p):
        m = re.search(r"(20\d{6})T", p.name) or re.search(r"(20\d{6})", p.name)
        return m.group(1) if m else p.name
    files = sorted(files, key=datekey)
    pre_f, post_f = files[0], files[-1]
    print(f"pre : {pre_f.name}\npost: {post_f.name}")

    pre, xs, ys, epsg = read_gcov(pre_f)
    post, xs2, ys2, _ = read_gcov(post_f)
    if pre.shape != post.shape:
        raise SystemExit("pre/post grids differ - are these the same track?")

    m = scar_mask(xs, ys, epsg) & np.isfinite(pre) & np.isfinite(post)
    bg = (~scar_mask(xs, ys, epsg)) & np.isfinite(pre) & np.isfinite(post)
    scar_pre = 10 * np.log10(np.nanmean(pre[m]))
    scar_post = 10 * np.log10(np.nanmean(post[m]))
    bg_pre = 10 * np.log10(np.nanmean(pre[bg]))
    bg_post = 10 * np.log10(np.nanmean(post[bg]))
    print(f"\nL-band HH, {int(m.sum())} px on the scar:")
    print(f"  scar   {scar_pre:6.2f} -> {scar_post:6.2f} dB  ({scar_post-scar_pre:+.2f})")
    print(f"  around {bg_pre:6.2f} -> {bg_post:6.2f} dB  ({bg_post-bg_pre:+.2f})")

    # figure: pre / post / change over the crop, scar outlined
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    ch = 10 * np.log10(post / pre)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for a, img, t, kw in [
            (ax[0], 10 * np.log10(pre), f"pre  {datekey(pre_f)}", dict(cmap="gray")),
            (ax[1], 10 * np.log10(post), f"post {datekey(post_f)}", dict(cmap="gray")),
            (ax[2], ch, "change (dB)", dict(cmap="RdBu_r", vmin=-8, vmax=8))]:
        a.imshow(img, extent=ext, origin="upper", **kw)
        a.set_title(t, fontsize=12, fontweight="bold", loc="left")
        a.set_xticks([]); a.set_yticks([])
    ax[2].text(.02, .02, f"scar {scar_post-scar_pre:+.1f} dB (L-band HH)  ·  "
               f"C-band was +12.7 dB", transform=ax[2].transAxes, fontsize=11,
               fontweight="bold", va="bottom",
               bbox=dict(fc="white", ec="none", alpha=.85))
    Path("figures").mkdir(exist_ok=True)
    fig.suptitle("NISAR L-band over the Lhende source zone — same scar, longer wavelength",
                 fontsize=14, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .96])
    out = "figures/nisar_lband_scar.png"
    fig.savefig(out, dpi=125)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "analyze"])
    a = ap.parse_args()
    (download if a.step == "download" else analyze)()
