#!/usr/bin/env python
"""
false_pos_grating.py

Place random off-source positions in each continuum-subtracted MUSE subcube,
the first step of the false-positive calibration of the Lya detection S/N
(Patrick et al., Paper 1, Sect. 3.3.3).

For each source, --n (500) positions are drawn uniformly over the subcube,
outside an --box (8 arcsec) square centred on the source and at least
--edge-margin (1 arcsec) from the edges, as in Paper 1. The seed is --seed
plus the source ID, so the positions are the same on every run.

Noise only (new, not in Paper 1)
--------------------------------
The false positions should measure noise and the search boost, never another
galaxy's light, but otherwise have the same freedom as the real sources. A
position is rejected only if its starting aperture would contain an object
from Isaac's catalogue, i.e. an object within --clear (1.0 arcsec: the 0.6
arcsec radius plus about half the MUSE PSF FWHM) of the position, plus 0.15
arcsec so the aperture still qualifies after snapping to a pixel.
optimize_lya_positions_f_grating.py then blocks any shift that would bring an
object within --clear of the aperture, just as the real sources cannot shift
onto a neighbour. --clear 0 gives the Paper 1 placement. The fraction of the
subcube left by the cut is in the manifest.

Where the subcube runs past the edge of the MUSE footprint, its spaxels have
no data. Positions are only placed where no aperture of the shift search can
reach such a spaxel (the 0.6 arcsec radius + 0.57 arcsec shift + 0.28 arcsec
for pixel snapping, about 1.45 arcsec), so every false aperture is full of
data, as the real ones are. The fraction of the subcube without data is in
the manifest (nodata_frac).

A source is left out of the false positives only when its Lya band has data
in less than --min-coverage (half) of its channels in the extracted spectrum,
e.g. a band in the AO laser gap. Its positions would all give S/N = 0 and
pull the thresholds down.

What changed from the JELS-ID version
  - reads jwst_catalogs/primer_minerva_in_muse_zsys.csv (z_sys) and
    MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits
  - the coverage check, and a fixed seed per source
  - outputs in each source's folder, and the manifest in MUSE_catalogs
  - the figure shows the positions on the Lya-band narrowband image rather
    than the white-light image, and only the band is read from disk

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz          band and coverage
  jwst_catalogs/cosmos_primer_minerva_production.fits   Isaac positions

Outputs
-------
  MUSE_subcubes/<ID>/<ID>_false_positions.csv
      index, ra, dec, x_pix, y_pix, nearest_isaac_arcsec
  MUSE_subcubes/<ID>/<ID>_false_positions.png
      positions on the Lya-band image, with Isaac objects and the circles
      no starting aperture may contain
  MUSE_catalogs/lya_false_pos_sources.csv
      ID, used (1/0), reason, n_positions, allowed_frac (area left by the
      Isaac and no-data cuts), nodata_frac, band_coverage, lya_obs, band edges

A full run replaces the manifest. --ids patches only those rows.

Usage
-----
python false_pos_grating.py
python false_pos_grating.py --ids 23070 50803
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import p2_common as pc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--manifest", default=pc.FALSE_POS_SOURCES)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--n", type=int, default=500, help="positions per source (Paper 1: 500)")
    p.add_argument("--box", type=float, default=8.0,
                   help="side of the excluded square about the source, arcsec (Paper 1: 8)")
    p.add_argument("--edge-margin", type=float, default=1.0, help="arcsec (Paper 1: 1)")
    p.add_argument("--aperture", type=float, default=0.6,
                   help="radius, arcsec, as the real sources (sets how far from no-data "
                        "spaxels a position must be)")
    p.add_argument("--dx-max", type=float, default=0.4, help="arcsec, as the real sources")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="minimum fraction of band channels with data")
    p.add_argument("--clear", type=float, default=pc.FALSE_POS_CLEAR,
                   help="an aperture contains an Isaac object within this, arcsec "
                        "(0 for Paper 1). Must match optimize_lya_positions_f_grating.py")
    p.add_argument("--isaac", default=pc.ISAAC_CATALOG)
    p.add_argument("--seed", type=int, default=2026, help="seed is this plus the ID")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def band_coverage(spec_path):
    """Fraction of channels in the Lya band with data, and the band."""
    d = np.load(spec_path)
    wave, var = d["wave"], d["var"]
    band = (float(d["band_min"]), float(d["band_max"]))
    inb = (wave >= band[0]) & (wave <= band[1])
    good = np.isfinite(var[inb]) & (var[inb] > 0)
    return (float(good.mean()) if inb.any() else 0.0), band, float(d["lya_obs"]), wave


def nearest_object(x, y, obj_x, obj_y, pixscale):
    """Distance in arcsec from each (x, y) to the nearest object."""
    if len(obj_x) == 0:
        return np.full(np.shape(x), np.inf)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.column_stack([obj_x, obj_y])).query(np.column_stack([x, y]))
    return d * pixscale


def place(nx, ny, sx, sy, n, half_box_pix, margin_pix, rng, obj_x, obj_y,
          pixscale, avoid, data_ok=None):
    """n uniform pixel positions outside the box about (sx, sy), inside the
    edge margin, at least avoid arcsec from every object and, if data_ok is
    given, on a pixel where data_ok is True. Also returns the fraction of the
    area outside the box that these cuts leave."""
    x_lo, x_hi = margin_pix, nx - 1 - margin_pix
    y_lo, y_hi = margin_pix, ny - 1 - margin_pix
    if not (x_lo < x_hi and y_lo < y_hi):
        raise RuntimeError("edge margin leaves no area")

    def ok(x, y):
        good = (np.abs(x - sx) > half_box_pix[0]) | (np.abs(y - sy) > half_box_pix[1])
        if avoid > 0:
            good &= nearest_object(x, y, obj_x, obj_y, pixscale) >= avoid
        if data_ok is not None:
            ix = np.clip(np.round(x).astype(int), 0, nx - 1)
            iy = np.clip(np.round(y).astype(int), 0, ny - 1)
            good &= data_ok[iy, ix]
        return good

    # Area left by the object cut, from a fine uniform sample
    tx = rng.uniform(x_lo, x_hi, 20000)
    ty = rng.uniform(y_lo, y_hi, 20000)
    in_area = (np.abs(tx - sx) > half_box_pix[0]) | (np.abs(ty - sy) > half_box_pix[1])
    frac = float(ok(tx, ty)[in_area].mean()) if in_area.any() else 0.0

    xs, ys = [], []
    for _ in range(200):
        need = n - len(xs)
        if need <= 0:
            break
        x = rng.uniform(x_lo, x_hi, size=20 * need + 200)
        y = rng.uniform(y_lo, y_hi, size=20 * need + 200)
        g = ok(x, y)
        xs.extend(x[g][:need])
        ys.extend(y[g][:need])
    return np.array(xs[:n]), np.array(ys[:n]), (x_lo, x_hi, y_lo, y_hi), frac


def plot_positions(path, sid, nb, xs, ys, sx, sy, half_box_pix, limits, band, n,
                   obj_x=(), obj_y=(), avoid_pix=0):
    fig, ax = plt.subplots(figsize=(6, 6))
    fin = nb[np.isfinite(nb)]
    lim = np.percentile(np.abs(fin), 99) if fin.size else 1.0
    ax.imshow(nb, origin="lower", cmap="RdGy_r", vmin=-lim, vmax=lim, interpolation="none")
    ax.scatter(xs, ys, marker="x", s=14, c="tab:cyan", linewidths=0.8, label="false positions")
    ax.scatter([sx], [sy], marker="+", s=80, c="k", linewidths=1.5, label="source")
    for i, (ox, oy) in enumerate(zip(obj_x, obj_y)):
        ax.add_patch(plt.Circle((ox, oy), avoid_pix, fill=False, color="tab:purple",
                                lw=0.5, alpha=0.6, label="Isaac objects" if i == 0 else None))
    ax.set_xlim(-0.5, nb.shape[1] - 0.5)
    ax.set_ylim(-0.5, nb.shape[0] - 0.5)
    hx, hy = half_box_pix
    ax.plot([sx - hx, sx + hx, sx + hx, sx - hx, sx - hx],
            [sy - hy, sy - hy, sy + hy, sy + hy, sy - hy], color="gold", lw=1.2,
            label="excluded box")
    x_lo, x_hi, y_lo, y_hi = limits
    ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo],
            color="tab:green", lw=1.0, label="edge margin")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"ID {sid}, {n} false positions on the Lya band "
                 f"({band[0]:.0f}-{band[1]:.0f} " + r"$\AA$)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process(row, args, isaac):
    sid = int(row["ID"])
    rec = dict(ID=sid, used=0, reason="", n_positions=0, allowed_frac=np.nan,
               nodata_frac=0.0, band_coverage=np.nan,
               lya_obs=np.nan, band_min=np.nan, band_max=np.nan)
    cube_path = pc.muse_path(sid, "contsub", args.subcube_dir)
    spec_path = pc.muse_path(sid, "spectrum", args.subcube_dir)
    for path, what in ((cube_path, "contsub cube"), (spec_path, "spectrum")):
        if not os.path.exists(path):
            rec["reason"] = f"no {what}"
            print(f"  {sid}  SKIP no {what} at {path}")
            return rec

    cov, band, lya, wave = band_coverage(spec_path)
    rec.update(band_coverage=round(cov, 3), lya_obs=round(lya, 3),
               band_min=round(band[0], 3), band_max=round(band[1], 3))
    if cov < args.min_coverage:
        rec["reason"] = "band coverage"
        print(f"  {sid}  SKIP only {cov:.0%} of the band has data (e.g. the AO gap)")
        return rec

    with fits.open(cube_path, memmap=True) as h:
        hdr = h[1].header
        # The spectrum covers every channel of the cube, so its wavelengths
        # index the cube directly.
        k = np.where((wave >= band[0]) & (wave <= band[1]))[0]
        cut = np.array(h[1].data[k[0]:k[-1] + 1], dtype=float)
        w2d = WCS(hdr).celestial
    ny, nx = cut.shape[1:]
    pixscale = proj_plane_pixel_scales(w2d) * 3600.0

    # Spaxels without data in the band (beyond the edge of the MUSE
    # footprint). A position is only allowed if no aperture of the shift
    # search can reach one: 0.6" radius + 0.57" shift + 0.14" snap + 0.14".
    nodata = np.all(~np.isfinite(cut), axis=0)
    data_ok = None
    if nodata.any():
        from scipy.ndimage import distance_transform_edt
        reach = (args.aperture + np.sqrt(2) * (args.dx_max + 0.2)) / float(np.mean(pixscale))
        data_ok = distance_transform_edt(~nodata) >= reach
        rec["nodata_frac"] = round(float(nodata.mean()), 3)
    sx, sy = w2d.world_to_pixel_values(float(row["ra"]), float(row["dec"]))
    half_box = (args.box / 2 / pixscale[0], args.box / 2 / pixscale[1])
    margin = args.edge_margin / float(np.mean(pixscale))
    # Isaac objects on or near the subcube, in its pixels
    obj_x = obj_y = np.array([])
    if isaac is not None:
        _, ora, odec = isaac
        ox, oy = w2d.world_to_pixel_values(ora, odec)
        pad = (args.clear + 1) / float(np.mean(pixscale))
        m = (ox > -pad) & (ox < nx + pad) & (oy > -pad) & (oy < ny + pad)
        obj_x, obj_y = ox[m], oy[m]
    ps = float(np.mean(pixscale))
    rng = np.random.default_rng(args.seed + sid)
    avoid = args.clear + 0.15 if args.clear > 0 else 0.0   # 0.15": pixel snap
    xs, ys, limits, frac = place(nx, ny, sx, sy, args.n, half_box, margin, rng,
                                 obj_x, obj_y, ps, avoid, data_ok)
    rec["allowed_frac"] = round(frac, 3)
    if len(xs) < args.n:
        print(f"  {sid}  WARNING only {len(xs)} positions fit the allowed area")
    ra, dec = w2d.pixel_to_world_values(xs, ys)
    near = nearest_object(xs, ys, obj_x, obj_y, ps)

    out = pc.muse_path(sid, "false_pos", args.subcube_dir)
    pd.DataFrame(dict(index=np.arange(len(xs)), ra=ra, dec=dec, x_pix=np.round(xs, 3),
                      y_pix=np.round(ys, 3), nearest_isaac_arcsec=np.round(near, 3))
                 ).to_csv(out, index=False)
    msg = (f"  {sid}  {len(xs)} positions, {len(obj_x)} Isaac objects on the subcube, "
           + (f"{rec['nodata_frac']:.0%} of it without data, " if data_ok is not None else "")
           + f"{frac:.0%} of the area allowed\n         {out}")
    if not args.no_plots:
        png = pc.muse_path(sid, "false_pos_png", args.subcube_dir)
        nb = np.nansum(cut, axis=0)
        plot_positions(png, sid, nb, xs, ys, sx, sy, half_box, limits, band, len(xs),
                       obj_x, obj_y, avoid / ps)
        msg += f"\n         {png}"
    print(msg, flush=True)
    rec.update(used=1, n_positions=len(xs))
    return rec


def main():
    args = parse_args()
    print("false_pos_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")
    print(f"{args.n} positions per source outside a {args.box:g}\" box, "
          f"{args.edge_margin:g}\" from the edges, seed {args.seed} + ID")
    isaac = None
    if args.clear > 0:
        print(f"Starting apertures clear of Isaac objects by {args.clear:g}\" "
              f"(+0.15\" for the pixel snap), reading {os.path.abspath(args.isaac)}")
        isaac = pc.load_isaac_positions(args.isaac)
    else:
        print("Isaac objects not avoided (Paper 1 placement)")
    print("")

    rows = [process(row, args, isaac) for _, row in df.iterrows()]
    man = pd.DataFrame(rows)
    out = os.path.abspath(args.manifest)
    if args.ids and os.path.exists(out):
        old = pd.read_csv(out)
        man = pd.concat([old[~old["ID"].isin(man["ID"])], man]).sort_values("ID")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    man.to_csv(out, index=False)

    r = pd.DataFrame(rows)
    print("\n--- Summary ---")
    print(f"  with positions   {int(r['used'].sum())}")
    for reason, g in r[r["used"] == 0].groupby("reason"):
        print(f"  left out, {reason:<16} {g['ID'].tolist()}")
    print(f"\nWrote\n  {out}")


if __name__ == "__main__":
    main()