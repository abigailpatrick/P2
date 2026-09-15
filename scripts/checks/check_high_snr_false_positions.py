#!/usr/bin/env python
"""
check_high_snr_false_positions.py

Diagnostic for the P2 false-positive calibration. For a given source it reads
the optimiser output (source_{ID}_false_optimal_snr.csv), finds every false
position whose peak_snr exceeds a threshold, and builds a multi-panel grid
figure: one row per flagged position, showing a white-light cutout and a
pseudo-NB image centred on the wavelength where that position's S/N peaked.

The point is to see by eye whether the high-S/N false positions are landing on
real objects (neighbours, line emitters) or artefacts, rather than clean noise.

The optimiser CSV does not store the peak wavelength, so for each flagged
position it is recomputed the same way the optimiser found it: slide the S/N
window over z_av +/- half-width at the position's own (dx, dy) offset and take
the argmax. This keeps the NB centred on the wavelength that produced peak_snr.

Cubes loaded with ext=(data_ext, var_ext) to avoid the mpdaf name collision.
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Cube
import astropy.units as u

LYA_REST = 1215.67  # AA


# ============================================================
# Geometry / extraction utilities (matched to the optimiser)
# ============================================================

def shift_radec(ra_deg, dec_deg, dx_arcsec=0.0, dy_arcsec=0.0):
    d_ra = dx_arcsec / 3600.0 / np.cos(np.deg2rad(dec_deg))
    d_dec = dy_arcsec / 3600.0
    return ra_deg + d_ra, dec_deg + d_dec


def coords_to_pixel(cube, ra, dec):
    y, x = cube.wcs.sky2pix([[dec, ra]], nearest=True, unit="deg")[0]
    return int(x), int(y)


def build_aperture_mask(nx, ny, x0, y0, radius_pix):
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return rr <= radius_pix


def z_to_wavelength(z, rest=LYA_REST):
    return (rest * (1 + z)) * u.AA


def integrate_flux(wave, flux, region):
    m = (wave >= region[0]) & (wave <= region[1])
    if m.sum() < 2:
        return np.nan
    return np.trapz(flux[m], wave[m])


def integrate_flux_error(wave, var, region):
    m = (wave >= region[0]) & (wave <= region[1])
    if m.sum() < 2:
        return np.nan
    v = np.clip(var[m], 0.0, None)
    dw = np.median(np.diff(wave[m]))
    return np.sqrt(np.sum(v * dw ** 2))


def snr_curve(wave, flux, var, window_width, step, smooth_sigma):
    half = window_width / 2.0
    centers = np.arange(wave.min() + half, wave.max() - half, step)
    vals = []
    for c in centers:
        region = (c - half, c + half)
        f = integrate_flux(wave, flux, region)
        e = integrate_flux_error(wave, var, region) if var is not None else np.nan
        vals.append(f / e if (e is not None and np.isfinite(e) and e > 0) else np.nan)
    vals = np.nan_to_num(vals, nan=0.0)
    if smooth_sigma > 0:
        vals = gaussian_filter1d(vals, sigma=smooth_sigma)
    return centers, vals


def peak_wave_in_range(wave, flux, var, wmin, wmax,
                       window_width, step, smooth_sigma):
    centers, vals = snr_curve(wave, flux, var, window_width, step, smooth_sigma)
    mask = (centers >= wmin) & (centers <= wmax)
    if not np.any(mask):
        return np.nan, 0.0
    imax = np.argmax(vals[mask])
    return float(centers[mask][imax]), float(vals[mask][imax])


def extract_1d_from_sub(sub, apmask):
    data = sub.data.filled(np.nan) if np.ma.isMaskedArray(sub.data) else np.asarray(sub.data)
    flux = np.nansum(np.where(apmask, data, np.nan), axis=(1, 2))
    if sub.var is not None:
        var_arr = sub.var.filled(np.nan) if np.ma.isMaskedArray(sub.var) else np.asarray(sub.var)
        var = np.nansum(np.where(apmask, var_arr, np.nan), axis=(1, 2))
    else:
        var = None
    return sub.wave.coord(), flux, var


def white_light_cutout(cube, xpix, ypix, size_arcsec, pixscale):
    data = cube.data.filled(np.nan) if np.ma.isMaskedArray(cube.data) else np.asarray(cube.data)
    white = np.nanmedian(data, axis=0)
    half = int((size_arcsec / pixscale) / 2.0)
    y0, y1 = int(ypix) - half, int(ypix) + half
    x0, x1 = int(xpix) - half, int(xpix) + half
    y0, x0 = max(y0, 0), max(x0, 0)
    return white[y0:y1, x0:x1]


def nb_cutout(cube, wmin, wmax, xpix, ypix, size_arcsec, pixscale):
    sub = cube.select_lambda(wmin, wmax)
    img = np.nansum(sub.data, axis=0)
    half = int((size_arcsec / pixscale) / 2.0)
    y0, y1 = int(ypix) - half, int(ypix) + half
    x0, x1 = int(xpix) - half, int(xpix) + half
    y0, x0 = max(y0, 0), max(x0, 0)
    return img[y0:y1, x0:x1]


# ============================================================
# Per-source diagnostic
# ============================================================

def process_source(sid, opt_csv, false_csv, cube_path, outdir, args):
    opt = pd.read_csv(opt_csv)
    if "peak_snr" not in opt.columns:
        print(f"[SKIP] Src {sid}: no peak_snr column in {opt_csv}")
        return

    # Merge the optimiser rows with the position CSV to recover ra/dec/z_av.
    # Both are row-aligned per false position (false_index is 1-based).
    false = pd.read_csv(false_csv)
    if len(false) != len(opt):
        print(f"[WARN] Src {sid}: position CSV ({len(false)}) and optimiser CSV "
              f"({len(opt)}) differ in length; matching on index anyway")
    n = min(len(false), len(opt))
    opt = opt.iloc[:n].reset_index(drop=True)
    false = false.iloc[:n].reset_index(drop=True)

    z_av = float(false[args.z_col].iloc[0])

    flagged = opt[opt["peak_snr"] >= args.threshold].copy()
    if len(flagged) == 0:
        print(f"[OK] Src {sid}: no false positions above S/N {args.threshold}")
        return
    flagged = flagged.sort_values("peak_snr", ascending=False)
    if args.max_panels and len(flagged) > args.max_panels:
        print(f"[INFO] Src {sid}: {len(flagged)} above threshold, "
              f"showing top {args.max_panels}")
        flagged = flagged.iloc[:args.max_panels]

    cube = Cube(cube_path, ext=(args.data_ext, args.var_ext))
    radius_pix = args.aperture / args.pixscale

    lya_centre = z_to_wavelength(z_av).value
    lya_wmin = lya_centre - args.half_width
    lya_wmax = lya_centre + args.half_width

    nrows = len(flagged)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(6, 3 * nrows),
        squeeze=False
    )

    for row_i, (opt_idx, r) in enumerate(flagged.iterrows()):
        # Map this optimiser row back to its position row. Prefer the
        # 1-based false_index if present, else fall back to the aligned index.
        if "false_index" in r and np.isfinite(r["false_index"]):
            pos_idx = int(r["false_index"]) - 1
        else:
            pos_idx = int(opt_idx)
        pos_idx = max(0, min(pos_idx, len(false) - 1))

        base_ra = float(false["ra"].iloc[pos_idx])
        base_dec = float(false["dec"].iloc[pos_idx])
        dx = float(r.get("dx_arcsec", 0.0))
        dy = float(r.get("dy_arcsec", 0.0))
        snr_val = float(r["peak_snr"])

        ra, dec = shift_radec(base_ra, base_dec, dx, dy)
        xpix, ypix = coords_to_pixel(cube, ra, dec)

        # Recompute the peak wavelength at this position
        sub = cube.select_lambda(lya_wmin - 50, lya_wmax + 50)
        nz, ny, nx = sub.shape
        apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)
        wave, flux, var = extract_1d_from_sub(sub, apmask)
        peak_wave, recomputed_snr = peak_wave_in_range(
            wave, flux, var, lya_wmin, lya_wmax,
            args.window_width, args.snr_step, args.smooth_sigma
        )

        # White-light panel
        axw = axes[row_i][0]
        wl = white_light_cutout(cube, xpix, ypix, args.cutout, args.pixscale)
        if np.isfinite(wl).any():
            vmin = np.nanpercentile(wl, 5)
            vmax = np.nanpercentile(wl, 99)
        else:
            vmin, vmax = 0, 1
        axw.imshow(wl, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        c = (wl.shape[1] / 2.0, wl.shape[0] / 2.0)
        axw.add_patch(Circle(c, radius_pix, edgecolor="cyan",
                             facecolor="none", lw=1.5))
        axw.set_xticks([]); axw.set_yticks([])
        axw.set_ylabel(f"S/N={snr_val:.1f}", fontsize=9)
        if row_i == 0:
            axw.set_title("White light", fontsize=10)

        # Pseudo-NB panel at the recomputed peak wavelength
        axn = axes[row_i][1]
        if np.isfinite(peak_wave):
            half_w = args.window_width / 2.0
            nb = nb_cutout(cube, peak_wave - half_w, peak_wave + half_w,
                           xpix, ypix, args.cutout, args.pixscale)
            axn.imshow(nb, origin="lower", cmap="inferno")
            axn.add_patch(Circle(c, radius_pix, edgecolor="cyan",
                                 facecolor="none", lw=1.5))
            axn.set_title(f"NB @ {peak_wave:.0f} \u00c5" if row_i == 0
                          else f"{peak_wave:.0f} \u00c5", fontsize=9)
        else:
            axn.text(0.5, 0.5, "no peak", ha="center", va="center")
        axn.set_xticks([]); axn.set_yticks([])

    fig.suptitle(f"Src {sid}: false positions with S/N \u2265 {args.threshold} "
                 f"(z_av={z_av:.3f})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    out_path = os.path.join(outdir, f"source_{sid}_highsnr_falsepos.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Src {sid}: {len(flagged)} panel(s) -> {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot high-S/N false positions as white-light + NB grids."
    )
    p.add_argument("--false-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions",
                   help="Directory of _false_positions.csv and _false_optimal_snr.csv")
    p.add_argument("--cube-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/",
                   help="Directory of continuum-subtracted line cubes")
    p.add_argument("--outdir", default=None,
                   help="Output directory for the grids. Defaults to "
                        "<false-dir>/highsnr_checks")

    p.add_argument("--id", type=int, default=None,
                   help="Single source ID. If omitted, run all sources that "
                        "have an optimiser CSV.")
    p.add_argument("--threshold", type=float, default=8.0,
                   help="Plot every false position with peak_snr >= this")
    p.add_argument("--max-panels", type=int, default=12,
                   help="Cap on rows per source figure (top-N by S/N). "
                        "0 for no cap.")

    p.add_argument("--z-col", default="z_av")
    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--pixscale", type=float, default=0.2)
    p.add_argument("--cutout", type=float, default=4.0,
                   help="Cutout size in arcsec for both panels")
    p.add_argument("--half-width", type=float, default=35.0)
    p.add_argument("--window-width", type=float, default=10.0)
    p.add_argument("--snr-step", type=float, default=1.0)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--data-ext", type=int, default=1)
    p.add_argument("--var-ext", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    false_dir = os.path.abspath(args.false_dir)
    cube_dir = os.path.abspath(args.cube_dir)
    outdir = os.path.abspath(args.outdir) if args.outdir \
        else os.path.join(false_dir, "highsnr_checks")
    os.makedirs(outdir, exist_ok=True)

    print("[PATHS]")
    print(f"  False-position dir : {false_dir}")
    print(f"  Cube directory     : {cube_dir}")
    print(f"  Output directory   : {outdir}")
    print(f"  S/N threshold      : {args.threshold}")
    print("")

    if args.id is not None:
        opt_csvs = [os.path.join(false_dir, f"source_{args.id}_false_optimal_snr.csv")]
    else:
        opt_csvs = sorted(glob.glob(
            os.path.join(false_dir, "source_*_false_optimal_snr.csv")))
    if not opt_csvs:
        raise SystemExit(f"No optimiser CSVs found in {false_dir}")

    for opt_csv in opt_csvs:
        m = re.search(r"source_(\d+)_false_optimal_snr", os.path.basename(opt_csv))
        if not m:
            continue
        sid = int(m.group(1))

        false_csv = os.path.join(false_dir, f"source_{sid}_false_positions.csv")
        cube_path = os.path.join(cube_dir, f"source_{sid}_lya_contsub_cube_velocity.fits")

        if not os.path.exists(false_csv):
            print(f"[SKIP] Src {sid}: no position CSV at {false_csv}")
            continue
        if not os.path.exists(cube_path):
            print(f"[SKIP] Src {sid}: no cube at {cube_path}")
            continue

        process_source(sid, opt_csv, false_csv, cube_path, outdir, args)

    print("")
    print("[DONE]")


if __name__ == "__main__":
    main()


"""
check one source (e.g. the worst offender)
python check_high_snr_false_positions.py \
  --id 45547 \
  --threshold 8.0

check every source, flagging all false positions above S/N 8
python check_high_snr_false_positions.py \
  --threshold 8.0
"""