#!/usr/bin/env python
"""
Run the significance-maximising grid search over the random false-positive
positions for a P2 grating source, to build the false-positive peak-S/N
distribution used to calibrate the detection threshold.

Adapted from the P1 version:
  - z_av is the spectroscopic redshift; the search window is z_av mapped to
    observed Lya +/- a fixed half-width (default 35 AA), matching the real
    source optimiser. No zmin/zmax scan.
  - Cube loaded with ext=(data_ext, var_ext).
  - NaN-safe aperture sums so edge apertures do not warn or silently
    under-count.
  - Each false position runs through the identical spatial grid search and
    positional prior as the real sources, so the false-positive population
    samples the same optimisation bias.
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd

from mpdaf.obj import Cube
from scipy.ndimage import gaussian_filter1d
import astropy.units as u

LYA_REST = 1215.67  # AA


# ============================================================
# Utilities
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


def extract_1d_spectrum_with_var_from_sub(sub, apmask):
    wave = sub.wave.coord()
    data = sub.data.filled(np.nan) if np.ma.isMaskedArray(sub.data) else np.asarray(sub.data)
    flux = np.nansum(np.where(apmask, data, np.nan), axis=(1, 2))
    if sub.var is not None:
        var_arr = sub.var.filled(np.nan) if np.ma.isMaskedArray(sub.var) else np.asarray(sub.var)
        var = np.nansum(np.where(apmask, var_arr, np.nan), axis=(1, 2))
    else:
        var = None
    return wave, flux, var


def z_to_wavelength(z, rest=LYA_REST):
    return (rest * (1 + z)) * u.AA


# ============================================================
# SNR utilities
# ============================================================

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


def peak_snr_in_range(wave, flux, var, wmin, wmax,
                      window_width=10.0, step=1.0, smooth_sigma=2.0):
    half = window_width / 2.0
    centers = np.arange(wave.min() + half, wave.max() - half, step)

    snr_vals = []
    for c in centers:
        region = (c - half, c + half)
        f = integrate_flux(wave, flux, region)
        e = integrate_flux_error(wave, var, region) if var is not None else np.nan
        snr_vals.append(f / e if (e is not None and np.isfinite(e) and e > 0) else np.nan)

    snr_vals = np.nan_to_num(snr_vals, nan=0.0)
    if smooth_sigma > 0:
        snr_vals = gaussian_filter1d(snr_vals, sigma=smooth_sigma)

    mask = (centers >= wmin) & (centers <= wmax)
    if not np.any(mask):
        return 0.0
    return float(np.max(snr_vals[mask]))


# ============================================================
# Per-source run
# ============================================================

def run_one(false_path, cube_path, out_path, args):
    print("[SOURCE]")
    print(f"  False positions CSV : {false_path}")
    print(f"  Cube                : {cube_path}")
    print(f"  Output CSV          : {out_path}")

    df = pd.read_csv(false_path)
    if args.z_col not in df.columns:
        raise KeyError(
            f"Column '{args.z_col}' not found in false CSV. Available: {list(df.columns)}"
        )

    cube = Cube(cube_path, ext=(args.data_ext, args.var_ext))

    offsets = np.arange(-args.dx_max, args.dx_max + 1e-6, args.dx_step)
    radius_pix = args.aperture / args.pixscale

    rows_out = []

    for i, row in df.iterrows():
        ra0, dec0 = float(row["ra"]), float(row["dec"])
        z_av = float(row[args.z_col])

        lya_centre = z_to_wavelength(z_av).value
        lya_wmin = lya_centre - args.half_width
        lya_wmax = lya_centre + args.half_width

        sub = cube.select_lambda(lya_wmin - 50, lya_wmax + 50)
        nz, ny, nx = sub.shape

        best = dict(score=-np.inf, dx=0.0, dy=0.0, snr=0.0)

        for dx in offsets:
            for dy in offsets:
                ra, dec = shift_radec(ra0, dec0, dx, dy)
                xpix, ypix = coords_to_pixel(cube, ra, dec)

                apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)
                wave, flux, var = extract_1d_spectrum_with_var_from_sub(sub, apmask)

                snr = peak_snr_in_range(
                    wave, flux, var, lya_wmin, lya_wmax,
                    window_width=args.window_width,
                    step=args.snr_step,
                    smooth_sigma=args.smooth_sigma,
                )

                if args.prior_scale > 0:
                    r2 = dx ** 2 + dy ** 2
                    penalty = np.exp(-r2 / (2 * args.prior_scale ** 2))
                else:
                    penalty = 1.0

                score = snr * penalty
                if score > best["score"]:
                    best.update(dx=dx, dy=dy, snr=snr, score=score)

        rows_out.append(dict(
            false_index=i + 1,
            ra_deg=ra0, dec_deg=dec0,
            dx_arcsec=best["dx"], dy_arcsec=best["dy"],
            peak_snr=best["snr"],
        ))

    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"  [DONE] Saved {out_path}")
    print("")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    # Single-source inputs
    p.add_argument("--false-csv", default=None,
                   help="Per-source false-position CSV (single-source mode)")
    p.add_argument("--cube", default=None,
                   help="Continuum-subtracted line cube (single-source mode)")

    # All-sources inputs
    p.add_argument("--all", action="store_true",
                   help="Loop over every false-position CSV in --false-dir")
    p.add_argument("--false-dir", default=None,
                   help="Directory of source_{ID}_false_positions.csv files")
    p.add_argument("--cube-dir", default=None,
                   help="Directory of source_{ID}_lya_contsub_cube_velocity.fits cubes")
    p.add_argument("--outdir", default=None,
                   help="Directory for the per-source _false_optimal_snr.csv "
                        "outputs. Defaults to --false-dir.")

    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--pixscale", type=float, default=0.2)

    p.add_argument("--dx-max", type=float, default=0.4)
    p.add_argument("--dx-step", type=float, default=0.1)
    p.add_argument("--prior-scale", type=float, default=1.0)

    p.add_argument("--half-width", type=float, default=35.0,
                   help="Fixed observed-frame half-width of the Lya search "
                        "window around z_av (AA)")
    p.add_argument("--z-col", default="z_av")

    p.add_argument("--window-width", type=float, default=10.0)
    p.add_argument("--snr-step", type=float, default=1.0)
    p.add_argument("--smooth-sigma", type=float, default=2.0)

    p.add_argument("--data-ext", type=int, default=1)
    p.add_argument("--var-ext", type=int, default=2)

    p.add_argument("--outfile", default=None,
                   help="Output path (single-source mode only)")
    args = p.parse_args()

    # ---------------- All-sources mode ----------------
    if args.all:
        if not args.false_dir or not args.cube_dir:
            raise SystemExit("--all requires --false-dir and --cube-dir")

        false_dir = os.path.abspath(args.false_dir)
        cube_dir = os.path.abspath(args.cube_dir)
        outdir = os.path.abspath(args.outdir) if args.outdir else false_dir
        os.makedirs(outdir, exist_ok=True)

        print("[PATHS]")
        print(f"  False-position dir : {false_dir}")
        print(f"  Cube directory     : {cube_dir}")
        print(f"  Output directory   : {outdir}")
        print("")

        false_csvs = sorted(glob.glob(
            os.path.join(false_dir, "source_*_false_positions.csv")))
        if not false_csvs:
            raise SystemExit(f"No false-position CSVs found in {false_dir}")

        n_done, n_skipped = 0, 0
        for false_path in false_csvs:
            m = re.search(r"source_(\d+)_false_positions", os.path.basename(false_path))
            if not m:
                print(f"[SKIP] Cannot parse ID from {false_path}")
                n_skipped += 1
                continue
            sid = int(m.group(1))

            cube_path = os.path.join(
                cube_dir, f"source_{sid}_lya_contsub_cube_velocity.fits")
            if not os.path.exists(cube_path):
                print(f"[SKIP] Src {sid}: cube not found at {cube_path}")
                n_skipped += 1
                continue

            out_path = os.path.join(outdir, f"source_{sid}_false_optimal_snr.csv")
            run_one(false_path, cube_path, out_path, args)
            n_done += 1

        print(f"[ALL DONE] {n_done} sources processed, {n_skipped} skipped")
        return

    # ---------------- Single-source mode ----------------
    if not args.false_csv or not args.cube:
        raise SystemExit(
            "Single-source mode requires --false-csv and --cube "
            "(or use --all with --false-dir and --cube-dir)"
        )

    false_path = os.path.abspath(args.false_csv)
    cube_path = os.path.abspath(args.cube)

    if args.outfile is None:
        base = os.path.splitext(os.path.basename(cube_path))[0]
        outdir = os.path.dirname(false_path)
        args.outfile = os.path.join(outdir, f"{base}_false_optimal_snr.csv")
        print(f"[INFO] No --outfile specified, using {args.outfile}")
    out_path = os.path.abspath(args.outfile)

    print("[PATHS]")
    run_one(false_path, cube_path, out_path, args)


if __name__ == "__main__":
    main()


"""
run on all sources (replaces the slurm array)
python optimize_lya_positions_f_grating.py \
  --all \
  --false-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --outdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions \
  --aperture 0.6 \
  --dx-max 0.4 \
  --dx-step 0.1 \
  --prior-scale 1.0

run on one source
python optimize_lya_positions_f_grating.py \
  --false-csv /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions/source_15479_false_positions.csv \
  --cube /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/source_15479_lya_contsub_cube_velocity.fits \
  --aperture 0.6 \
  --dx-max 0.4 \
  --dx-step 0.1 \
  --prior-scale 1.0
"""