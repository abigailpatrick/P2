#!/usr/bin/env python
"""
Extract aperture spectra for P2 grating (spectroscopic) sources from
continuum-subtracted MUSE subcubes.

Adapted from the P1 photometric version:
  - Keyed on the catalogue ID column (cast to int), cube pattern
    source_{ID}_continuum_cube_velocity.fits
  - z_av is the spectroscopic redshift; Lya window is z_av mapped to observed
    wavelength +/- --window.
  - Cubes loaded with ext=(data_ext, var_ext) to avoid the mpdaf name collision
    from the megacube's stray EXTNAME='DATA' primary card.
  - Spatial dx/dy offsets read internally from the optimal offsets CSV, so the
    script loops over all sources itself. No slurm array needed. --id runs one.
"""

import argparse
import os
import numpy as np
import pandas as pd
from mpdaf.obj import Cube
import astropy.units as u

LYA_REST = 1215.67  # AA


# ============================================================
# Utilities
# ============================================================

def load_cube(path, data_ext, var_ext):
    return Cube(path, ext=(data_ext, var_ext))


def coords_to_pixel(cube, ra, dec):
    pix = cube.wcs.sky2pix([[dec, ra]], nearest=True, unit="deg")
    y, x = pix[0]
    return int(x), int(y)


def shift_radec(ra_deg, dec_deg, dx_arcsec=0.0, dy_arcsec=0.0):
    d_ra = dx_arcsec / 3600.0 / np.cos(np.deg2rad(dec_deg))
    d_dec = dy_arcsec / 3600.0
    return ra_deg + d_ra, dec_deg + d_dec


def build_aperture_mask(nx, ny, x0, y0, radius_pix):
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return rr <= radius_pix


def extract_1d_spectrum_with_var(cube, xpix, ypix, aperture_arcsec, pixel_scale,
                                 wave_min, wave_max):
    radius_pix = aperture_arcsec / pixel_scale
    ny, nx = cube.shape[1], cube.shape[2]
    apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)

    sub = cube.select_lambda(wave_min, wave_max)
    wave = sub.wave.coord()

    flux_1d = (sub.data * apmask).sum(axis=(1, 2))
    var_1d = (sub.var * apmask).sum(axis=(1, 2))

    return wave, flux_1d, var_1d


def z_to_wavelength(z, rest_wavelength=LYA_REST):
    return (rest_wavelength * (1 + z)) * u.AA


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract aperture spectra for P2 grating sources from MUSE subcubes"
    )

    parser.add_argument("--csv", required=True,
                        help="Input catalogue with ID, ra, dec, z_av columns")
    parser.add_argument("--offset-csv", required=True,
                        help="optimal_offsets_grating.csv with ID, dx_arcsec, dy_arcsec")

    parser.add_argument("--id", type=int, default=None,
                        help="Single source ID to process. If omitted, run all.")

    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--z-col", default="z_av")

    parser.add_argument("--aperture", type=float, default=0.6,
                        help="Aperture radius in arcsec")
    parser.add_argument("--pixscale", type=float, default=0.2,
                        help="Arcsec per pixel")

    parser.add_argument("--window", type=float, default=35.0,
                        help="Half-width wavelength window around Lya (AA). "
                             "Default 35 AA matches the position optimiser.")

    parser.add_argument("--cube-dir", required=True,
                        help="Directory of continuum-subtracted cubes")
    parser.add_argument("--data-ext", type=int, default=1,
                        help="FITS extension index for the data cube")
    parser.add_argument("--var-ext", type=int, default=2,
                        help="FITS extension index for the variance cube")

    parser.add_argument("--outdir",
                        default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts",
                        help="Output directory for extracted spectra")

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    offset_path = os.path.abspath(args.offset_csv)
    cube_dir = os.path.abspath(args.cube_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("[PATHS]")
    print(f"  Input catalogue : {csv_path}")
    print(f"  Offset CSV      : {offset_path}")
    print(f"  Cube directory  : {cube_dir}")
    print(f"  Output directory: {outdir}")
    print("")

    df = pd.read_csv(csv_path)
    offsets = pd.read_csv(offset_path)

    for col in (args.id_col, "ra", "dec", args.z_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in catalogue. Available: {list(df.columns)}"
            )
    if "ID" not in offsets.columns:
        raise KeyError(
            f"Column 'ID' not found in offset CSV. Available: {list(offsets.columns)}"
        )

    # Index offsets by ID for fast lookup
    offsets = offsets.set_index("ID")

    if args.id is not None:
        df = df[df[args.id_col].astype("Int64") == args.id]
        if len(df) == 0:
            raise ValueError(f"Requested ID {args.id} not found in catalogue")

    for _, row in df.iterrows():
        idx = int(row[args.id_col])
        ra0 = row["ra"]
        dec0 = row["dec"]
        z = row[args.z_col]

        if idx not in offsets.index:
            print(f"[SKIP] Src {idx}: no entry in offset CSV")
            continue

        dx = float(offsets.loc[idx, "dx_arcsec"])
        dy = float(offsets.loc[idx, "dy_arcsec"])

        ra, dec = shift_radec(ra0, dec0, dx, dy)

        lya_obs = z_to_wavelength(z).value
        wmin = lya_obs - args.window
        wmax = lya_obs + args.window

        cube_path = os.path.join(
            cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits"
        )
        if not os.path.exists(cube_path):
            print(f"[SKIP] Src {idx}: cube not found at {cube_path}")
            continue

        print(
            f"[INFO] Src {idx}: dx={dx:.3f}\" dy={dy:.3f}\" "
            f"RA={ra:.6f} Dec={dec:.6f} lambda0={lya_obs:.1f} AA"
        )

        cube = load_cube(cube_path, args.data_ext, args.var_ext)
        xpix, ypix = coords_to_pixel(cube, ra, dec)

        wave, flux, var = extract_1d_spectrum_with_var(
            cube, xpix, ypix,
            args.aperture, args.pixscale,
            wmin, wmax,
        )

        out = os.path.join(outdir, f"{idx}_spectrum.npz")
        np.savez(
            out,
            wave=wave, flux=flux, var=var,
            ra=ra, dec=dec, z=z, lya_obs=lya_obs,
            dx=dx, dy=dy,
        )
        print(f"[OK] Src {idx}: saved {out}")

    print("")
    print("[DONE]")


if __name__ == "__main__":
    main()


"""
run on one source
python ap_extract_specs_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
  --offset-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --id 15479 \
  --aperture 0.6

run on all sources (no slurm needed)
python ap_extract_specs_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
  --offset-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --aperture 0.6 \
  --window 200
"""