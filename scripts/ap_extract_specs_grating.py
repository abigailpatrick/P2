#!/usr/bin/env python
"""
Extract aperture spectra for P2 grating (spectroscopic) sources from
continuum-subtracted MUSE subcubes.

Adapted from the P1 photometric version:
  - Keyed on the catalogue ID column (cast to int), cube pattern
    source_{ID}_continuum_cube_velocity.fits
  - z_sys is the systemic redshift (falling back to z_dja where blank).
  - The FULL spectral axis of each subcube is extracted, not a window. The
    asymmetric velocity band (-dv_blue/+dv_red km/s about systemic Lya, default
    -300/+1200) is recorded in the .npz so downstream plots can shade the search
    region while the extracted spectrum continues either side of it.
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
C_KMS = 299792.458


# ============================================================
# Utilities
# ============================================================

def resolve_redshift(row, zcol, zcol_fallback):
    """Return (z, source_tag) from zcol, else zcol_fallback, else (NaN, None)."""
    z = pd.to_numeric(row.get(zcol), errors="coerce")
    if np.isfinite(z):
        return float(z), zcol
    if zcol_fallback:
        z_fb = pd.to_numeric(row.get(zcol_fallback), errors="coerce")
        if np.isfinite(z_fb):
            return float(z_fb), zcol_fallback
    return np.nan, None


def velocity_band(lya_obs, dv_blue, dv_red):
    """Observed-frame edges (AA) of the asymmetric velocity band about lya_obs."""
    wmin = lya_obs * (1.0 - dv_blue / C_KMS)
    wmax = lya_obs * (1.0 + dv_red / C_KMS)
    return wmin, wmax


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
                                 wave_min=None, wave_max=None):
    """Aperture-summed 1D spectrum. With wave_min/wave_max None, the full
    spectral axis of the cube is returned."""
    radius_pix = aperture_arcsec / pixel_scale
    ny, nx = cube.shape[1], cube.shape[2]
    apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)

    if wave_min is None or wave_max is None:
        sub = cube
    else:
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
                        help="Input catalogue with ID, ra, dec, z_sys columns")
    parser.add_argument("--offset-csv", required=True,
                        help="optimal_offsets_grating.csv with ID, dx_arcsec, dy_arcsec")

    parser.add_argument("--id", type=int, default=None,
                        help="Single source ID to process. If omitted, run all.")

    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--z-col", default="z_sys")
    parser.add_argument("--z-col-fallback", default="z_dja",
                        help="Column used where z-col is blank (default z_dja). "
                             "Set to '' to disable.")

    parser.add_argument("--aperture", type=float, default=0.6,
                        help="Aperture radius in arcsec")
    parser.add_argument("--pixscale", type=float, default=0.2,
                        help="Arcsec per pixel")

    parser.add_argument("--dv-blue", type=float, default=300.0,
                        help="Blue half-width (km/s) of the search band recorded "
                             "in the .npz. Default 300, matches contsub.")
    parser.add_argument("--dv-red", type=float, default=1200.0,
                        help="Red half-width (km/s) of the search band recorded "
                             "in the .npz. Default 1200, matches contsub.")

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
    use_fallback = bool(args.z_col_fallback)
    if use_fallback and args.z_col_fallback not in df.columns:
        raise KeyError(
            f"Fallback column '{args.z_col_fallback}' not found. "
            f"Pass --z-col-fallback '' to disable. Available: {list(df.columns)}"
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

        z, z_src = resolve_redshift(row, args.z_col, args.z_col_fallback)
        if not np.isfinite(z):
            print(f"[SKIP] Src {idx}: no finite {args.z_col}"
                  f"{' or ' + args.z_col_fallback if use_fallback else ''}")
            continue
        if z_src != args.z_col:
            print(f"[INFO] Src {idx}: {args.z_col} blank, using {z_src}={z:.4f}")

        if idx not in offsets.index:
            print(f"[SKIP] Src {idx}: no entry in offset CSV")
            continue

        dx = float(offsets.loc[idx, "dx_arcsec"])
        dy = float(offsets.loc[idx, "dy_arcsec"])

        ra, dec = shift_radec(ra0, dec0, dx, dy)

        lya_obs = z_to_wavelength(z).value
        band_min, band_max = velocity_band(lya_obs, args.dv_blue, args.dv_red)

        cube_path = os.path.join(
            cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits"
        )
        if not os.path.exists(cube_path):
            print(f"[SKIP] Src {idx}: cube not found at {cube_path}")
            continue

        print(
            f"[INFO] Src {idx}: dx={dx:.3f}\" dy={dy:.3f}\" "
            f"RA={ra:.6f} Dec={dec:.6f} lambda0={lya_obs:.1f} AA "
            f"band=[{band_min:.1f},{band_max:.1f}] AA"
        )

        cube = load_cube(cube_path, args.data_ext, args.var_ext)
        xpix, ypix = coords_to_pixel(cube, ra, dec)

        # Full spectral axis, no window
        wave, flux, var = extract_1d_spectrum_with_var(
            cube, xpix, ypix,
            args.aperture, args.pixscale,
            wave_min=None, wave_max=None,
        )

        out = os.path.join(outdir, f"{idx}_spectrum.npz")
        np.savez(
            out,
            wave=wave, flux=flux, var=var,
            ra=ra, dec=dec, z=z, z_source=z_src, lya_obs=lya_obs,
            band_min=band_min, band_max=band_max,
            dv_blue=args.dv_blue, dv_red=args.dv_red,
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
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --offset-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --id 15479 \
  --aperture 0.6

run on all sources (no slurm needed). Full spectrum extracted, band recorded.
python ap_extract_specs_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --offset-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --aperture 0.6 \
  --dv-blue 300 --dv-red 1200
"""