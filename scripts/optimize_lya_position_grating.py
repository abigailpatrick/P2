#!/usr/bin/env python
"""
Optimise the spatial extraction position for Lya emission in continuum-subtracted
MUSE cubes, adapted for P2 grating (spectroscopic) sources.

Differences from the P1 photometric version:
  - Cubes are keyed on the catalogue ID column, path pattern
    source_{ID}_continuum_cube_velocity.fits
  - z_sys is the systemic (spectroscopic) redshift, falling back to z_dja where
    z_sys is blank. There is no spectral scan.
    The S/N search window is fixed to the asymmetric velocity band about the
    observed systemic Lya wavelength: -dv_blue to +dv_red km/s (default
    -300/+1200), matching the continuum-subtraction mask. This tracks the region
    where Lya could physically fall, narrow to the blue and wide to the red.
  - Only the spatial position (dx, dy) is optimised, with a Gaussian positional
    prior centred on the JWST position.
"""

import argparse
import os
import numpy as np
import pandas as pd

from mpdaf.obj import Cube
from scipy.ndimage import gaussian_filter1d
import astropy.units as u

LYA_REST = 1215.67  # AA
C_KMS = 299792.458


# ============================================================
# Utilities
# ============================================================

def resolve_redshift(row, zcol, zcol_fallback):
    """Return a finite redshift from zcol, else zcol_fallback, else NaN.

    Returns (z, source_tag) where source_tag names which column was used.
    """
    z = pd.to_numeric(row.get(zcol), errors="coerce")
    if np.isfinite(z):
        return float(z), zcol
    if zcol_fallback:
        z_fb = pd.to_numeric(row.get(zcol_fallback), errors="coerce")
        if np.isfinite(z_fb):
            return float(z_fb), zcol_fallback
    return np.nan, None


def velocity_band(lya_obs, dv_blue, dv_red):
    """Observed-frame wavelength edges of the asymmetric velocity band.

    dv_blue and dv_red are positive km/s half-widths to the blue and red of
    systemic. Returns (wmin, wmax) in Angstrom.
    """
    wmin = lya_obs * (1.0 - dv_blue / C_KMS)
    wmax = lya_obs * (1.0 + dv_red / C_KMS)
    return wmin, wmax

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


def extract_1d_spectrum_with_var(cube, xpix, ypix, aperture_arcsec, pixscale, wmin, wmax):
    radius_pix = aperture_arcsec / pixscale
    ny, nx = cube.shape[1], cube.shape[2]
    apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)

    sub = cube.select_lambda(wmin, wmax)
    wave = sub.wave.coord()

    flux = (sub.data * apmask).sum(axis=(1, 2))
    var = (sub.var * apmask).sum(axis=(1, 2))

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
    dw = np.median(np.diff(wave))
    return np.sqrt(np.sum(var[m] * dw ** 2))


def peak_snr_in_range(wave, flux, var, wmin, wmax, window_width=10.0, step=1.0):
    half = window_width / 2.0
    centers = np.arange(wave.min() + half, wave.max() - half, step)

    snr_vals = []
    for c in centers:
        region = (c - half, c + half)
        f = integrate_flux(wave, flux, region)
        e = integrate_flux_error(wave, var, region)
        snr_vals.append(f / e if e and e > 0 else np.nan)

    snr_vals = np.nan_to_num(snr_vals, nan=0.0)
    snr_smooth = gaussian_filter1d(snr_vals, sigma=2)

    mask = (centers >= wmin) & (centers <= wmax)
    if not np.any(mask):
        return 0.0

    return np.max(snr_smooth[mask])


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="Catalogue with ID, ra, dec, z_sys columns")
    p.add_argument("--cube-dir", required=True,
                   help="Directory holding source_{ID}_continuum_cube_velocity.fits")
    p.add_argument("--id-col", default="ID",
                   help="Name of the ID column in the catalogue")
    p.add_argument("--z-col", default="z_sys",
                   help="Name of the systemic redshift column (default z_sys)")
    p.add_argument("--z-col-fallback", default="z_dja",
                   help="Column used where z-col is blank/non-finite "
                        "(default z_dja). Set to '' to disable.")

    p.add_argument("--aperture", type=float, default=0.6,
                   help="Extraction aperture radius (arcsec)")
    p.add_argument("--pixscale", type=float, default=0.2,
                   help="MUSE spatial pixel scale (arcsec)")

    p.add_argument("--dx-max", type=float, default=0.4,
                   help="Maximum spatial offset searched (arcsec)")
    p.add_argument("--dx-step", type=float, default=0.1,
                   help="Spatial grid step (arcsec)")

    p.add_argument("--dv-blue", type=float, default=300.0,
                   help="Blue half-width of the Lya S/N search band (km/s), "
                        "measured from systemic. Default 300, matches contsub.")
    p.add_argument("--dv-red", type=float, default=1200.0,
                   help="Red half-width of the Lya S/N search band (km/s), "
                        "measured from systemic. Default 1200, matches contsub.")

    p.add_argument("--prior-scale", type=float, default=1.0,
                   help="Gaussian positional prior sigma (arcsec)")

    p.add_argument("--outfile", default="optimal_offsets_grating.csv")
    args = p.parse_args()

    # Resolve and report paths
    csv_path = os.path.abspath(args.csv)
    cube_dir = os.path.abspath(args.cube_dir)
    out_path = os.path.abspath(args.outfile)

    print("[PATHS]")
    print(f"  Input catalogue : {csv_path}")
    print(f"  Cube directory  : {cube_dir}")
    print(f"  Output CSV      : {out_path}")
    print("")

    df = pd.read_csv(csv_path)

    for col in (args.id_col, "ra", "dec", args.z_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    use_fallback = bool(args.z_col_fallback)
    if use_fallback and args.z_col_fallback not in df.columns:
        raise KeyError(
            f"Fallback column '{args.z_col_fallback}' not found. "
            f"Pass --z-col-fallback '' to disable. Available: {list(df.columns)}"
        )

    offsets = np.arange(-args.dx_max, args.dx_max + args.dx_step, args.dx_step)
    rows = []

    for _, row in df.iterrows():
        idx = int(row[args.id_col])
        ra0, dec0 = row["ra"], row["dec"]

        z, z_src = resolve_redshift(row, args.z_col, args.z_col_fallback)
        if not np.isfinite(z):
            print(f"[SKIP] Src {idx}: no finite {args.z_col}"
                  f"{' or ' + args.z_col_fallback if use_fallback else ''}")
            continue
        if z_src != args.z_col:
            print(f"[INFO] Src {idx}: {args.z_col} blank, using {z_src}={z:.4f}")

        # Asymmetric velocity band about systemic Lya, matches the contsub mask
        lya_centre = z_to_wavelength(z).value
        lya_wmin, lya_wmax = velocity_band(lya_centre, args.dv_blue, args.dv_red)

        cube_path = os.path.join(
            cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits"
        )
        if not os.path.exists(cube_path):
            print(f"[SKIP] Src {idx}: cube not found at {cube_path}")
            continue

        cube = Cube(cube_path, ext=(1, 2))

        best = dict(score=-np.inf, dx=0.0, dy=0.0, snr=0.0)

        for dx in offsets:
            for dy in offsets:
                ra, dec = shift_radec(ra0, dec0, dx, dy)
                xpix, ypix = coords_to_pixel(cube, ra, dec)

                wave, flux, var = extract_1d_spectrum_with_var(
                    cube, xpix, ypix,
                    args.aperture, args.pixscale,
                    lya_wmin - 50, lya_wmax + 50,
                )

                snr = peak_snr_in_range(wave, flux, var, lya_wmin, lya_wmax)

                if args.prior_scale > 0:
                    r2 = dx ** 2 + dy ** 2
                    penalty = np.exp(-r2 / (2 * args.prior_scale ** 2))
                else:
                    penalty = 1.0

                score = snr * penalty

                if score > best["score"]:
                    best.update(dx=dx, dy=dy, snr=snr, score=score)

        rows.append(dict(
            ID=idx,
            dx_arcsec=best["dx"],
            dy_arcsec=best["dy"],
            lya_wave_centre=lya_centre,
            peak_snr=best["snr"],
        ))

        print(
            f"[OK] Src {idx}: dx={best['dx']:.2f}\" dy={best['dy']:.2f}\" "
            f"lambda0={lya_centre:.1f} AA SNR={best['snr']:.2f}"
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("")
    print(f"[DONE] Saved {out_path}")


if __name__ == "__main__":
    main()




"""
python optimize_lya_position_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --prior-scale 1.0 \
  --dx-max 0.4 \
  --dx-step 0.1 \
  --dv-blue 300 --dv-red 1200 \
  --outfile /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv
"""