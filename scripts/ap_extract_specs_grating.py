#!/usr/bin/env python
"""
ap_extract_specs_grating.py

Extract the 0.6 arcsec radius aperture spectrum of each source from its
continuum-subtracted MUSE subcube, at the position from
optimize_lya_position_grating.py, for the PRIMER + MINERVA z_sys sample.

The method is unchanged from the JELS-ID version. The aperture is centred on
the pixel nearest Isaac's position plus the optimised offset, and flux and
variance are summed over the full wavelength range of the cube (NaN-safe).
The Lya band, -dv_blue to +dv_red km/s about Lya at z_sys (default
-300/+1200), is stored with the spectrum for the later steps.

What changed
  - reads jwst_catalogs/primer_minerva_in_muse_zsys.csv, keyed on Isaac's ID,
    with z_sys for every source (no z_dja fallback)
  - reads MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits and
    MUSE_catalogs/lya_optimal_offsets.csv
  - writes the spectrum into the source's folder
  - Lya and the band at their air wavelengths, as in lya_local_contsub.py
  - the spectrum at the original position is saved as well (flux_orig,
    var_orig), so the effect of the shift can be checked later

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  MUSE_catalogs/lya_optimal_offsets.csv             ID, dx_arcsec, dy_arcsec
  MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits

Output
------
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz
      wave (air, AA), flux, var          at the optimised position
      flux_orig, var_orig                at the original position
      ID, ra, dec (aperture centre), ra_src, dec_src (Isaac), dx, dy,
      x_pix, y_pix, aperture, z (= z_sys), z_sys_err, lya_obs (air),
      band_min, band_max, dv_blue, dv_red
  Flux in 1e-20 erg/s/cm2/AA, variance in its square.

With --no-offsets the aperture stays on Isaac's position (dx = dy = 0) and
the offsets CSV is not needed.

Usage
-----
python ap_extract_specs_grating.py
python ap_extract_specs_grating.py --ids 23070 50803
python ap_extract_specs_grating.py --no-offsets
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc

OFFSETS = f"{pc.MUSE_CAT_DIR}/lya_optimal_offsets.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--offset-csv", default=OFFSETS)
    p.add_argument("--no-offsets", action="store_true",
                   help="extract at Isaac's position, ignore the offsets CSV")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--aperture", type=float, default=0.6, help="radius, arcsec")
    p.add_argument("--dv-blue", type=float, default=300.0)
    p.add_argument("--dv-red", type=float, default=1200.0)
    return p.parse_args()


def shift_radec(ra, dec, dx_arcsec, dy_arcsec):
    return ra + dx_arcsec / 3600.0 / np.cos(np.deg2rad(dec)), dec + dy_arcsec / 3600.0


def coords_to_pixel(cube, ra, dec):
    y, x = cube.wcs.sky2pix([[dec, ra]], nearest=True, unit="deg")[0]
    return int(x), int(y)


def aperture_spectrum(flux, var, x0, y0, radius_pix):
    ny, nx = flux.shape[1:]
    yy, xx = np.indices((ny, nx))
    ap = np.hypot(xx - x0, yy - y0) <= radius_pix
    return np.nansum(flux[:, ap], axis=1), np.nansum(var[:, ap], axis=1)


def main():
    args = parse_args()
    print("ap_extract_specs_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")

    offsets = None
    if not args.no_offsets:
        off_path = os.path.abspath(args.offset_csv)
        print(f"Reading offsets     {off_path}")
        offsets = pd.read_csv(off_path).set_index("ID")
    else:
        print("No offsets, apertures on Isaac's positions")
    print(f"Aperture {args.aperture}\" radius\n")

    n_ok, skipped = 0, []
    for _, row in df.iterrows():
        sid, z = int(row["ID"]), float(row["z_sys"])
        ra0, dec0 = float(row["ra"]), float(row["dec"])
        cube_path = pc.muse_path(sid, "contsub", args.subcube_dir)
        if not os.path.exists(cube_path):
            print(f"  {sid}  SKIP no contsub cube at {cube_path}")
            skipped.append(sid)
            continue
        if offsets is not None and sid not in offsets.index:
            print(f"  {sid}  SKIP not in the offsets CSV")
            skipped.append(sid)
            continue
        dx = float(offsets.loc[sid, "dx_arcsec"]) if offsets is not None else 0.0
        dy = float(offsets.loc[sid, "dy_arcsec"]) if offsets is not None else 0.0

        cube = pc.load_muse_cube(cube_path)
        wave = cube.wave.coord()
        flux = cube.data.filled(np.nan)
        var = cube.var.filled(np.nan)
        radius_pix = args.aperture / float(np.mean(cube.wcs.get_step(unit="arcsec")))

        ra, dec = shift_radec(ra0, dec0, dx, dy)
        x, y = coords_to_pixel(cube, ra, dec)
        x0, y0 = coords_to_pixel(cube, ra0, dec0)
        fl, vr = aperture_spectrum(flux, var, x, y, radius_pix)
        fl0, vr0 = aperture_spectrum(flux, var, x0, y0, radius_pix)

        lya_wave = pc.lya_obs_air(z)
        band = pc.velocity_band(lya_wave, args.dv_blue, args.dv_red)
        z_err = float(row["z_sys_err"]) if "z_sys_err" in row and pd.notna(row["z_sys_err"]) else np.nan

        out = pc.muse_path(sid, "spectrum", args.subcube_dir)
        np.savez(out, wave=wave, flux=fl, var=vr, flux_orig=fl0, var_orig=vr0,
                 ID=sid, ra=ra, dec=dec, ra_src=ra0, dec_src=dec0, dx=dx, dy=dy,
                 x_pix=x, y_pix=y, aperture=args.aperture,
                 z=z, z_sys_err=z_err, z_source="z_sys", lya_obs=lya_wave,
                 band_min=band[0], band_max=band[1],
                 dv_blue=args.dv_blue, dv_red=args.dv_red)
        print(f"  {sid}  dx {dx:+.1f}\" dy {dy:+.1f}\", Lya {lya_wave:.1f} AA (air), "
              f"band {band[0]:.1f}-{band[1]:.1f} AA\n         {out}", flush=True)
        n_ok += 1

    print("\n--- Summary ---")
    print(f"  extracted   {n_ok}")
    print(f"  skipped     {len(skipped)}")


if __name__ == "__main__":
    main()