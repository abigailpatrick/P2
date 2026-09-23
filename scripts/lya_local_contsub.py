#!/usr/bin/env python
"""
lya_local_contsub.py

Local median-filter continuum subtraction for the JELS MUSE subcubes.

Adapted from the original megacube-extraction pipeline to run on the
per-source subcubes named  subcube_JELSID_<ID>.fits, keyed by the 'ID'
column of the input CSV. The source redshift is read from 'z_sys' (falling
back to 'z_dja' where z_sys is blank), and the continuum-exclusion mask
defaults to an ASYMMETRIC velocity window about the observed Lya wavelength:
narrow on the blue side and wide on the red side, since Lya escaping an
expanding medium emerges redward of systemic.

For each source the script builds a spectral mask, estimates a per-spaxel
continuum by median filtering with interpolation across the mask and any
spectral gaps, subtracts it, writes a continuum cube and a continuum-
subtracted (line) cube, and saves a QA plot.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpdaf.obj import Cube
import astropy.units as u
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

# -------------------------
# Constants
# -------------------------
LYA_REST = 1215.67 * u.AA
C_KMS = 299792.458

# -------------------------
# Utilities
# -------------------------

def z_to_lya_obs(z):
    return (LYA_REST * (1 + z)).value  # Angstrom


def velocity_mask(wave, lya_obs, dv_blue=300, dv_red=1200):
    """Boolean mask, True inside the asymmetric velocity window about lya_obs.

    dv is measured relative to the observed Lya wavelength, so dv < 0 is
    blueward (shorter wavelength) and dv > 0 is redward (longer wavelength).
    The window spans -dv_blue to +dv_red, allowing a wide red tail to cover
    Lya offsets while keeping the blue side tight.
    """
    dv = C_KMS * (wave - lya_obs) / lya_obs
    return (dv > -dv_blue) & (dv < dv_red)


def zrange_mask(wave, z_min, z_max, buffer_A=5.0):
    """Boolean mask, True between observed Lya at z_min and z_max, plus buffer."""
    lo = z_to_lya_obs(z_min) - buffer_A
    hi = z_to_lya_obs(z_max) + buffer_A
    print(f"[MASK] zrange: {lo:.2f}-{hi:.2f} A")
    print(f" Maximum mask size in A: {hi - lo:.2f}")
    return (wave >= lo) & (wave <= hi)


def build_mask(wave, mask_mode, lya_obs=None,
               z_min=None, z_max=None,
               dv_blue=300, dv_red=1200, buffer_A=5.0):
    """Construct the continuum-exclusion mask for the chosen mode.

    mask_mode:
      'none'     -> no channels excluded (all False)
      'velocity' -> exclude -dv_blue to +dv_red km/s around lya_obs (default)
      'zrange'   -> exclude z_min(-buffer) to z_max(+buffer) in Lya-obs space
    """
    if mask_mode == "none":
        return np.zeros(wave.shape, dtype=bool)

    if mask_mode == "velocity":
        if lya_obs is None:
            raise ValueError("velocity mask requires lya_obs")
        return velocity_mask(wave, lya_obs, dv_blue=dv_blue, dv_red=dv_red)

    if mask_mode == "zrange":
        if z_min is None or z_max is None:
            raise ValueError("zrange mask requires z_min and z_max")
        return zrange_mask(wave, z_min, z_max, buffer_A=buffer_A)

    raise ValueError(f"Unknown mask_mode '{mask_mode}'")


def median_continuum_cube(cube, lya_mask, window_pix=101):
    """Spectral median-filter continuum subtraction with explicit
    interpolation across the masked region and spectral gaps.

    lya_mask : 1D boolean array over the spectral axis, True where channels
               are EXCLUDED from the continuum estimate.

    Spaxels that are entirely non-finite (the NaN margin on edge-padded
    subcubes) are skipped and left as NaN, so the padding never enters the
    continuum estimate.
    """
    data = cube.data.copy()
    nz, ny, nx = data.shape
    wave = cube.wave.coord()

    cont = np.full_like(data, np.nan)

    for iy in range(ny):
        for ix in range(nx):
            spec = data[:, iy, ix]

            if not np.any(np.isfinite(spec)):
                continue

            # valid continuum pixels ONLY
            good = np.isfinite(spec) & (~lya_mask)

            # need enough points to define a continuum
            if good.sum() < window_pix:
                continue

            # 1) median filter on a mask-filled spectrum
            spec_filled = spec.copy()
            med_val = np.nanmedian(spec[good])
            spec_filled[~good] = med_val

            cont_med = median_filter(
                spec_filled,
                size=window_pix,
                mode="nearest"
            )

            # 2) interpolate continuum across gaps + masked region
            cont_good = cont_med[good]
            wave_good = wave[good]

            try:
                interp = interp1d(
                    wave_good,
                    cont_good,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                cont[:, iy, ix] = interp(wave)

            except Exception:
                # fallback: flat continuum
                cont[:, iy, ix] = med_val

    line = data - cont
    return cont, line


def plot_QA(cube, cont, line, lya_obs, source_id, outdir, mode_tag,
            dv_blue=None, dv_red=None):
    """QA plot using spatially summed spectra."""
    wave = cube.wave.coord()

    spec = np.nansum(cube.data, axis=(1, 2))
    cont_spec = np.nansum(cont, axis=(1, 2))
    line_spec = np.nansum(line, axis=(1, 2))

    plt.figure(figsize=(10, 5))
    plt.plot(wave, spec, color="0.6", label="Original")
    plt.plot(wave, cont_spec, color="r", label="Median continuum")
    plt.plot(wave, line_spec, color="k", label="Continuum-subtracted")
    plt.axvline(lya_obs, color="b", ls="--", label="Lya (systemic)")

    if dv_blue is not None and dv_red is not None:
        w_blue = lya_obs * (1 - dv_blue / C_KMS)
        w_red = lya_obs * (1 + dv_red / C_KMS)
        plt.axvspan(w_blue, w_red, color="b", alpha=0.08,
                    label=f"mask (-{dv_blue:.0f}/+{dv_red:.0f} km/s)")

    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Flux *10**-20")
    plt.title(f"Source {source_id}: median-filter continuum subtraction ({mode_tag})")
    plt.legend()

    img_dir = os.path.join(outdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(
        os.path.join(img_dir, f"source_{source_id}_median_contsub_QA_{mode_tag}.png"),
        dpi=150
    )
    plt.close()


# -------------------------
# Main processing
# -------------------------

def load_cube(cube_path):
    """Load a subcube, pointing mpdaf at the data and variance extensions.

    The subcubes have the flux in image extension 1 and the variance in
    extension 2. The primary HDU (index 0) is empty but, because it inherited
    an EXTNAME='DATA' card from the parent megacube, it shares the name 'DATA'
    with the real data extension. Selecting by name is therefore ambiguous and
    mpdaf resolves 'DATA' to the empty primary, raising "cannot manage data
    with 0 axes". Selecting by integer index avoids the collision.

    Falls back to name-based and default loads so the same function still
    opens cleanly written cubes that lack the duplicate name.
    """
    try:
        return Cube(cube_path, ext=(1, 2))
    except Exception:
        pass
    try:
        return Cube(cube_path, ext=1)
    except Exception:
        pass
    try:
        return Cube(cube_path, ext=("DATA", "STAT"))
    except Exception:
        pass
    return Cube(cube_path)


def process_source(cube_path, z, source_id, outdir,
                   mask_mode="velocity",
                   z_min=None, z_max=None,
                   window_pix=101,
                   dv_blue=300, dv_red=1200,
                   buffer_A=5.0,
                   make_plot=True):

    cube = load_cube(cube_path)
    wave = cube.wave.coord()
    lya_obs = z_to_lya_obs(z)

    lya_mask = build_mask(
        wave, mask_mode,
        lya_obs=lya_obs,
        z_min=z_min, z_max=z_max,
        dv_blue=dv_blue, dv_red=dv_red, buffer_A=buffer_A,
    )

    n_masked = int(lya_mask.sum())
    if mask_mode == "velocity":
        print(f"[MASK] {mask_mode}: -{dv_blue:.0f}/+{dv_red:.0f} km/s about "
              f"{lya_obs:.2f} A ({n_masked} channels excluded)")
    elif mask_mode == "zrange":
        lo = z_to_lya_obs(z_min) - buffer_A
        hi = z_to_lya_obs(z_max) + buffer_A
        print(f"[MASK] {mask_mode}: {lo:.2f}-{hi:.2f} A "
              f"({n_masked} channels excluded)")
    else:
        print(f"[MASK] {mask_mode}: no channels excluded")

    cont, line = median_continuum_cube(
        cube,
        lya_mask,
        window_pix=window_pix,
    )

    if make_plot:
        plot_QA(cube, cont, line, lya_obs, source_id, outdir, mask_mode,
                dv_blue=dv_blue if mask_mode == "velocity" else None,
                dv_red=dv_red if mask_mode == "velocity" else None)

    cont_cube = cube.copy()
    line_cube = cube.copy()

    cont_cube.data[:] = cont
    line_cube.data[:] = line

    cont_out = os.path.join(
        outdir, f"source_{source_id}_continuum_cube_{mask_mode}.fits")
    line_out = os.path.join(
        outdir, f"source_{source_id}_lya_contsub_cube_{mask_mode}.fits")

    for path, obj in zip([cont_out, line_out], [cont_cube, line_cube]):
        if os.path.exists(path):
            os.remove(path)
        obj.write(path)

    print(f"[OK] Source {source_id} processed ({mask_mode})")


# -------------------------
# CLI
# -------------------------

def format_id(raw_id):
    """Match the subcube naming: whole-number IDs render without a .0."""
    try:
        fval = float(raw_id)
        return str(int(fval)) if fval.is_integer() else str(raw_id)
    except (TypeError, ValueError):
        return str(raw_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cube_dir", required=True,
                        help="Directory holding subcube_JELSID_<ID>.fits.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--source", default=None,
                        help="Single source ID (matches the CSV ID column). "
                             "Omit to process all rows.")
    parser.add_argument("--window", type=int, default=101)
    parser.add_argument("--mask_mode", choices=["none", "velocity", "zrange"],
                        default="velocity",
                        help="Continuum-exclusion mode (default velocity).")
    parser.add_argument("--dv_blue", type=float, default=300,
                        help="Blue half-width in km/s for velocity mode "
                             "(default 300).")
    parser.add_argument("--dv_red", type=float, default=1200,
                        help="Red half-width in km/s for velocity mode "
                             "(default 1200).")
    parser.add_argument("--buffer_A", type=float, default=5.0,
                        help="Buffer in Angstrom per side for zrange mode.")
    parser.add_argument("--zcol", default="z_sys",
                        help="CSV column holding the redshift (default z_sys).")
    parser.add_argument("--zcol_fallback", default="z_dja",
                        help="CSV column used where zcol is blank/non-finite "
                             "(default z_dja). Set to '' to disable fallback.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for col in ("ID", args.zcol):
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column '{col}'.")

    use_fallback = bool(args.zcol_fallback)
    if use_fallback and args.zcol_fallback not in df.columns:
        raise SystemExit(
            f"CSV missing fallback column '{args.zcol_fallback}'. "
            f"Pass --zcol_fallback '' to disable the fallback.")

    # Select rows by ID, or all rows.
    if args.source is not None:
        target = format_id(args.source)
        rows = df[df["ID"].apply(format_id) == target]
        if rows.empty:
            raise SystemExit(f"No row with ID {args.source} in the CSV.")
    else:
        rows = df

    for _, row in rows.iterrows():
        sid = format_id(row["ID"])

        z = pd.to_numeric(row[args.zcol], errors="coerce")

        if not np.isfinite(z) and use_fallback:
            z_fb = pd.to_numeric(row[args.zcol_fallback], errors="coerce")
            if np.isfinite(z_fb):
                print(f"[INFO] Source {sid}: {args.zcol} blank, "
                      f"falling back to {args.zcol_fallback}={z_fb:.4f}.")
                z = z_fb

        if not np.isfinite(z):
            print(f"[WARN] Source {sid}: no finite {args.zcol}"
                  f"{' or ' + args.zcol_fallback if use_fallback else ''}, "
                  f"skipping.")
            continue

        # zrange columns kept optional, only used if mask_mode is zrange.
        z_min = row["z1_min"] if "z1_min" in df.columns else None
        z_max = row["z1_max"] if "z1_max" in df.columns else None

        if args.mask_mode == "zrange" and (z_min is None or z_max is None):
            raise SystemExit(
                "mask_mode 'zrange' needs z1_min and z1_max in the CSV.")

        cube_path = os.path.join(args.cube_dir, f"subcube_JELSID_{sid}.fits")

        if not os.path.exists(cube_path):
            print(f"[WARN] Missing cube for source {sid}: {cube_path}")
            continue

        process_source(
            cube_path,
            z,
            sid,
            args.outdir,
            mask_mode=args.mask_mode,
            z_min=z_min,
            z_max=z_max,
            window_pix=args.window,
            dv_blue=args.dv_blue,
            dv_red=args.dv_red,
            buffer_A=args.buffer_A,
        )


if __name__ == "__main__":
    main()



"""
Asymmetric velocity mask (-300/+1200 km/s about Lya at z_sys, z_dja
fallback), all sources:
python lya_local_contsub.py \
  --mask_mode velocity --dv_blue 300 --dv_red 1200 \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --cube_dir /cephfs/apatrick/P2/MUSE_subcubes/ \
  --outdir /cephfs/apatrick/P2/MUSE_subcubes/contsub

Single source by ID:
python lya_local_contsub.py \
  --source 16871 --mask_mode velocity --dv_blue 300 --dv_red 1200 \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --cube_dir /cephfs/apatrick/P2/MUSE_subcubes/ \
  --outdir /cephfs/apatrick/P2/MUSE_subcubes/contsub


"""