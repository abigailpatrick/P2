#!/usr/bin/env python
"""
lya_local_contsub.py

Local median-filter continuum subtraction of the MUSE subcubes, for the
PRIMER + MINERVA z_sys sample (Isaac IDs).

For each source the script masks the channels where Lya could fall, estimates
the continuum of every spaxel with a running median (default 101 channels)
across the unmasked channels, interpolates it across the mask and any gaps,
and subtracts it. The mask is an asymmetric velocity window about Lya at
z_sys, -dv_blue to +dv_red km/s (default -300/+1200), narrow to the blue and
wide to the red, since Lya escaping an outflow emerges redward of systemic.

The method is unchanged from the JELS-ID version. What changed
  - reads jwst_catalogs/primer_minerva_in_muse_zsys.csv, keyed on Isaac's ID,
    with z_sys for every source (no z_dja fallback)
  - reads and writes in the per-source folder MUSE_subcubes/<ID>/
  - Lya is placed at its air wavelength. z_sys is a vacuum redshift and the
    cube is in air, so 1215.67 (1 + z_sys) sat about 1.4 AA (80 km/s) too red
  - the QA plot has a second panel, the 0.6 arcsec aperture spectrum at the
    source zoomed on Lya

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  MUSE_subcubes/<ID>/<ID>_subcube.fits              extract_subcubes.py

Outputs, all in MUSE_subcubes/<ID>/
-------
  <ID>_lya_contsub_cube.fits       continuum-subtracted cube, use this for Lya
  <ID>_continuum_model_cube.fits   the continuum model, not for Lya
  <ID>_contsub_QA.png              summed spectrum of the whole subcube (top)
                                   and the aperture spectrum at Lya (bottom)
  Both cubes keep the original variance in extension 2.

Sources without a subcube are skipped, so it can be run while only some
subcubes exist.

Usage
-----
python lya_local_contsub.py
python lya_local_contsub.py --ids 23070 50803
python lya_local_contsub.py --skip-existing
python lya_local_contsub.py --dv-blue 300 --dv-red 1200 --window 101
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

import p2_common as pc

C_KMS = pc.C_KMS
APERTURE_ARCSEC = 0.6   # for the QA plot only


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR,
                   help="folder holding <ID>/<ID>_subcube.fits, outputs go "
                        "in the same <ID> folders")
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave sources that already have a contsub cube")
    p.add_argument("--window", type=int, default=101,
                   help="median filter width in channels (default 101)")
    p.add_argument("--mask-mode", choices=["none", "velocity"], default="velocity")
    p.add_argument("--dv-blue", type=float, default=300.0)
    p.add_argument("--dv-red", type=float, default=1200.0)
    return p.parse_args()


# ----------------------------------------------------------------------
# Continuum
# ----------------------------------------------------------------------
def build_mask(wave, mask_mode, lya_wave, dv_blue, dv_red):
    """True on the channels EXCLUDED from the continuum estimate."""
    if mask_mode == "none":
        return np.zeros(wave.shape, dtype=bool)
    lo, hi = pc.velocity_band(lya_wave, dv_blue, dv_red)
    return (wave > lo) & (wave < hi)


def median_continuum_cube(data, wave, lya_mask, window_pix=101):
    """Spectral median-filter continuum with interpolation across the masked
    region and spectral gaps. data is the flux cube with NaN for bad values.

    Spaxels that are entirely non-finite (the NaN margin of edge-padded
    subcubes) are skipped and left as NaN.
    """
    nz, ny, nx = data.shape
    cont = np.full(data.shape, np.nan, dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            spec = data[:, iy, ix]
            if not np.any(np.isfinite(spec)):
                continue

            good = np.isfinite(spec) & (~lya_mask)
            if good.sum() < window_pix:
                continue

            # 1) median filter on a spectrum with the masked channels filled
            spec_filled = spec.copy()
            med_val = np.nanmedian(spec[good])
            spec_filled[~good] = med_val
            cont_med = median_filter(spec_filled, size=window_pix, mode="nearest")

            # 2) interpolate the continuum across gaps and the masked region
            try:
                interp = interp1d(wave[good], cont_med[good], kind="linear",
                                  bounds_error=False, fill_value="extrapolate")
                cont[:, iy, ix] = interp(wave)
            except Exception:
                cont[:, iy, ix] = med_val

    return cont, data - cont


# ----------------------------------------------------------------------
# QA plot
# ----------------------------------------------------------------------
def aperture_mask(ny, nx, x0, y0, radius_pix):
    yy, xx = np.indices((ny, nx))
    return np.hypot(xx - x0, yy - y0) <= radius_pix


def plot_qa(path, sid, z, wave, data, cont, line, lya_wave, band, src_xy, pixscale):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top, the whole subcube summed, as before.
    ax1.plot(wave, np.nansum(data, axis=(1, 2)), color="0.6", lw=0.8, label="Original")
    ax1.plot(wave, np.nansum(cont, axis=(1, 2)), color="r", lw=0.8, label="Median continuum")
    ax1.plot(wave, np.nansum(line, axis=(1, 2)), color="k", lw=0.8, label="Continuum-subtracted")
    ax1.set_title(f"ID {sid}, z_sys = {z:.4f}, whole subcube summed", fontsize=10)

    # Bottom, the 0.6 arcsec aperture at the source, zoomed on Lya.
    ny, nx = data.shape[1:]
    ap = aperture_mask(ny, nx, src_xy[0], src_xy[1], APERTURE_ARCSEC / pixscale)
    zoom = (wave > band[0] - 60) & (wave < band[1] + 60)
    for arr, col, lab in ((data, "0.6", "Original"), (cont, "r", "Median continuum"),
                          (line, "k", "Continuum-subtracted")):
        ax2.step(wave[zoom], np.nansum(arr[zoom][:, ap], axis=1), where="mid",
                 color=col, lw=0.9, label=lab)
    ax2.axhline(0, color="0.8", lw=0.6)
    ax2.set_xlim(wave[zoom].min(), wave[zoom].max())
    ax2.set_title(f'{APERTURE_ARCSEC}" radius aperture at the source', fontsize=10)

    for ax in (ax1, ax2):
        ax.axvline(lya_wave, color="b", ls="--", lw=1, label="Lya at z_sys (air)")
        if band is not None:
            ax.axvspan(*band, color="b", alpha=0.08, label="masked band")
        ax.set_xlabel(r"Wavelength (air) [$\AA$]")
        ax.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def process_source(row, args):
    sid = int(row["ID"])
    z = float(row["z_sys"])
    sub_path = pc.muse_path(sid, "subcube", args.subcube_dir)
    line_path = pc.muse_path(sid, "contsub", args.subcube_dir)
    cont_path = pc.muse_path(sid, "continuum", args.subcube_dir)
    png_path = pc.muse_path(sid, "contsub_png", args.subcube_dir)

    if not os.path.exists(sub_path):
        print(f"  {sid}  SKIP no subcube at {sub_path}")
        return "missing"
    if args.skip_existing and os.path.exists(line_path):
        print(f"  {sid}  exists, skipped")
        return "skipped"

    t0 = time.time()
    cube = pc.load_muse_cube(sub_path)
    wave = cube.wave.coord()
    data = cube.data.filled(np.nan).astype(np.float64)
    pixscale = float(np.mean(cube.wcs.get_step(unit="arcsec")))

    lya_wave = pc.lya_obs_air(z)
    band = pc.velocity_band(lya_wave, args.dv_blue, args.dv_red) \
        if args.mask_mode == "velocity" else None
    mask = build_mask(wave, args.mask_mode, lya_wave, args.dv_blue, args.dv_red)
    print(f"  {sid}  z_sys {z:.4f}, Lya {lya_wave:.2f} AA (air), "
          f"{int(mask.sum())} channels masked", flush=True)

    cont, line = median_continuum_cube(data, wave, mask, window_pix=args.window)

    # Source position in the subcube, from the header written by
    # extract_subcubes.py (1-based), else from the WCS.
    hdr = cube.primary_header
    if "SRC_X" in hdr and "SRC_Y" in hdr:
        src_xy = (hdr["SRC_X"] - 1, hdr["SRC_Y"] - 1)
    else:
        yx = cube.wcs.sky2pix([[row["dec"], row["ra"]]], unit="deg")[0]
        src_xy = (yx[1], yx[0])
    plot_qa(png_path, sid, z, wave, data, cont, line, lya_wave, band, src_xy, pixscale)

    for path, arr, kind in ((line_path, line, "continuum-subtracted"),
                            (cont_path, cont, "continuum model")):
        out = cube.copy()
        out.data = np.ma.masked_invalid(arr.astype(np.float32))
        ph = out.primary_header
        ph["CS_KIND"] = (kind, "lya_local_contsub.py product")
        ph["CS_MASK"] = (args.mask_mode, "continuum mask mode")
        ph["CS_DVB"] = (args.dv_blue, "[km/s] mask blue edge about Lya")
        ph["CS_DVR"] = (args.dv_red, "[km/s] mask red edge about Lya")
        ph["CS_WIN"] = (args.window, "[channels] median filter width")
        ph["CS_LYA"] = (round(lya_wave, 3), "[AA, air] Lya at z_sys")
        out.write(path)

    print(f"         {line_path}\n         {cont_path}\n         {png_path}\n"
          f"         ({time.time() - t0:.0f} s)", flush=True)
    return "done"


def main():
    args = parse_args()
    print("lya_local_contsub.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")
    print(f"Mask {args.mask_mode}, -{args.dv_blue:g}/+{args.dv_red:g} km/s about Lya, "
          f"median filter {args.window} channels")
    print(f"Folders             {os.path.abspath(args.subcube_dir)}/<ID>/\n")

    status = [process_source(row, args) for _, row in df.iterrows()]
    print("\n--- Summary ---")
    for s in ("done", "skipped", "missing"):
        print(f"  {s:<10} {status.count(s)}")


if __name__ == "__main__":
    main()