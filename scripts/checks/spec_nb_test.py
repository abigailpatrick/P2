#!/usr/bin/env python
"""
spec_nb_test.py

Interactive test-extraction tool for P2 grating subcubes.

Produces the same style of output as sliding_snr_lya_grating.py (a spectrum
panel plus a pseudo-NB cutout) but gives manual control over:

  - the spatial extraction position, via --dx-arcsec / --dy-arcsec offsets
    applied to the catalogue RA/Dec (in sky coordinates, exactly as
    ap_extract_specs_grating.py applies its optimal offsets)
  - the wavelength the pseudo-NB and S/N window are centred on, via --nb-wave
    (integer AA)
  - the extraction / NB window width, via --window-width

The spectrum is extracted in exactly the same way as ap_extract_specs_grating.py
so the plot matches the pipeline output:
  - select_lambda(wmin, wmax) on a window around --nb-wave
  - coords_to_pixel via sky2pix(..., nearest=True)
  - boolean circular aperture mask, summed with (sub.data * apmask).sum(axis=(1,2))

Everything defaults to the pipeline values but each can be overridden. Output
images are named {ID}_test.png. The peak S/N is reported inside the selected
window (--nb-wave +/- window-width/2).

Cubes are loaded with ext=(data_ext, var_ext) to avoid the mpdaf name collision.
"""

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Cube

LYA_REST = 1215.67  # AA


# ============================================================
# Extraction utilities (matched to ap_extract_specs_grating.py)
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


# ============================================================
# Integration & SNR utilities (matched to sliding_snr_lya_grating.py)
# ============================================================

def integrate_flux(wave, flux, region):
    l1, l2 = region
    mask = (wave >= l1) & (wave <= l2)
    if np.sum(mask) < 2:
        return np.nan
    return np.trapz(flux[mask], wave[mask])


def integrate_flux_error(wave, var, region):
    l1, l2 = region
    mask = (wave >= l1) & (wave <= l2)
    if np.sum(mask) < 2:
        return np.nan
    dw = np.median(np.diff(wave))
    return np.sqrt(np.sum(var[mask] * dw ** 2))


def line_snr(f, ferr):
    if not np.isfinite(ferr) or ferr <= 0:
        return np.nan
    return f / ferr


def smooth_snr_optimal(snr, sigma=2):
    return gaussian_filter1d(np.nan_to_num(snr, nan=0.0), sigma=sigma)


def sliding_snr(wave, flux, var, window_width=10.0, step=1.0):
    half = window_width / 2.0
    centers = np.arange(wave.min() + half, wave.max() - half, step)

    snr_vals = []
    for c in centers:
        region = (c - half, c + half)
        f = integrate_flux(wave, flux, region)
        e = integrate_flux_error(wave, var, region)
        snr_vals.append(line_snr(f, e))

    return centers, np.array(snr_vals)


# ============================================================
# Pseudo-NB image
# ============================================================

def extract_pseudonb_image(cube, wmin, wmax, xpix, ypix, size_arcsec, pixscale):
    sub = cube.select_lambda(wmin, wmax)
    img = np.nansum(sub.data, axis=0)

    half_pix = (size_arcsec / pixscale) / 2.0
    x0, x1 = int(xpix - half_pix), int(xpix + half_pix)
    y0, y1 = int(ypix - half_pix), int(ypix + half_pix)

    return img[y0:y1, x0:x1]


# ============================================================
# Plotting (matched to plot_flux_with_pseudonb)
# ============================================================

def plot_test(
    wave, flux, var,
    nb_wave, window_width,
    cube, xpix, ypix,
    aperture_arcsec, pixscale,
    ra, dec, source_id,
    dx_arcsec, dy_arcsec,
    peak_snr,
    nb_size_arcsec,
    output_path, smooth_sigma=2
):
    half = window_width / 2.0
    wmin, wmax = nb_wave - half, nb_wave + half

    nb_img = extract_pseudonb_image(cube, wmin, wmax, xpix, ypix,
                                    nb_size_arcsec, pixscale)

    flux_smooth = gaussian_filter1d(flux, smooth_sigma)
    noise = np.sqrt(var)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]}
    )

    ax1.step(wave, flux, where="mid", color="black", lw=1.0, alpha=0.6, label="Flux")
    ax1.fill_between(wave, -noise, noise, color="gray", alpha=0.3, step="mid",
                     label=r"$\pm 1\sigma$")
    ax1.plot(wave, flux_smooth, color="crimson", lw=1.8, label="Smoothed")

    ax1.axhline(0.0, color="k", ls="--", lw=0.8)
    ax1.axvspan(wmin, wmax, color="lightblue", alpha=0.3, label="NB window")
    ax1.axvline(nb_wave, color="orange", ls="--", lw=2.0, label="NB centre")

    ax1.set_xlabel("Wavelength [\u00c5]")
    ax1.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ \u00c5$^{-1}$]")
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title(
        f"Source {source_id} | RA={ra:.5f} Dec={dec:.5f} | "
        f"dx={dx_arcsec:+.2f}\" dy={dy_arcsec:+.2f}\" | "
        f"peak S/N={peak_snr:.2f}"
    )

    ax2.imshow(nb_img, origin="lower", cmap="inferno")
    half_pix = (nb_size_arcsec / pixscale) / 2.0
    r_pix = aperture_arcsec / pixscale
    ax2.add_patch(Circle((half_pix, half_pix), r_pix,
                         edgecolor="cyan", facecolor="none", lw=2.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"Pseudo-NB @ {nb_wave} \u00c5")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIG] {output_path}")


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Test spectrum + NB extraction with manual position and wavelength."
    )

    p.add_argument("--id", type=int, required=True,
                   help="Source ID to process")
    p.add_argument("--cube-dir", required=True,
                   help="Directory of continuum-subtracted line cubes")
    p.add_argument("--catalog", required=True,
                   help="Catalogue with ID, ra, dec columns")

    p.add_argument("--nb-wave", type=int, required=True,
                   help="Observed wavelength (integer AA) to centre the NB / window on")

    p.add_argument("--dx-arcsec", type=float, default=0.0,
                   help="Spatial offset in RA direction (arcsec), default 0")
    p.add_argument("--dy-arcsec", type=float, default=0.0,
                   help="Spatial offset in Dec direction (arcsec), default 0")

    p.add_argument("--window-width", type=float, default=10.0,
                   help="NB / S/N window width (AA), default 10")
    p.add_argument("--extract-window", type=float, default=35.0,
                   help="Half-width of the spectral extraction window around "
                        "--nb-wave (AA). Default 35 matches the pipeline.")

    p.add_argument("--match-pipeline", action="store_true",
                   help="Report the smoothed sliding-window peak S/N (as "
                        "sliding_snr_lya_grating.py does) instead of the raw "
                        "single-window S/N. Slides across the extraction window, "
                        "smooths the S/N curve with gaussian_filter1d(sigma=2), "
                        "and takes the max.")
    p.add_argument("--step", type=float, default=1.0,
                   help="Sliding window step (AA), only used with "
                        "--match-pipeline. Default 1.")

    p.add_argument("--id-col", default="ID")
    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--pixscale", type=float, default=0.2)
    p.add_argument("--nb-size", type=float, default=3.0,
                   help="Pseudo-NB cutout size (arcsec), default 3.0")
    p.add_argument("--data-ext", type=int, default=1)
    p.add_argument("--var-ext", type=int, default=2)

    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures",
                   help="Output directory for the _test.png image")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    cube_dir = os.path.abspath(args.cube_dir)
    catalog_path = os.path.abspath(args.catalog)
    figdir = os.path.abspath(args.figdir)
    os.makedirs(figdir, exist_ok=True)

    print("[PATHS]")
    print(f"  Cube directory   : {cube_dir}")
    print(f"  Catalogue        : {catalog_path}")
    print(f"  Figure directory : {figdir}")
    print("")

    catalog = pd.read_csv(catalog_path)
    for col in (args.id_col, "ra", "dec"):
        if col not in catalog.columns:
            raise KeyError(
                f"Column '{col}' not found in catalogue. Available: {list(catalog.columns)}"
            )
    catalog[args.id_col] = catalog[args.id_col].astype("Int64")
    catalog = catalog.set_index(args.id_col)

    idx = int(args.id)
    if idx not in catalog.index:
        raise KeyError(f"Src {idx} not in catalogue")

    ra0 = float(catalog.loc[idx, "ra"])
    dec0 = float(catalog.loc[idx, "dec"])

    cube_path = os.path.join(
        cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits"
    )
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Cube not found at {cube_path}")

    cube = load_cube(cube_path, args.data_ext, args.var_ext)

    # Apply the requested spatial offset in sky coordinates, same as the pipeline
    ra, dec = shift_radec(ra0, dec0, args.dx_arcsec, args.dy_arcsec)
    xpix, ypix = coords_to_pixel(cube, ra, dec)

    print("[POSITION]")
    print(f"  Catalogue RA/Dec : {ra0:.6f}, {dec0:.6f}")
    print(f"  Offset applied   : dx={args.dx_arcsec:+.2f}\" dy={args.dy_arcsec:+.2f}\"")
    print(f"  Shifted RA/Dec   : {ra:.6f}, {dec:.6f}")
    print(f"  Extraction pixel : x={xpix}, y={ypix}")
    print("")

    # Spectral extraction window around the chosen NB wavelength
    wmin_ext = args.nb_wave - args.extract_window
    wmax_ext = args.nb_wave + args.extract_window

    wave, flux, var = extract_1d_spectrum_with_var(
        cube, xpix, ypix,
        args.aperture, args.pixscale,
        wmin_ext, wmax_ext,
    )

    # Raw single-window S/N at exactly the selected NB window
    half = args.window_width / 2.0
    wmin, wmax = args.nb_wave - half, args.nb_wave + half
    f = integrate_flux(wave, flux, (wmin, wmax))
    e = integrate_flux_error(wave, var, (wmin, wmax))
    snr_single = line_snr(f, e)

    # Pipeline-matched smoothed sliding peak S/N over the extraction window
    wave_centers, snr_vals = sliding_snr(
        wave, flux, var,
        window_width=args.window_width, step=args.step
    )
    snr_smooth = smooth_snr_optimal(snr_vals)
    if len(snr_smooth) > 0 and np.any(np.isfinite(snr_smooth)):
        imax = np.nanargmax(snr_smooth)
        snr_pipeline = snr_smooth[imax]
        center_pipeline = wave_centers[imax]
    else:
        snr_pipeline = np.nan
        center_pipeline = np.nan

    # Which one drives the reported peak_snr and the plot title
    if args.match_pipeline:
        peak_snr = snr_pipeline
    else:
        peak_snr = snr_single

    print("[EXTRACTION WINDOW]")
    print(f"  Centre           : {args.nb_wave} \u00c5")
    print(f"  Half-width       : {args.extract_window} \u00c5  "
          f"({wmin_ext:.1f} to {wmax_ext:.1f})")
    print(f"  N pixels in spec : {len(wave)}")
    print("")
    print("[NB / S/N WINDOW]")
    print(f"  Centre           : {args.nb_wave} \u00c5")
    print(f"  Width            : {args.window_width} \u00c5  ({wmin:.1f} to {wmax:.1f})")
    print(f"  Integrated flux  : {f:.4g}")
    print(f"  Flux error       : {e:.4g}")
    print("")
    print("[S/N COMPARISON]")
    print(f"  Single-window (raw)      : {snr_single:.2f}  at {args.nb_wave} \u00c5")
    print(f"  Pipeline (slide+smooth)  : {snr_pipeline:.2f}  at {center_pipeline:.1f} \u00c5")
    mode = "pipeline (slide+smooth)" if args.match_pipeline else "single-window (raw)"
    print(f"  Reported peak S/N        : {peak_snr:.2f}  [{mode}]")
    print("")

    out_path = os.path.join(figdir, f"{idx}_test.png")
    plot_test(
        wave, flux, var,
        args.nb_wave, args.window_width,
        cube, xpix, ypix,
        args.aperture, args.pixscale,
        ra, dec, idx,
        args.dx_arcsec, args.dy_arcsec,
        peak_snr,
        args.nb_size,
        out_path
    )

    print("[DONE]")
    print(f"  Reported peak S/N = {peak_snr:.2f}  [{mode}]")
    print(f"  Image written to: {out_path}")


if __name__ == "__main__":
    main()


"""
default extraction (no offset, standard 10 AA S/N window)
python spec_nb_test.py \
  --id 15479 \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
  --nb-wave 5800

shift the extraction position and widen the S/N window
python spec_nb_test.py \
  --id 15479 \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
  --nb-wave 5800 \
  --dx-arcsec 0.2 --dy-arcsec -0.1 \
  --window-width 10
  --extract-window 150
  --match-pipeline

"""
