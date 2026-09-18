#!/usr/bin/env python
"""
spec_nb_single.py

Single-source reproduction and test tool for the P2 grating Lya pipeline.

With just --id it reproduces the exact {ID}_flux_pseudonb.png figure and the
exact peak S/N recorded in lya_sliding_snr_grating.csv, by reading the spectrum
ap_extract_specs_grating.py already saved to {ID}_spectrum.npz and running the
identical sliding_snr_lya_grating.py routine on it. No re-extraction, so the
result is byte-identical to the pipeline.

Overrides (each optional):
  --dx           override dx offset (arcsec); default is the pipeline value
  --dy           override dy offset (arcsec); default is the pipeline value
  --linecenter   override the NB / best-centre wavelength (AA); default is the
                 sliding-S/N peak inside lya_obs +/- half-width

When an override is given the spectrum is re-extracted from the cube at the new
position, but over the SAME wavelength bounds as the stored npz. Because the
cube spectral axis does not depend on spatial position, select_lambda snaps to
the identical wavelength grid, so the sliding-window centres stay aligned and
the new S/N is directly comparable to the pipeline number.

The yellow dashed line marks the z_av-predicted (systemic) Lya position.

Cubes loaded with ext=(data_ext, var_ext) to dodge the mpdaf name collision.
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
import astropy.units as u

LYA_REST = 1215.67  # AA


# ============================================================
# Extraction utilities (verbatim from ap_extract_specs_grating.py)
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
# S/N utilities (verbatim from sliding_snr_lya_grating.py)
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
# Pseudo-NB image (verbatim from sliding_snr_lya_grating.py)
# ============================================================

def extract_pseudonb_image(cube, wmin, wmax, xpix, ypix, size_arcsec, pixscale):
    sub = cube.select_lambda(wmin, wmax)
    img = np.nansum(sub.data, axis=0)

    half_pix = (size_arcsec / pixscale) / 2.0
    x0, x1 = int(xpix - half_pix), int(xpix + half_pix)
    y0, y1 = int(ypix - half_pix), int(ypix + half_pix)

    return img[y0:y1, x0:x1]


# ============================================================
# Plot (verbatim from plot_flux_with_pseudonb)
# ============================================================

def plot_flux_with_pseudonb(
    wave, flux, var,
    best_center, window_width,
    cube, xpix, ypix,
    aperture_arcsec, pixscale,
    ra, dec, source_id,
    lya_obs, search_min, search_max,
    output_path, smooth_sigma=2,
):
    half = window_width / 2.0
    wmin, wmax = best_center - half, best_center + half

    nb_img = extract_pseudonb_image(cube, wmin, wmax, xpix, ypix, 3.0, pixscale)

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
    ax1.axvspan(wmin, wmax, color="lightblue", alpha=0.3)
    ax1.axvline(search_min, color="royalblue", ls=":", lw=1.5, label="search min")
    ax1.axvline(search_max, color="royalblue", ls=":", lw=1.5, label="search max")
    ax1.axvline(lya_obs, color="orange", ls="--", lw=2.0, label="lya(spec)")

    ax1.set_xlabel("Wavelength [\u00c5]")
    ax1.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ \u00c5$^{-1}$]")
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title(f"Source {source_id} | RA={ra:.5f} | Dec={dec:.5f}")

    ax2.imshow(nb_img, origin="lower", cmap="inferno")
    half_pix = (3.0 / pixscale) / 2.0
    r_pix = aperture_arcsec / pixscale
    ax2.add_patch(Circle((half_pix, half_pix), r_pix,
                         edgecolor="cyan", facecolor="none", lw=2.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Pseudo-NB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIG] {output_path}")


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce the pipeline single-source flux + pseudo-NB "
                    "figure, with optional position and line-centre overrides."
    )

    p.add_argument("--id", type=int, required=True, help="Source ID to process")

    p.add_argument("--indir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts",
                   help="Directory of extracted {ID}_spectrum.npz files")
    p.add_argument("--catalog",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv",
                   help="Catalogue with ID, ra, dec, z_av columns")
    p.add_argument("--cube-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/",
                   help="Directory of continuum-subtracted line cubes")

    # Overrides. When omitted the pipeline value is used and the figure reproduces.
    p.add_argument("--dx", type=float, default=None,
                   help="Override dx offset (arcsec). Default: pipeline value.")
    p.add_argument("--dy", type=float, default=None,
                   help="Override dy offset (arcsec). Default: pipeline value.")
    p.add_argument("--linecenter", type=float, default=None,
                   help="Override the NB / best-centre wavelength (AA). "
                        "Default: sliding-S/N peak inside lya_obs +/- half-width.")

    # Pipeline parameters, defaulted to match the pipeline exactly.
    p.add_argument("--id-col", default="ID")
    p.add_argument("--z-col", default="z_av")
    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--pixscale", type=float, default=0.2)
    p.add_argument("--half-width", type=float, default=35.0,
                   help="Half-width of the peak search window around lya_obs (AA)")
    p.add_argument("--window-width", type=float, default=10.0,
                   help="Sliding integration window width (AA)")
    p.add_argument("--step", type=float, default=1.0,
                   help="Sliding window step (AA)")
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

    indir = os.path.abspath(args.indir)
    catalog_path = os.path.abspath(args.catalog)
    cube_dir = os.path.abspath(args.cube_dir)
    figdir = os.path.abspath(args.figdir)
    os.makedirs(figdir, exist_ok=True)

    idx = int(args.id)
    override = (args.dx is not None) or (args.dy is not None) or (args.linecenter is not None)

    print("[PATHS]")
    print(f"  Spectra directory : {indir}")
    print(f"  Catalogue         : {catalog_path}")
    print(f"  Cube directory    : {cube_dir}")
    print(f"  Figure directory  : {figdir}")
    print("")

    # ---- Catalogue (ra0, dec0, z_av) ----
    catalog = pd.read_csv(catalog_path)
    for col in (args.id_col, "ra", "dec", args.z_col):
        if col not in catalog.columns:
            raise KeyError(
                f"Column '{col}' not in catalogue. Available: {list(catalog.columns)}"
            )
    catalog[args.id_col] = catalog[args.id_col].astype("Int64")
    catalog = catalog.set_index(args.id_col)
    if idx not in catalog.index:
        raise KeyError(f"Src {idx} not in catalogue")

    ra0 = float(catalog.loc[idx, "ra"])
    dec0 = float(catalog.loc[idx, "dec"])
    z_av = float(catalog.loc[idx, args.z_col])
    lya_obs = z_to_wavelength(z_av).value

    # ---- Load the stored pipeline spectrum ----
    npz_path = os.path.join(indir, f"{idx}_spectrum.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Spectrum not found at {npz_path}")
    npz = np.load(npz_path)
    wave_ref = npz["wave"]
    dx_pipe = float(npz["dx"])
    dy_pipe = float(npz["dy"])

    # Cube (needed for the pseudo-NB, and for re-extraction if overriding)
    cube_path = os.path.join(cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits")
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Cube not found at {cube_path}")
    cube = load_cube(cube_path, args.data_ext, args.var_ext)

    # ---- Resolve offsets ----
    dx = dx_pipe if args.dx is None else args.dx
    dy = dy_pipe if args.dy is None else args.dy
    dx_src = "pipeline (npz)" if args.dx is None else "override"
    dy_src = "pipeline (npz)" if args.dy is None else "override"

    ra, dec = shift_radec(ra0, dec0, dx, dy)

    if not override:
        # Exact reproduction: use the stored spectrum as-is.
        wave = wave_ref
        flux = npz["flux"]
        var = npz["var"]
        mode = "reproduce (stored npz)"
    else:
        # Re-extract at the new position, over the SAME wavelength bounds as the
        # stored spectrum, so the sliding-centre grid stays aligned.
        xpix_e, ypix_e = coords_to_pixel(cube, ra, dec)
        wave, flux, var = extract_1d_spectrum_with_var(
            cube, xpix_e, ypix_e,
            args.aperture, args.pixscale,
            float(wave_ref.min()), float(wave_ref.max()),
        )
        mode = "override (re-extracted, grid-aligned)"
        if len(wave) != len(wave_ref) or not np.allclose(wave, wave_ref):
            print("[WARN] Re-extracted wavelength grid differs from stored npz. "
                  "S/N may not be directly comparable.")

    # ---- Sliding S/N exactly as sliding_snr_lya_grating.py ----
    search_min = lya_obs - args.half_width
    search_max = lya_obs + args.half_width

    wave_centers, snr_vals = sliding_snr(
        wave, flux, var,
        window_width=args.window_width, step=args.step,
    )
    snr_smooth = smooth_snr_optimal(snr_vals)

    smask = (wave_centers >= search_min) & (wave_centers <= search_max)
    if not np.any(smask):
        raise ValueError("Search window falls outside the spectrum")

    imax = np.argmax(snr_smooth[smask])
    best_center_auto = wave_centers[smask][imax]
    peak_snr = snr_smooth[smask][imax]

    if args.linecenter is not None:
        best_center = float(args.linecenter)
        bc_src = "override"
    else:
        best_center = best_center_auto
        bc_src = "sliding-S/N peak"

    # ---- Pixel for the pseudo-NB (float, as in sliding_snr_lya_grating.py) ----
    ypix, xpix = cube.wcs.sky2pix([[dec, ra]], unit="deg")[0]

    print("[POSITION]")
    print(f"  Catalogue RA/Dec : {ra0:.6f}, {dec0:.6f}")
    print(f"  z_av             : {z_av:.5f}  ->  Lya(spec) = {lya_obs:.2f} AA")
    print(f"  dx applied       : {dx:+.3f}\"  ({dx_src})")
    print(f"  dy applied       : {dy:+.3f}\"  ({dy_src})")
    print(f"  Shifted RA/Dec   : {ra:.6f}, {dec:.6f}")
    print(f"  Mode             : {mode}")
    print("")
    print("[DETECTION]")
    print(f"  Search window    : {search_min:.1f} to {search_max:.1f} AA")
    print(f"  Peak S/N         : {peak_snr:.4f}")
    print(f"  Sliding peak at  : {best_center_auto:.2f} AA")
    print(f"  NB centre used   : {best_center:.2f} AA  ({bc_src})")
    print(f"  Lya(spec) at     : {lya_obs:.2f} AA  (systemic, yellow)")
    print("")

    out_path = os.path.join(figdir, f"{idx}_test.png")
    plot_flux_with_pseudonb(
        wave, flux, var,
        best_center, args.window_width,
        cube, xpix, ypix,
        args.aperture, args.pixscale,
        ra, dec, idx,
        lya_obs, search_min, search_max,
        out_path,
    )

    print("[DONE]")
    print(f"  Reported peak S/N = {peak_snr:.4f}")
    print(f"  Image written to  : {out_path}")


if __name__ == "__main__":
    main()


"""
Reproduce the pipeline figure and S/N for one source (reads the stored npz)
python spec_nb_single.py --id 18502

Explore a wider spatial offset than the grid allowed (re-extracts, grid-aligned)
python spec_nb_single.py --id 18502 --dx -0.6 --dy -0.1

Force a specific NB / best-centre wavelength
python spec_nb_single.py --id 18502 --linecenter 5323
"""