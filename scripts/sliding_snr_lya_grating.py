#!/usr/bin/env python
"""
Sliding-window Lya S/N measurement and diagnostic plots for P2 grating sources.

Adapted from the P1 photometric version:
  - Reads spectra written by ap_extract_specs_grating.py ({ID}_spectrum.npz).
    These now hold the FULL spectral axis, so the sliding S/N runs across the
    whole spectrum and the peak search is restricted to the asymmetric velocity
    band about systemic Lya.
  - z_sys is the systemic redshift (falling back to z_dja where blank).
  - The peak search band is read from the .npz (band_min, band_max), written by
    the extractor as -dv_blue/+dv_red km/s about systemic (default -300/+1200).
    Older .npz files without the band fall back to computing it from z_sys.
  - Cubes loaded with ext=(data_ext, var_ext) to avoid the mpdaf name collision.
  - Loops over all spectra by default; --id runs a single source.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Cube

LYA_REST = 1215.67  # AA
C_KMS = 299792.458


# ============================================================
# Redshift / band helpers
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


# ============================================================
# Integration & SNR utilities
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
# Plotting: SNR
# ============================================================

def plot_snr_with_pseudonb(
    wave_centers, snr, snr_smooth,
    best_center, peak_snr, window_width,
    cube, xpix, ypix,
    aperture_arcsec, pixscale,
    ra, dec, source_id,
    lya_obs, search_min, search_max,
    output_path, plot_halfwidth=250.0
):
    half = window_width / 2.0
    wmin, wmax = best_center - half, best_center + half

    nb_img = extract_pseudonb_image(cube, wmin, wmax, xpix, ypix, 3.0, pixscale)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]}
    )

    ax1.step(wave_centers, snr, where="mid", color="black", lw=1.0, alpha=0.6)
    ax1.plot(wave_centers, snr_smooth, color="crimson", lw=1.8)

    ax1.axhline(0.0, color="k", ls="--", lw=0.8)
    ax1.axvspan(wmin, wmax, color="lightblue", alpha=0.3, label="peak window (10 \u00c5)")

    ax1.axvline(search_min, color="royalblue", ls=":", lw=1.5)
    ax1.axvline(search_max, color="royalblue", ls=":", lw=1.5)
    ax1.axvline(lya_obs, color="orange", ls="--", lw=2.0,
                label=r"Ly$\alpha$ ($z_{\rm sys}$)")

    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    ax1.set_xlim(lya_obs - plot_halfwidth, lya_obs + plot_halfwidth)
    ax1.set_xlabel("Wavelength [\u00c5]")
    ax1.set_ylabel("S/N")

    ax1.text(
        0.05, 0.95, f"Peak S/N = {peak_snr:.2f}",
        transform=ax1.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8)
    )
    ax1.set_title(
        f"Source {source_id} | RA={ra:.5f} | Dec={dec:.5f} | Aperture {aperture_arcsec}\""
    )

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
# Plotting: FLUX
# ============================================================

def plot_flux_with_pseudonb(
    wave, flux, var,
    best_center, window_width,
    cube, xpix, ypix,
    aperture_arcsec, pixscale,
    ra, dec, source_id,
    lya_obs, search_min, search_max,
    output_path, smooth_sigma=2, plot_halfwidth=250.0
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
    ax1.axvspan(wmin, wmax, color="lightblue", alpha=0.3, label="peak window (10 \u00c5)")
    ax1.axvline(search_min, color="royalblue", ls=":", lw=1.5)
    ax1.axvline(search_max, color="royalblue", ls=":", lw=1.5)
    ax1.axvline(lya_obs, color="orange", ls="--", lw=2.0,
                label=r"Ly$\alpha$ ($z_{\rm sys}$)")

    ax1.set_xlim(lya_obs - plot_halfwidth, lya_obs + plot_halfwidth)

    # y-limits from the flux and noise within the plotted window only, so a
    # bright feature far outside +/-plot_halfwidth does not squash the line
    in_win = (wave >= lya_obs - plot_halfwidth) & (wave <= lya_obs + plot_halfwidth)
    if np.any(in_win):
        vals = np.concatenate([
            flux_smooth[in_win], flux[in_win], noise[in_win], -noise[in_win],
        ])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax1.set_ylim(lo - pad, hi + pad)

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
    p = argparse.ArgumentParser()

    p.add_argument("--indir", required=True,
                   help="Directory of extracted {ID}_spectrum.npz files")
    p.add_argument("--cube-dir", required=True)
    p.add_argument("--catalog", required=True,
                   help="Catalogue with ID, ra, dec, z_av columns")

    p.add_argument("--id", type=int, default=None,
                   help="Single source ID to process. If omitted, run all.")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--z-col", default="z_sys")
    p.add_argument("--z-col-fallback", default="z_dja",
                   help="Column used where z-col is blank (default z_dja). "
                        "Set to '' to disable.")

    p.add_argument("--window-width", type=float, default=10.0,
                   help="Sliding integration window width (AA)")
    p.add_argument("--step", type=float, default=1.0,
                   help="Sliding window step (AA)")
    p.add_argument("--dv-blue", type=float, default=300.0,
                   help="Blue half-width (km/s) of the peak search band, used "
                        "only if the .npz lacks a recorded band. Default 300.")
    p.add_argument("--dv-red", type=float, default=1200.0,
                   help="Red half-width (km/s) of the peak search band, used "
                        "only if the .npz lacks a recorded band. Default 1200.")

    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--pixscale", type=float, default=0.2)
    p.add_argument("--plot-halfwidth", type=float, default=250.0,
                   help="Half-width (AA) of the plotted x-axis window, centred "
                        "on systemic Lya. Default 250. The extracted spectrum "
                        "itself is unchanged, this only sets the plot xlim.")
    p.add_argument("--data-ext", type=int, default=1)
    p.add_argument("--var-ext", type=int, default=2)

    p.add_argument("--outfile",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    indir = os.path.abspath(args.indir)
    cube_dir = os.path.abspath(args.cube_dir)
    catalog_path = os.path.abspath(args.catalog)
    outfile = os.path.abspath(args.outfile)
    figdir = os.path.abspath(args.figdir)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    print("[PATHS]")
    print(f"  Spectra directory : {indir}")
    print(f"  Cube directory    : {cube_dir}")
    print(f"  Catalogue         : {catalog_path}")
    print(f"  Output CSV        : {outfile}")
    print(f"  Figure directory  : {figdir}")
    print("")

    catalog = pd.read_csv(catalog_path)
    for col in (args.id_col, "ra", "dec", args.z_col):
        if col not in catalog.columns:
            raise KeyError(
                f"Column '{col}' not found in catalogue. Available: {list(catalog.columns)}"
            )
    use_fallback = bool(args.z_col_fallback)
    if use_fallback and args.z_col_fallback not in catalog.columns:
        raise KeyError(
            f"Fallback column '{args.z_col_fallback}' not found. "
            f"Pass --z-col-fallback '' to disable. Available: {list(catalog.columns)}"
        )
    catalog[args.id_col] = catalog[args.id_col].astype("Int64")
    catalog = catalog.set_index(args.id_col)

    if args.id is not None:
        files = [os.path.join(indir, f"{args.id}_spectrum.npz")]
    else:
        files = sorted(glob.glob(os.path.join(indir, "*_spectrum.npz")))

    rows = []

    for path in files:
        if not os.path.exists(path):
            print(f"[SKIP] Missing spectrum: {path}")
            continue

        idx = int(os.path.basename(path).split("_")[0])
        if idx not in catalog.index:
            print(f"[SKIP] Src {idx}: not in catalogue")
            continue

        z, z_src = resolve_redshift(catalog.loc[idx], args.z_col,
                                    args.z_col_fallback)
        if not np.isfinite(z):
            print(f"[SKIP] Src {idx}: no finite {args.z_col}"
                  f"{' or ' + args.z_col_fallback if use_fallback else ''}")
            continue

        data = np.load(path)
        wave, flux, var = data["wave"], data["flux"], data["var"]
        lya_obs = float(data["lya_obs"])
        ra, dec = float(data["ra"]), float(data["dec"])

        # Prefer the search band recorded by the extractor. Older .npz files
        # without it fall back to computing the band from systemic here.
        if "band_min" in data.files and "band_max" in data.files:
            search_min = float(data["band_min"])
            search_max = float(data["band_max"])
        else:
            search_min, search_max = velocity_band(
                lya_obs, args.dv_blue, args.dv_red)

        wave_centers, snr_vals = sliding_snr(
            wave, flux, var,
            window_width=args.window_width, step=args.step
        )
        snr_smooth = smooth_snr_optimal(snr_vals)

        mask = (wave_centers >= search_min) & (wave_centers <= search_max)
        if not np.any(mask):
            print(f"[SKIP] Src {idx}: search window outside spectrum")
            continue

        imax = np.argmax(snr_smooth[mask])
        best_center = wave_centers[mask][imax]
        peak_snr = snr_smooth[mask][imax]

        cube_path = os.path.join(
            cube_dir, f"source_{idx}_lya_contsub_cube_velocity.fits"
        )
        if not os.path.exists(cube_path):
            print(f"[SKIP] Src {idx}: cube not found at {cube_path}")
            continue

        cube = Cube(cube_path, ext=(args.data_ext, args.var_ext))
        ypix, xpix = cube.wcs.sky2pix([[dec, ra]], unit="deg")[0]

        plot_snr_with_pseudonb(
            wave_centers, snr_vals, snr_smooth,
            best_center, peak_snr, args.window_width,
            cube, xpix, ypix,
            args.aperture, args.pixscale,
            ra, dec, idx,
            lya_obs, search_min, search_max,
            os.path.join(figdir, f"{idx}_snr_pseudonb.png"),
            plot_halfwidth=args.plot_halfwidth,
        )

        plot_flux_with_pseudonb(
            wave, flux, var,
            best_center, args.window_width,
            cube, xpix, ypix,
            args.aperture, args.pixscale,
            ra, dec, idx,
            lya_obs, search_min, search_max,
            os.path.join(figdir, f"{idx}_flux_pseudonb.png"),
            plot_halfwidth=args.plot_halfwidth,
        )

        rows.append(dict(
            ID=idx, ra=ra, dec=dec,
            peak_snr=peak_snr, best_center=best_center,
            lya_obs=lya_obs,
        ))
        print(f"[OK] Src {idx}: peak S/N={peak_snr:.2f} at {best_center:.1f} AA")

    pd.DataFrame(rows).to_csv(outfile, index=False)
    print("")
    print(f"[DONE] Saved {outfile}")


if __name__ == "__main__":
    main()


"""
run on all sources (no slurm needed)
python sliding_snr_lya_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --aperture 0.6

run on one source
python sliding_snr_lya_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --id 15479 \
  --aperture 0.6
"""