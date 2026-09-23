#!/usr/bin/env python3
"""
Ly-alpha skewed-Gaussian fitter and velocity-offset (Delta_v) measurement.

For each extracted 1D spectrum (the {ID}_spectrum.npz written by
ap_extract_specs_grating.py) this script:
  - fits a skew-normal profile to the Lya line, over a window centred on the
    sliding-S/N peak (best_center from lya_sliding_snr_grating.csv),
  - measures the line flux and FWHM (km/s),
  - defines z_lya from the PEAK of the fitted skewed profile (the standard
    convention for Lya velocity offsets),
  - computes the velocity offset from systemic,
        Delta_v = c (z_lya - z_sys) / (1 + z_sys),
    positive meaning Lya is redshifted relative to systemic.

The skewed-Gaussian machinery matches Patrick et al. (Paper 1): a skewnorm.pdf
normalised so the fitted amplitude parameter is the integrated line flux.

z_sys is read from the catalogue 'z_sys' column, falling back to 'z_dja' where
z_sys is blank, consistent with the rest of the P2 pipeline.

Outputs
  - one CSV (default /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv)
  - one fit figure per source, saved alongside the flux/SNR PNGs, named
    {ID}_lya_fit.png, showing the spectrum, the skewed-Gaussian model, and both
    the systemic (z_sys) and Lya-peak (z_lya) wavelengths.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LYA_REST = 1215.67       # AA
C_KMS = 299792.458


# ============================================================
# Redshift helper (matches the rest of the P2 pipeline)
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


# ============================================================
# Skewed-Gaussian model (as Paper 1)
# ============================================================

def skew_model(wave, flux_total, mu, sigma, alpha):
    """Skew-normal profile. skewnorm.pdf integrates to 1, so flux_total is the
    integrated line flux."""
    return flux_total * skewnorm.pdf(wave, alpha, loc=mu, scale=sigma)


def compute_fwhm_from_model(mu, sigma, alpha):
    """FWHM in Angstrom of the skewed-Gaussian profile, found numerically."""
    grid = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 4000)
    prof = skew_model(grid, 1.0, mu, sigma, alpha)
    peak = np.nanmax(prof)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    above = prof >= 0.5 * peak
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    return float(grid[idx[-1]] - grid[idx[0]])


def compute_peak_from_model(mu, sigma, alpha):
    """Wavelength (AA) of maximum flux in the skewed-Gaussian profile."""
    grid = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 4000)
    prof = skew_model(grid, 1.0, mu, sigma, alpha)
    if not np.any(np.isfinite(prof)):
        return np.nan
    return float(grid[int(np.nanargmax(prof))])


def fit_skewed_gaussian(wave, flux, var, centre,
                        fit_window=25.0, sigma_min=1.0, alpha_max=8.0):
    """Fit a skew-normal profile within +/- fit_window of centre.

    Returns a dict of fit parameters and derived quantities. On failure the
    numeric fields are NaN and fit_success is False.
    """
    fail = dict(flux_fit=np.nan, flux_fit_err=np.nan, fwhm_ang=np.nan,
                fwhm_kms=np.nan, alpha_skew=np.nan, mu=np.nan, sigma=np.nan,
                lya_peak=np.nan, fit_success=False)

    sel = (wave >= centre - fit_window) & (wave <= centre + fit_window)
    sel &= np.isfinite(flux) & np.isfinite(var) & (var > 0)
    if sel.sum() < 5:
        return fail

    w = wave[sel]
    f = flux[sel]
    sig = np.sqrt(np.clip(var[sel], 1e-30, None))

    sigma0 = max(sigma_min * 1.5, 2.5)
    amp0 = max(np.nanmax(f) * np.sqrt(2.0 * np.pi) * sigma0, 1e-20)
    p0 = [amp0, centre, sigma0, 2.0]
    bounds = (
        [0.0, centre - 10.0, sigma_min, 0.0],
        [np.inf, centre + 10.0, 50.0, alpha_max],
    )

    try:
        popt, pcov = curve_fit(
            skew_model, w, f, p0=p0, sigma=sig, absolute_sigma=True,
            bounds=bounds, maxfev=20000,
        )
    except Exception:
        return fail

    flux_fit, mu_fit, sigma_fit, alpha_fit = popt
    flux_err = float(np.sqrt(np.clip(pcov[0, 0], 0.0, np.inf)))

    fwhm_ang = compute_fwhm_from_model(mu_fit, sigma_fit, alpha_fit)
    fwhm_kms = (fwhm_ang / mu_fit) * C_KMS if np.isfinite(fwhm_ang) else np.nan
    lya_peak = compute_peak_from_model(mu_fit, sigma_fit, alpha_fit)

    return dict(
        flux_fit=float(flux_fit), flux_fit_err=flux_err,
        fwhm_ang=float(fwhm_ang) if np.isfinite(fwhm_ang) else np.nan,
        fwhm_kms=float(fwhm_kms) if np.isfinite(fwhm_kms) else np.nan,
        alpha_skew=float(alpha_fit), mu=float(mu_fit), sigma=float(sigma_fit),
        lya_peak=float(lya_peak) if np.isfinite(lya_peak) else np.nan,
        fit_success=True,
    )


# ============================================================
# Velocity offset
# ============================================================

def delta_v_from_z(z_lya, z_sys):
    """Velocity offset in km/s, positive when Lya is redward of systemic."""
    if not (np.isfinite(z_lya) and np.isfinite(z_sys)):
        return np.nan
    return C_KMS * (z_lya - z_sys) / (1.0 + z_sys)


def assess_quality(fit, dv, lya_snr, centre, dwave, fit_window,
                   dv_blue, dv_red, min_fwhm_kms, alpha_max):
    """Return (quality_flag, reason_string) from fit-reliability checks.

    quality_flag is 1 when every check passes, 0 if any trips. reason lists the
    checks that tripped, so a 0 can be understood without re-running. S/N is
    deliberately not part of this, it is carried separately by snr_flag, so a
    faint but reliable line keeps quality_flag=1.
    """
    reasons = []

    if not fit["fit_success"]:
        reasons.append("fit_failed")

    # Delta_v outside the physical band. best_center is inside the band by
    # construction, but the fitted profile peak (which sets z_lya) can drift.
    if np.isfinite(dv) and (dv < -dv_blue or dv > dv_red):
        reasons.append("dv_out_of_band")

    # Suspiciously narrow line, at or below the instrumental resolution.
    if np.isfinite(fit["fwhm_kms"]) and fit["fwhm_kms"] < min_fwhm_kms:
        reasons.append("narrow_fwhm")

    # Skew pinned at the fit bound means the fit railed rather than converged.
    if np.isfinite(fit["alpha_skew"]) and fit["alpha_skew"] >= 0.999 * alpha_max:
        reasons.append("skew_pinned")

    # Peak within one wavelength pixel of the fit-window edge.
    if (np.isfinite(fit["lya_peak"]) and np.isfinite(centre)
            and np.isfinite(dwave)):
        if abs(fit["lya_peak"] - centre) > (fit_window - dwave):
            reasons.append("peak_at_edge")

    quality_flag = 1 if not reasons else 0
    return quality_flag, ";".join(reasons) if reasons else "ok"


# ============================================================
# Plotting
# ============================================================

def plot_fit(wave, flux, var, fit, lya_sys, lya_peak, source_id,
             ra, dec, delta_v, outpath, plot_halfwidth=250.0):
    """Spectrum with the skewed-Gaussian fit, systemic and Lya-peak lines."""
    noise = np.sqrt(np.clip(var, 0.0, np.inf))

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.step(wave, flux, where="mid", color="black", lw=1.0, alpha=0.6,
            label="Flux")
    ax.fill_between(wave, -noise, noise, color="gray", alpha=0.3, step="mid",
                    label=r"$\pm 1\sigma$")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)

    if fit["fit_success"] and np.isfinite(fit["mu"]):
        model = skew_model(wave, fit["flux_fit"], fit["mu"], fit["sigma"],
                           fit["alpha_skew"])
        ax.plot(wave, model, color="crimson", lw=1.8, label="Skewed Gaussian")

    ax.axvline(lya_sys, color="orange", ls="--", lw=1.8,
               label=r"Ly$\alpha$ ($z_{\rm sys}$)")
    if np.isfinite(lya_peak):
        ax.axvline(lya_peak, color="green", ls="--", lw=1.8,
                   label=r"Ly$\alpha$ ($z_{\rm Ly\alpha}$)")

    txt = []
    if np.isfinite(delta_v):
        txt.append(rf"$\Delta v = {delta_v:.0f}$ km/s")
    if np.isfinite(fit["fwhm_kms"]):
        txt.append(rf"FWHM $= {fit['fwhm_kms']:.0f}$ km/s")
    if txt:
        ax.text(0.03, 0.95, "\n".join(txt), transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))

    ax.set_xlim(lya_sys - plot_halfwidth, lya_sys + plot_halfwidth)

    in_win = (wave >= lya_sys - plot_halfwidth) & (wave <= lya_sys + plot_halfwidth)
    if np.any(in_win):
        vals = np.concatenate([flux[in_win], noise[in_win], -noise[in_win]])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Wavelength [\u00c5]")
    ax.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ " + "\u00c5" + r"$^{-1}$]")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title(f"Source {source_id} | RA={ra:.5f} | Dec={dec:.5f}")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit skewed Gaussian to Lya, measure flux/FWHM/z_lya and "
                    "the velocity offset from systemic.")
    p.add_argument("--indir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts",
                   help="Directory of {ID}_spectrum.npz files.")
    p.add_argument("--snr-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv",
                   help="Sliding-S/N CSV providing best_center per ID.")
    p.add_argument("--catalog",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv",
                   help="Catalogue with ID, z_sys columns.")
    p.add_argument("--outfile",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv",
                   help="Output CSV of Lya properties.")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures",
                   help="Where fit PNGs go (same folder as flux/SNR PNGs).")

    p.add_argument("--id", type=int, default=None,
                   help="Single source ID to fit. If omitted, fit all.")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--z-col", default="z_sys")
    p.add_argument("--z-col-fallback", default="z_dja",
                   help="Column used where z-col is blank (default z_dja). "
                        "Set to '' to disable.")

    p.add_argument("--fit-window", type=float, default=25.0,
                   help="Fit half-width in AA about best_center (default 25).")
    p.add_argument("--sigma-min", type=float, default=1.0,
                   help="Minimum sigma (AA) for the fit.")
    p.add_argument("--alpha-max", type=float, default=8.0,
                   help="Maximum skew parameter alpha.")
    p.add_argument("--plot-halfwidth", type=float, default=250.0,
                   help="Half-width (AA) of the plotted x-axis window about "
                        "systemic Lya (default 250).")
    p.add_argument("--dv-blue", type=float, default=300.0,
                   help="Blue edge (km/s) of the physical Delta_v band for the "
                        "quality flag. Default 300, matches the pipeline.")
    p.add_argument("--dv-red", type=float, default=1200.0,
                   help="Red edge (km/s) of the physical Delta_v band for the "
                        "quality flag. Default 1200, matches the pipeline.")
    p.add_argument("--min-fwhm-kms", type=float, default=100.0,
                   help="Flag lines narrower than this FWHM (km/s) as suspect. "
                        "Default 100, approx MUSE instrumental resolution.")
    p.add_argument("--snr-threshold", type=float, default=5.0,
                   help="lya_snr at or above this sets snr_flag=1 (default 5).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the per-source fit figures.")
    return p.parse_args()


def main():
    args = parse_args()

    indir = os.path.abspath(args.indir)
    outfile = os.path.abspath(args.outfile)
    figdir = os.path.abspath(args.figdir)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if not args.no_plots:
        os.makedirs(figdir, exist_ok=True)

    print("[CONFIG]")
    print(f"  Spectra directory : {indir}")
    print(f"  Sliding-S/N CSV   : {os.path.abspath(args.snr_csv)}")
    print(f"  Catalogue         : {os.path.abspath(args.catalog)}")
    print(f"  Output CSV        : {outfile}")
    print(f"  Figure directory  : {figdir if not args.no_plots else '(plots off)'}")
    print("")

    # z_sys catalogue
    catalog = pd.read_csv(args.catalog)
    for col in (args.id_col, args.z_col):
        if col not in catalog.columns:
            raise KeyError(
                f"Column '{col}' not found in catalogue. Available: {list(catalog.columns)}")
    use_fallback = bool(args.z_col_fallback)
    if use_fallback and args.z_col_fallback not in catalog.columns:
        raise KeyError(
            f"Fallback column '{args.z_col_fallback}' not found. "
            f"Pass --z-col-fallback '' to disable. Available: {list(catalog.columns)}")
    catalog[args.id_col] = catalog[args.id_col].astype("Int64")
    catalog = catalog.set_index(args.id_col)

    # best_center per ID from the sliding-S/N run
    snr = pd.read_csv(args.snr_csv)
    if "ID" not in snr.columns or "best_center" not in snr.columns:
        raise KeyError(
            f"Sliding-S/N CSV must have ID and best_center columns. "
            f"Available: {list(snr.columns)}")
    snr["ID"] = snr["ID"].astype("Int64")
    best_center_by_id = snr.set_index("ID")["best_center"].to_dict()
    peak_snr_by_id = (snr.set_index("ID")["peak_snr"].to_dict()
                      if "peak_snr" in snr.columns else {})
    if not peak_snr_by_id:
        print("[WARN] sliding-S/N CSV has no peak_snr column, "
              "lya_snr and snr_flag will be NaN/0.")

    if args.id is not None:
        files = [os.path.join(indir, f"{args.id}_spectrum.npz")]
    else:
        files = sorted(glob.glob(os.path.join(indir, "*_spectrum.npz")))

    rows = []
    for path in files:
        if not os.path.exists(path):
            print(f"[SKIP] missing {path}")
            continue
        idx = int(os.path.basename(path).split("_")[0])

        if idx not in catalog.index:
            print(f"[SKIP] Src {idx}: not in catalogue")
            continue

        z_sys, z_src = resolve_redshift(catalog.loc[idx], args.z_col,
                                        args.z_col_fallback)
        if not np.isfinite(z_sys):
            print(f"[SKIP] Src {idx}: no finite {args.z_col}"
                  f"{' or ' + args.z_col_fallback if use_fallback else ''}")
            continue
        if z_src != args.z_col:
            print(f"[INFO] Src {idx}: {args.z_col} blank, using {z_src}={z_sys:.4f}")

        data = np.load(path)
        wave = data["wave"]
        flux = data["flux"]
        var = data["var"]
        ra = float(data["ra"])
        dec = float(data["dec"])
        lya_sys = float(data["lya_obs"])

        # Centre the fit on the sliding-S/N peak, falling back to systemic
        centre = best_center_by_id.get(idx, np.nan)
        if not np.isfinite(centre):
            print(f"[INFO] Src {idx}: no best_center, centring fit on systemic")
            centre = lya_sys

        fit = fit_skewed_gaussian(
            wave, flux, var, centre,
            fit_window=args.fit_window,
            sigma_min=args.sigma_min,
            alpha_max=args.alpha_max,
        )

        lya_peak = fit["lya_peak"]
        z_lya = (lya_peak / LYA_REST) - 1.0 if np.isfinite(lya_peak) else np.nan
        dv = delta_v_from_z(z_lya, z_sys)

        lya_snr = float(peak_snr_by_id.get(idx, np.nan))
        snr_flag = 1 if (np.isfinite(lya_snr) and lya_snr >= args.snr_threshold) else 0

        dwave = float(np.median(np.diff(wave))) if wave.size > 1 else np.nan
        quality_flag, quality_reason = assess_quality(
            fit, dv, lya_snr, centre, dwave, args.fit_window,
            args.dv_blue, args.dv_red, args.min_fwhm_kms, args.alpha_max,
        )

        if not args.no_plots:
            outpng = os.path.join(figdir, f"{idx}_lya_fit.png")
            plot_fit(wave, flux, var, fit, lya_sys, lya_peak, idx,
                     ra, dec, dv, outpng, plot_halfwidth=args.plot_halfwidth)

        rows.append(dict(
            ID=idx, ra=ra, dec=dec,
            z_sys=z_sys, z_sys_source=z_src,
            z_lya=z_lya, delta_v_kms=dv,
            lya_snr=lya_snr, snr_flag=snr_flag,
            quality_flag=quality_flag, quality_reason=quality_reason,
            lya_sys_wave=lya_sys, lya_peak_wave=lya_peak,
            best_center=centre,
            flux_fit=fit["flux_fit"], flux_fit_err=fit["flux_fit_err"],
            fwhm_ang=fit["fwhm_ang"], fwhm_kms=fit["fwhm_kms"],
            alpha_skew=fit["alpha_skew"], mu=fit["mu"], sigma=fit["sigma"],
            fit_success=fit["fit_success"],
        ))

        if fit["fit_success"]:
            print(f"[OK] Src {idx}: dv={dv:.0f} km/s  FWHM={fit['fwhm_kms']:.0f} "
                  f"km/s  S/N={lya_snr:.1f}  Q={quality_flag} ({quality_reason})")
        else:
            print(f"[FAIL] Src {idx}: fit did not converge")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(outfile, index=False)
    print("")
    print(f"[DONE] Fitted {len(rows)} sources, "
          f"{int(out_df['fit_success'].sum()) if len(rows) else 0} successful.")
    print(f"[DONE] Saved {outfile}")


if __name__ == "__main__":
    main()


"""
run on all sources
python fit_lya_properties_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --snr-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --outfile /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv \
  --figdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures

run on one source
python fit_lya_properties_grating.py --id 46212
"""