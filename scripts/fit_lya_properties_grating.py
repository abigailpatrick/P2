#!/usr/bin/env python3
"""
fit_lya_properties_grating.py

Fit the Lya line in every aperture spectrum and measure z_lya, Delta_v, flux
and width, following Patrick et al. (Paper 1, Sect. 3.3 and 3.4). Every
source is fitted. Detections are decided later by comparing lya_snr with the
false-positive thresholds.

Detection S/N (Paper 1, Sect. 3.3.3)
-------------------------------------
lya_snr is Sigma_Lya of Paper 1, the integrated flux in a 10 AA window over
its propagated error, at the window position that maximises it inside the
Lya band. It is measured by sliding_snr_lya_grating.py (p2_common.best_centre)
and copied here. The 10 AA window at best_center is the detection window.

The fit (Paper 1, Sect. 3.4, with the MUSE LSF added)
------------------------------------------------------
A skew-normal is fitted within +/- fit_window (25 AA) of best_center,
normalised so the amplitude is the integrated line flux. Unlike Paper 1 the
model is convolved with the MUSE LSF (Bacon et al. 2017, Eq. 7),
    FWHM_LSF(lambda) = 5.835e-8 lambda^2 - 9.080e-4 lambda + 5.983   [AA, air]
so the fitted (mu, sigma, alpha) describe the intrinsic line and FWHM_int is
LSF-corrected. This replaces the sigma >= 1 AA floor of Paper 1. A
skew-normal convolved with a Gaussian of width s is another skew-normal with
    sigma_obs = sqrt(sigma^2 + s^2)
    alpha_obs = alpha sigma / sqrt(sigma^2 + s^2 (1 + alpha^2))
z_lya comes from the peak of the convolved model (--peak obs), moved from air
to vacuum before it is compared with z_sys,
    Delta_v = c (z_lya - z_sys) / (1 + z_sys).

Flux check (Paper 1, Sect. 3.4)
-------------------------------
As in Paper 1, the fitted flux is checked against a direct integration of
the spectrum over the line window. The line window is the central 90 per
cent (--line-frac) of the fitted convolved model's flux, snapped to whole
channels, so flux_line should be about 0.9 flux_fit for a good fit.

Pseudo-NB (Paper 1, Fig. 3)
---------------------------
The continuum-subtracted cube collapsed over the detection window, shown as a
3 arcsec cutout. --nb-window line collapses it over the line window instead.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, z_sys, z_sys_err
  MUSE_catalogs/lya_sliding_snr.csv                 best_center, peak_snr_slide
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz
  MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits     for the pseudo-NB

Outputs
-------
  MUSE_catalogs/lya_fits.csv          one row per source
  MUSE_subcubes/<ID>/<ID>_lya_fit.png
      the spectrum with the fit and the detection window shaded (left), and
      the 3 arcsec pseudo-NB (right)
  MUSE_subcubes/<ID>/<ID>_lya_nb.fits
      [1] NB flux (1e-20 erg/s/cm2 per spaxel), [2] its variance, with the
      window in the header

A full run replaces the CSV. --ids patches only those rows.

Usage
-----
python fit_lya_properties_grating.py
python fit_lya_properties_grating.py --ids 23070 50803
python fit_lya_properties_grating.py --nb-window line
python fit_lya_properties_grating.py --plots-only                       # redraw figures, no refit
python fit_lya_properties_grating.py --plots-only --plot-halfwidth 60 --ids 31364
"""

import argparse
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p2_common as pc

OUTFILE = f"{pc.MUSE_CAT_DIR}/lya_fits.csv"
SLIDE_CSV = f"{pc.MUSE_CAT_DIR}/lya_sliding_snr.csv"
LINE_FRAC_DEFAULT = 0.9

LYA_REST = 1215.67       # AA, vacuum rest wavelength of Lya
C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 1 / 2.3548

# Fit defaults, imported by mc_lya_errors_grating.py and
# plot_lya_vs_systemic.py so every script fits with identical settings.
# SIGMA_MIN is now the lower bound on the INTRINSIC sigma. It no longer needs
# to stand in for the instrumental width (Paper 1 used 1.0 AA for that), the
# LSF convolution provides the floor.
SIGMA_MIN_DEFAULT = 0.1      # AA
SIGMA_MAX = 50.0             # AA
ALPHA_MAX_DEFAULT = 15.0
RESOLVE_FRAC_DEFAULT = 0.5
PEAK_DEFAULT = "obs"         # which model peak sets z_lya, 'obs' or 'int'

# Bacon et al. 2017 (A&A 608, A1) Eq. 7, MUSE mosaic LSF FWHM in AA,
# lambda in AA (air).
BACON17_LSF_COEFFS = (5.835e-8, -9.080e-4, 5.983)



# ============================================================
# Air and vacuum (kept here for the scripts that import them)
# ============================================================

def air_to_vac(wave_air):
    """Air to vacuum, Ciddor (1996). MUSE is in air, z_sys in vacuum, so the
    fitted MUSE peak is moved to vacuum before z_lya is formed."""
    return pc.air_to_vac(wave_air)


def vac_to_air(wave_vac):
    """Vacuum to air, the inverse of air_to_vac."""
    return pc.vac_to_air(wave_vac)


# ============================================================
# MUSE line spread function
# ============================================================

def muse_lsf_fwhm(wave_air):
    """MUSE LSF FWHM in AA at air wavelength(s) in AA, Bacon et al. 2017 Eq. 7."""
    a, b, c = BACON17_LSF_COEFFS
    w = np.asarray(wave_air, dtype=float)
    return a * w ** 2 + b * w + c


def muse_lsf_sigma(wave_air):
    """MUSE LSF Gaussian sigma in AA at air wavelength(s) in AA."""
    return muse_lsf_fwhm(wave_air) * FWHM_TO_SIGMA


# ============================================================
# Skewed-Gaussian model (as Paper 1), with LSF convolution
# ============================================================

def skew_model(wave, flux_total, mu, sigma, alpha):
    """Skew-normal profile. skewnorm.pdf integrates to 1, so flux_total is the
    integrated line flux. With the fitted parameters this is the INTRINSIC
    line, before the instrument."""
    return flux_total * skewnorm.pdf(wave, alpha, loc=mu, scale=sigma)


def convolved_skew_params(sigma, alpha, sigma_lsf):
    """Scale and shape of a skew-normal after convolution with a Gaussian of
    width sigma_lsf. The location (mu) and integral (flux) are unchanged.

    Exact, from the stochastic representation of the skew-normal:
        sigma_obs = sqrt(sigma^2 + sigma_lsf^2)
        alpha_obs = alpha sigma / sqrt(sigma^2 + sigma_lsf^2 (1 + alpha^2))
    """
    sigma_obs = np.sqrt(sigma ** 2 + sigma_lsf ** 2)
    alpha_obs = alpha * sigma / np.sqrt(sigma ** 2 + sigma_lsf ** 2 * (1.0 + alpha ** 2))
    return sigma_obs, alpha_obs


def skew_model_lsf(wave, flux_total, mu, sigma, alpha, sigma_lsf):
    """Intrinsic skew-normal convolved with the Gaussian LSF. This is the
    model that is compared with the MUSE data."""
    sigma_obs, alpha_obs = convolved_skew_params(sigma, alpha, sigma_lsf)
    return skew_model(wave, flux_total, mu, sigma_obs, alpha_obs)


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


def _to_kms(width_ang, wave):
    if not (np.isfinite(width_ang) and np.isfinite(wave) and wave > 0):
        return np.nan
    return float(width_ang / wave * C_KMS)


def fit_skewed_gaussian(wave, flux, var, centre,
                        fit_window=25.0, sigma_min=SIGMA_MIN_DEFAULT,
                        alpha_max=ALPHA_MAX_DEFAULT,
                        resolve_frac=RESOLVE_FRAC_DEFAULT,
                        peak=PEAK_DEFAULT):
    """Fit an LSF-convolved skew-normal within +/- fit_window of centre.

    The free parameters (flux, mu, sigma, alpha) are those of the intrinsic
    profile. The LSF sigma is fixed from the Bacon et al. 2017 polynomial at
    the fit centre, then re-evaluated at the fitted mu and the fit repeated
    once if they differ by more than 1e-3 AA (they almost never do, since the
    LSF changes by ~1e-4 AA per AA).

    Returns a dict of fit parameters and derived quantities. On failure the
    numeric fields are NaN and fit_success is False. 'lya_peak_int' is the peak
    of the intrinsic profile and 'lya_peak_obs' the peak of the convolved
    model, both in air AA. 'lya_peak' is whichever of the two 'peak' selects
    ('obs' by default), and is what sets z_lya downstream.
    """
    fail = dict(flux_fit=np.nan, flux_fit_err=np.nan,
                fwhm_int_ang=np.nan, fwhm_int_kms=np.nan,
                fwhm_obs_ang=np.nan, fwhm_obs_kms=np.nan,
                sigma_lsf_ang=np.nan, fwhm_lsf_kms=np.nan,
                alpha_skew=np.nan, mu=np.nan, sigma=np.nan,
                sigma_obs=np.nan, alpha_obs=np.nan,
                lya_peak=np.nan, lya_peak_int=np.nan, lya_peak_obs=np.nan,
                resolved=False, skew_at_max=False, fit_success=False)

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
        [np.inf, centre + 10.0, SIGMA_MAX, alpha_max],
    )

    sigma_lsf = float(muse_lsf_sigma(centre))
    popt = pcov = None
    for _ in range(2):
        def model(x, ftot, mu, sigma, alpha, _s=sigma_lsf):
            return skew_model_lsf(x, ftot, mu, sigma, alpha, _s)
        try:
            popt, pcov = curve_fit(
                model, w, f, p0=p0, sigma=sig, absolute_sigma=True,
                bounds=bounds, maxfev=20000,
            )
        except Exception:
            return fail
        sigma_lsf_new = float(muse_lsf_sigma(popt[1]))
        if abs(sigma_lsf_new - sigma_lsf) < 1e-3:
            break
        sigma_lsf = sigma_lsf_new
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        p0 = list(np.clip(popt, lo + 1e-6 * span, hi - 1e-6 * span))

    flux_fit, mu_fit, sigma_fit, alpha_fit = popt
    flux_err = float(np.sqrt(np.clip(pcov[0, 0], 0.0, np.inf)))
    sigma_obs, alpha_obs = convolved_skew_params(sigma_fit, alpha_fit, sigma_lsf)

    # Intrinsic profile, for science.
    fwhm_int_ang = compute_fwhm_from_model(mu_fit, sigma_fit, alpha_fit)
    lya_peak_int = compute_peak_from_model(mu_fit, sigma_fit, alpha_fit)
    # Convolved model, what MUSE sees.
    fwhm_obs_ang = compute_fwhm_from_model(mu_fit, sigma_obs, alpha_obs)
    lya_peak_obs = compute_peak_from_model(mu_fit, sigma_obs, alpha_obs)

    fwhm_lsf_ang = sigma_lsf / FWHM_TO_SIGMA
    at_floor = sigma_fit <= sigma_min * (1.0 + 1e-3)
    resolved = bool(np.isfinite(fwhm_int_ang) and not at_floor
                    and fwhm_int_ang >= resolve_frac * fwhm_lsf_ang)

    def _f(x):
        return float(x) if np.isfinite(x) else np.nan

    return dict(
        flux_fit=float(flux_fit), flux_fit_err=flux_err,
        fwhm_int_ang=_f(fwhm_int_ang), fwhm_int_kms=_to_kms(fwhm_int_ang, mu_fit),
        fwhm_obs_ang=_f(fwhm_obs_ang), fwhm_obs_kms=_to_kms(fwhm_obs_ang, mu_fit),
        sigma_lsf_ang=float(sigma_lsf), fwhm_lsf_kms=_to_kms(fwhm_lsf_ang, mu_fit),
        alpha_skew=float(alpha_fit), mu=float(mu_fit), sigma=float(sigma_fit),
        sigma_obs=float(sigma_obs), alpha_obs=float(alpha_obs),
        lya_peak=_f(lya_peak_obs if peak == "obs" else lya_peak_int),
        lya_peak_int=_f(lya_peak_int), lya_peak_obs=_f(lya_peak_obs),
        resolved=resolved,
        skew_at_max=bool(alpha_fit >= 0.999 * alpha_max),
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
                   dv_blue, dv_red, alpha_max):
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

    # Line not resolved by MUSE once the LSF is accounted for. This replaces
    # the old fixed 100 km/s 'narrow_fwhm' cut, which was below the MUSE
    # resolution at z < 5 (the LSF is ~180 km/s at z = 3, ~90 km/s at z = 6).
    if fit["fit_success"] and not fit["resolved"]:
        reasons.append("unresolved")

    # Intrinsic skew at alpha_max is NOT treated as a failure any more. With
    # the LSF in the model, alpha is the intrinsic skew, and alpha >~ 5 is
    # already close to a half-Gaussian (a blue edge sharper than MUSE can
    # resolve). Railing at alpha_max therefore means 'maximally asymmetric',
    # which is common for bright Lya, not 'fit did not converge'. It is
    # recorded in the skew_at_max column instead. In the old observed-frame
    # fit alpha could rail because it was fitting edges sharper than the LSF.

    # Peak within one wavelength pixel of the fit-window edge.
    if (np.isfinite(fit["lya_peak"]) and np.isfinite(centre)
            and np.isfinite(dwave)):
        if abs(fit["lya_peak"] - centre) > (fit_window - dwave):
            reasons.append("peak_at_edge")

    quality_flag = 1 if not reasons else 0
    return quality_flag, ";".join(reasons) if reasons else "ok"



# ============================================================
# Display
# ============================================================

def rebin_for_display(wave, flux, var, nbin):
    """Rebin a spectrum by an integer factor for display only.

    Wavelength is averaged per bin, flux is averaged, and variance is combined
    as the variance of a mean (sum of var / n^2) so the plotted +/-1 sigma band
    drops by ~sqrt(n). Any leftover channels that don't fill a final bin are
    dropped. The fit and all measured quantities are untouched, this affects
    only what is drawn.
    """
    nbin = int(nbin)
    if nbin <= 1:
        return wave, flux, var

    n = (wave.size // nbin) * nbin
    if n < nbin:
        return wave, flux, var

    w = wave[:n].reshape(-1, nbin)
    f = flux[:n].reshape(-1, nbin)
    v = var[:n].reshape(-1, nbin)

    with np.errstate(invalid="ignore"):
        wb = np.nanmean(w, axis=1)
        fb = np.nanmean(f, axis=1)
        # variance of the mean of nbin channels
        vb = np.nansum(v, axis=1) / (nbin ** 2)

    return wb, fb, vb



# ============================================================
# Detection window, line window and pseudo-NB
# ============================================================

def detection_window(wave, centre, width=pc.SLIDE_WINDOW):
    """Channels (k0, k1, inclusive) of the 10 AA detection window, the same
    channels p2_common.window_snr sums for lya_snr."""
    if not np.isfinite(centre):
        return None, None
    k = np.where((wave >= centre - width / 2) & (wave <= centre + width / 2))[0]
    if k.size < 2:
        return None, None
    return int(k[0]), int(k[-1])


def line_window(wave, fit, frac=LINE_FRAC_DEFAULT, centre=np.nan,
                fit_window=25.0, slide_window=pc.SLIDE_WINDOW):
    """Channel range (k0, k1, inclusive) holding the central frac of the
    fitted convolved model, and where it came from ('fit' or 'slide').

    Falls back to the sliding window about centre if the fit failed. The
    window is kept inside the fit range and has at least 2 channels.
    """
    if fit["fit_success"] and np.isfinite(fit["mu"]):
        q = (1.0 - frac) / 2.0
        lo, hi = skewnorm.ppf([q, 1.0 - q], fit["alpha_obs"],
                              loc=fit["mu"], scale=fit["sigma_obs"])
        src = "fit"
    elif np.isfinite(centre):
        lo, hi = centre - slide_window / 2, centre + slide_window / 2
        src = "slide"
    else:
        return None, None, "none"
    if np.isfinite(centre):
        lo, hi = max(lo, centre - fit_window), min(hi, centre + fit_window)
    k = np.where((wave >= lo) & (wave <= hi))[0]
    if k.size < 2:
        # Narrower than two channels, take the two nearest the window centre.
        mid = 0.5 * (lo + hi)
        k = np.sort(np.argsort(np.abs(wave - mid))[:2])
    return int(k[0]), int(k[-1]), src


def line_measure(wave, flux, var, k0, k1):
    """Directly integrated flux, its error and S/N over channels k0..k1."""
    sl = slice(k0, k1 + 1)
    dw = float(np.median(np.diff(wave)))
    f, v = flux[sl], var[sl]
    good = np.isfinite(f) & np.isfinite(v)
    if good.sum() < 2:
        return np.nan, np.nan, np.nan
    fl = float(np.sum(f[good]) * dw)
    er = float(np.sqrt(np.sum(v[good])) * dw)
    return fl, er, (fl / er if er > 0 else np.nan)


def narrowband(cube_path, k0, k1, dw):
    """NB flux and variance images summed over channels k0..k1, the 2D WCS
    header and the primary header of the cube. Read by slice with memmap so
    only those channels come off disk."""
    with fits.open(cube_path, memmap=True) as h:
        ph = h[0].header.copy()
        d = np.array(h[1].data[k0:k1 + 1], dtype=float)
        v = np.array(h[2].data[k0:k1 + 1], dtype=float)
        wcs_hdr = WCS(h[1].header).celestial.to_header()
    allnan = ~np.any(np.isfinite(d), axis=0)
    nb = np.nansum(d, axis=0) * dw
    nb_var = np.nansum(v, axis=0) * dw ** 2
    nb[allnan] = np.nan
    nb_var[allnan] = np.nan
    return nb, nb_var, wcs_hdr, ph


def write_nb(path, nb, nb_var, wcs_hdr, sid, wmin, wmax, src):
    hdr = wcs_hdr.copy()
    hdr["BUNIT"] = "1e-20 erg/s/cm2"
    hdr["P2_ID"] = sid
    hdr["NB_WMIN"] = (round(wmin, 3), "[AA, air] window start, channel edge")
    hdr["NB_WMAX"] = (round(wmax, 3), "[AA, air] window end, channel edge")
    hdr["NB_FROM"] = (src, "detect (10 AA) or line window")
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(nb.astype(np.float32), header=hdr, name="DATA"),
                  fits.ImageHDU(nb_var.astype(np.float32), header=hdr, name="STAT"),
                  ]).writeto(path, overwrite=True)


# ============================================================
# Figure
# ============================================================

def _ylim_from(ax, wave, arrays, lo_w, hi_w):
    """y-limits from the plotted range only, as in the JELS-ID plots."""
    inw = (wave >= lo_w) & (wave <= hi_w)
    vals = np.concatenate([a[inw] for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = vals.min(), vals.max()
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)


def plot_fit(path, sid, ra, dec, z_sys, wave, flux, var, fit, lya_sys, lya_peak, dv,
             det_win, lya_snr, nb, ap_xy, radius_pix, halfwidth=250.0, plot_bin=2,
             box_pix=15):
    """Spectrum with the fit and the detection window (left) and the pseudo-NB
    (right), in the style of the Paper 1 figures.

    The plotted model is the fitted line convolved with the LSF, i.e. what
    MUSE sees. The FWHM quoted is the intrinsic, LSF-corrected value.
    """
    wd, fd, vd = rebin_for_display(wave, flux, var, plot_bin)
    nd = np.sqrt(np.clip(vd, 0.0, np.inf))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    ax1.step(wd, fd, where="mid", color="black", lw=1.0, alpha=0.6,
             label="Flux" if plot_bin <= 1 else f"Flux (bin {int(plot_bin)})")
    ax1.fill_between(wd, -nd, nd, color="gray", alpha=0.3, step="mid", label=r"$\pm 1\sigma$")
    ax1.axhline(0.0, color="k", ls="--", lw=0.8)
    if det_win is not None:
        ax1.axvspan(*det_win, color="lightblue", alpha=0.4,
                    label=f"detection window ({pc.SLIDE_WINDOW:g} " + r"$\AA$)")
    if fit["fit_success"] and np.isfinite(fit["mu"]):
        ax1.plot(wave, skew_model_lsf(wave, fit["flux_fit"], fit["mu"], fit["sigma"],
                                      fit["alpha_skew"], fit["sigma_lsf_ang"]),
                 color="crimson", lw=1.8, label="Skewed Gaussian fit")
    ax1.axvline(lya_sys, color="orange", ls="--", lw=1.8, label=r"Ly$\alpha$ ($z_{\rm sys}$)")
    if np.isfinite(lya_peak):
        ax1.axvline(lya_peak, color="green", ls="--", lw=1.8,
                    label=r"Ly$\alpha$ ($z_{\rm Ly\alpha}$)")

    txt = []
    if np.isfinite(lya_snr):
        txt.append(rf"S/N = {lya_snr:.1f}")
    if np.isfinite(dv):
        txt.append(rf"$\Delta v = {dv:.0f}$ km/s")
    if np.isfinite(fit["fwhm_int_kms"]):
        rel = "=" if fit["resolved"] else r"$\leq$"
        txt.append(rf"FWHM {rel} {fit['fwhm_int_kms']:.0f} km/s")
    if txt:
        ax1.text(0.03, 0.95, "\n".join(txt), transform=ax1.transAxes, va="top", ha="left",
                 fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))

    ax1.set_xlim(lya_sys - halfwidth, lya_sys + halfwidth)
    inw = (wd >= lya_sys - halfwidth) & (wd <= lya_sys + halfwidth)
    vals = np.concatenate([fd[inw], nd[inw], -nd[inw]])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = vals.min(), vals.max()
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax1.set_ylim(lo - pad, hi + pad)
    ax1.set_xlabel(r"Wavelength [$\AA$]")
    ax1.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    ax1.set_title(f"Source {sid} | RA={ra:.5f} | Dec={dec:.5f}"
                  + rf" | $z_{{\rm sys}}={z_sys:.4f}$")

    # Pseudo-NB, 3 arcsec (15 pixels) centred on the extraction aperture
    h = box_pix // 2
    xc, yc = ap_xy
    cut = np.full((box_pix, box_pix), np.nan)
    if nb is not None:
        ys = slice(max(yc - h, 0), min(yc + h + 1, nb.shape[0]))
        xs = slice(max(xc - h, 0), min(xc + h + 1, nb.shape[1]))
        cut[ys.start - (yc - h):ys.stop - (yc - h),
            xs.start - (xc - h):xs.stop - (xc - h)] = nb[ys, xs]
    ax2.imshow(cut, origin="lower", cmap="inferno")
    ax2.add_patch(plt.Circle((h, h), radius_pix, edgecolor="cyan", facecolor="none", lw=2.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Pseudo-NB")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--slide-csv", default=SLIDE_CSV)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outfile", default=OUTFILE)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="fit half-width about best_center, AA (default 25)")
    p.add_argument("--line-frac", type=float, default=LINE_FRAC_DEFAULT,
                   help="fraction of the fitted model flux in the line window used "
                        "for the flux check (default 0.9)")
    p.add_argument("--nb-window", choices=["detect", "line"], default="detect",
                   help="collapse the pseudo-NB over the 10 AA detection window "
                        "(default, as Paper 1) or the line window")
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN_DEFAULT,
                   help="lower bound on the intrinsic sigma, AA")
    p.add_argument("--resolve-frac", type=float, default=RESOLVE_FRAC_DEFAULT,
                   help="resolved if FWHM_int >= this x FWHM_LSF (default 0.5)")
    p.add_argument("--alpha-max", type=float, default=ALPHA_MAX_DEFAULT)
    p.add_argument("--peak", choices=["obs", "int"], default=PEAK_DEFAULT,
                   help="model peak that sets z_lya (default obs)")
    p.add_argument("--dv-blue", type=float, default=300.0,
                   help="band edges for the dv_out_of_band flag, km/s")
    p.add_argument("--dv-red", type=float, default=1200.0)
    p.add_argument("--plot-halfwidth", type=float, default=150.0,
                   help="half-width of the plotted spectrum about systemic, AA (default 150)")
    p.add_argument("--plot-bin", type=int, default=2,
                   help="rebin the plotted spectrum by this factor (display only)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plots-only", action="store_true",
                   help="redraw <ID>_lya_fit.png from the saved fits in --outfile and the "
                        "saved <ID>_lya_nb.fits, no refitting, nothing else is written")
    return p.parse_args()


def fit_source(row, slide, args):
    sid = int(row["ID"])
    z_sys = float(row["z_sys"])
    z_sys_err = float(row["z_sys_err"]) if "z_sys_err" in row and pd.notna(row["z_sys_err"]) else np.nan
    spec_path = pc.muse_path(sid, "spectrum", args.subcube_dir)
    if not os.path.exists(spec_path):
        print(f"  {sid}  SKIP no spectrum at {spec_path}")
        return None
    d = np.load(spec_path)
    wave, flux, var = d["wave"], d["flux"], d["var"]
    lya_sys = pc.lya_obs_air(z_sys)

    centre = float(slide.loc[sid, "best_center"]) if sid in slide.index else np.nan
    lya_snr = float(slide.loc[sid, "peak_snr_slide"]) if sid in slide.index else np.nan
    det_k0, det_k1 = detection_window(wave, centre)
    if not np.isfinite(centre):
        print(f"  {sid}  no best_center, fit centred on Lya at z_sys")
        centre = lya_sys

    fit = fit_skewed_gaussian(wave, flux, var, centre, fit_window=args.fit_window,
                              sigma_min=args.sigma_min, alpha_max=args.alpha_max,
                              resolve_frac=args.resolve_frac, peak=args.peak)

    def peak_to_z(peak_air):
        if not np.isfinite(peak_air):
            return np.nan, np.nan, np.nan
        pv = float(air_to_vac(peak_air))
        zl = pv / LYA_REST - 1.0
        return pv, zl, delta_v_from_z(zl, z_sys)

    lya_peak = fit["lya_peak"]
    lya_peak_vac, z_lya, dv = peak_to_z(lya_peak)
    _, z_lya_obs, dv_obs = peak_to_z(fit["lya_peak_obs"])
    _, z_lya_int, dv_int = peak_to_z(fit["lya_peak_int"])

    dw = float(np.median(np.diff(wave)))
    quality_flag, quality_reason = assess_quality(
        fit, dv, np.nan, centre, dw, args.fit_window, args.dv_blue, args.dv_red, args.alpha_max)

    # Line window, for the flux check
    k0, k1, win_src = line_window(wave, fit, args.line_frac, centre, args.fit_window)
    if k0 is not None:
        win = (float(wave[k0] - dw / 2), float(wave[k1] + dw / 2))
        flux_line, flux_line_err, _ = line_measure(wave, flux, var, k0, k1)
    else:
        win, flux_line, flux_line_err = None, np.nan, np.nan
    det_win = ((float(wave[det_k0] - dw / 2), float(wave[det_k1] + dw / 2))
               if det_k0 is not None else None)

    # Pseudo-NB over the detection window (or the line window)
    nb_k = (det_k0, det_k1, det_win, "detect") if args.nb_window == "detect" \
        else (k0, k1, win, "line")
    cube_path = pc.muse_path(sid, "contsub", args.subcube_dir)
    nb = None
    nb_path = pc.muse_path(sid, "nb", args.subcube_dir)
    if nb_k[0] is not None and os.path.exists(cube_path):
        nb, nb_var, wcs_hdr, _ = narrowband(cube_path, nb_k[0], nb_k[1], dw)
        write_nb(nb_path, nb, nb_var, wcs_hdr, sid, nb_k[2][0], nb_k[2][1], nb_k[3])
        pixscale = float(np.mean(proj_plane_pixel_scales(WCS(wcs_hdr))) * 3600)
    else:
        nb_path = ""

    msg = (f"  {sid}  S/N {lya_snr:.2f}, "
           + (f"dv {dv:.0f} km/s, Q {quality_reason}" if fit["fit_success"] else "fit FAILED"))
    if nb_path:
        msg += f"\n         {nb_path}"
    if not args.no_plots:
        png = pc.muse_path(sid, "fit_png", args.subcube_dir)
        radius_pix = float(d["aperture"]) / pixscale if nb is not None else 3.0
        plot_fit(png, sid, float(d["ra"]), float(d["dec"]), z_sys, wave, flux, var, fit,
                 lya_sys, lya_peak, dv, det_win, lya_snr, nb, (int(d["x_pix"]), int(d["y_pix"])),
                 radius_pix, args.plot_halfwidth, args.plot_bin)
        msg += f"\n         {png}"
    print(msg, flush=True)

    return dict(
        ID=sid, z_sys=z_sys, z_sys_err=z_sys_err,
        z_lya=z_lya, delta_v_kms=dv,
        z_lya_obs=z_lya_obs, delta_v_obs_kms=dv_obs,
        z_lya_int=z_lya_int, delta_v_int_kms=dv_int, peak_used=args.peak,
        lya_snr=lya_snr, best_center=centre,
        det_win_min=det_win[0] if det_win else np.nan,
        det_win_max=det_win[1] if det_win else np.nan,
        flux_line=flux_line, flux_line_err=flux_line_err,
        line_win_min=win[0] if win else np.nan, line_win_max=win[1] if win else np.nan,
        line_win_from=win_src, line_frac=args.line_frac,
        quality_flag=quality_flag, quality_reason=quality_reason,
        lya_sys_wave_air=lya_sys, lya_peak_wave=lya_peak, lya_peak_wave_vac=lya_peak_vac,
        lya_peak_obs_wave=fit["lya_peak_obs"], lya_peak_int_wave=fit["lya_peak_int"],
        flux_fit=fit["flux_fit"], flux_fit_err=fit["flux_fit_err"],
        fwhm_int_ang=fit["fwhm_int_ang"], fwhm_int_kms=fit["fwhm_int_kms"],
        fwhm_obs_ang=fit["fwhm_obs_ang"], fwhm_obs_kms=fit["fwhm_obs_kms"],
        sigma_lsf_ang=fit["sigma_lsf_ang"], fwhm_lsf_kms=fit["fwhm_lsf_kms"],
        resolved=fit["resolved"], skew_at_max=fit["skew_at_max"],
        alpha_skew=fit["alpha_skew"], mu=fit["mu"], sigma=fit["sigma"],
        sigma_obs=fit["sigma_obs"], alpha_obs=fit["alpha_obs"],
        fit_success=fit["fit_success"],
        ra_src=float(d["ra_src"]), dec_src=float(d["dec_src"]),
        ra_ap=float(d["ra"]), dec_ap=float(d["dec"]),
        dx=float(d["dx"]), dy=float(d["dy"]), aperture=float(d["aperture"]),
    )


def plots_only(args):
    """Redraw <ID>_lya_fit.png from lya_fits.csv and the saved pseudo-NBs."""
    out_path = os.path.abspath(args.outfile)
    print(f"Reading fits        {out_path}")
    print(f"Writing             {os.path.abspath(args.subcube_dir)}/<ID>/<ID>_lya_fit.png "
          "(figures only)")
    if not os.path.exists(out_path):
        raise SystemExit(f"No {out_path}. Run the fit first")
    fits_df = pd.read_csv(out_path)
    fits_df["ID"] = fits_df["ID"].astype(int)
    if args.ids:
        fits_df = fits_df[fits_df["ID"].isin(args.ids)]
    n = 0
    for _, r in fits_df.iterrows():
        sid = int(r["ID"])
        spec_path = pc.muse_path(sid, "spectrum", args.subcube_dir)
        if not os.path.exists(spec_path):
            print(f"  {sid}  no spectrum, skipped")
            continue
        d = np.load(spec_path)
        wave, flux, var = d["wave"], d["flux"], d["var"]
        ok = str(r.get("fit_success", "")).lower() in ("true", "1")
        fit = dict(fit_success=ok, flux_fit=r.get("flux_fit", np.nan), mu=r.get("mu", np.nan),
                   sigma=r.get("sigma", np.nan), alpha_skew=r.get("alpha_skew", np.nan),
                   sigma_lsf_ang=r.get("sigma_lsf_ang", np.nan),
                   fwhm_int_kms=r.get("fwhm_int_kms", np.nan),
                   resolved=str(r.get("resolved", "")).lower() in ("true", "1"))
        det_win = ((float(r["det_win_min"]), float(r["det_win_max"]))
                   if np.isfinite(r.get("det_win_min", np.nan)) else None)
        nb, radius_pix = None, 3.0
        nb_path = pc.muse_path(sid, "nb", args.subcube_dir)
        if os.path.exists(nb_path):
            with fits.open(nb_path) as h:
                nb = np.array(h[1].data, dtype=float)
                pixscale = float(np.mean(proj_plane_pixel_scales(WCS(h[1].header).celestial)) * 3600)
            radius_pix = float(d["aperture"]) / pixscale
        z_sys = float(r["z_sys"])
        png = pc.muse_path(sid, "fit_png", args.subcube_dir)
        plot_fit(png, sid, float(d["ra"]), float(d["dec"]), z_sys, wave, flux, var, fit,
                 pc.lya_obs_air(z_sys), float(r.get("lya_peak_wave", np.nan)),
                 float(r.get("delta_v_kms", np.nan)), det_win, float(r.get("lya_snr", np.nan)),
                 nb, (int(d["x_pix"]), int(d["y_pix"])), radius_pix,
                 args.plot_halfwidth, args.plot_bin)
        n += 1
    print(f"\n  redrew {n} figures, +/-{args.plot_halfwidth:g} AA, bin {args.plot_bin}")
    print(f"Wrote  {os.path.abspath(args.subcube_dir)}/<ID>/<ID>_lya_fit.png")


def main():
    args = parse_args()
    print("fit_lya_properties_grating.py")
    if args.plots_only:
        plots_only(args)
        return
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")
    slide_path = os.path.abspath(args.slide_csv)
    print(f"Reading best_center {slide_path}")
    slide = pd.read_csv(slide_path)
    slide["ID"] = slide["ID"].astype(int)
    slide = slide.set_index("ID")
    print(f"Detection S/N from the {pc.SLIDE_WINDOW:g} AA sliding window (Paper 1), "
          f"fit +/-{args.fit_window:g} AA with the MUSE LSF, z_lya from the {args.peak} peak, "
          f"pseudo-NB over the {args.nb_window} window\n")

    rows = [r for r in (fit_source(row, slide, args) for _, row in df.iterrows()) if r]
    new = pd.DataFrame(rows)
    out_path = os.path.abspath(args.outfile)
    if args.ids and os.path.exists(out_path) and len(new):
        old = pd.read_csv(out_path)
        new = pd.concat([old[~old["ID"].isin(new["ID"])], new]).sort_values("ID")
    if len(new):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        new.to_csv(out_path, index=False)

    print("\n--- Summary ---")
    print(f"  fitted           {len(rows)}")
    print(f"  no spectrum      {len(df) - len(rows)}")
    if rows:
        r = pd.DataFrame(rows)
        print(f"  fit successful   {int(r['fit_success'].sum())}")
        print(f"  resolved         {int(r.loc[r['fit_success'], 'resolved'].sum())}")
        print(f"  quality ok       {int((r['quality_flag'] == 1).sum())}")
        print(f"\nWrote\n  {out_path}")


if __name__ == "__main__":
    main()