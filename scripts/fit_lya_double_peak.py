#!/usr/bin/env python3
"""
fit_lya_double_peak.py

A second Lya fit, with a blue peak added to the single fit, to find the
sources whose line is double peaked and to measure each peak's velocity
offset. Run it after fit_lya_properties_grating.py. build_lya_catalog.py then
uses whichever fit is preferred.

The double model
----------------
The single fit's LSF-convolved skewed Gaussian for the red peak, plus an
LSF-convolved Gaussian for a blue peak (MUSE LSF of Bacon et al. 2017, as the
single fit, so the widths are intrinsic):
  red peak   skewed Gaussian, location within best_center +/- 10 AA and
             alpha 0 to 15, exactly the single fit's limits
  blue peak  Gaussian, centre between -1000 and +100 km/s of systemic Lya
             (the 100 km/s allows for the z_sys error)
The single fit is the double fit with no blue flux, so the two models are
nested and the comparison asks only whether a blue peak is needed. A
symmetric red peak would lose on the red tail what it gained from the blue
peak, and could split one asymmetric line into a narrow and a broad peak.
The red peak's velocity is the peak of its LSF-convolved profile, as for the
single fit's delta_v. A fit whose peaks end up closer than --min-sep-kms
(100 km/s, about 0.6 of the MUSE resolution) is not a double peak and is
flagged peaks_merged. The fit uses the same pixels as the single fit
(best_center +/- 25 AA) and the best of several starts (the single fit plus
a blue peak at -500 to -150 km/s, and the single fit's peak taken as the blue
peak with a red peak 200 or 400 km/s redward).

Single or double: the BIC
-------------------------
BIC = chi2 + k ln(n), with n the number of pixels fitted and k the number of
free parameters (4 for the single skewed Gaussian, 7 for the double).
Lower is better. The ln(n) term charges each extra parameter, so the double
model only wins if it improves chi2 by more than the extra freedom alone
would. dbic = BIC_single - BIC_double, so positive favours the double. By the
usual scale (Kass & Raftery 1995) dbic > 2 is positive, > 6 strong and > 10
very strong evidence. Because the models are nested, the double needs chi2
lower by 3 ln(n) + dbic_min, about 13 for n = 40 pixels and dbic_min = 2.
On synthetic MUSE-like spectra (Sigma 7 to 18, 1.25 AA pixels) no single
line, including lines with long red tails, reached dbic > 2 (0 of 90),
while a blue peak with a quarter of the red flux at -330 to -420 km/s was
flagged in 17 of 30 at dbic > 2 and 10 of 30 at dbic > 6. A source is
flagged double_peak = 1 if
  dbic > --dbic-min (2, "positive" evidence),
  both peak fluxes are detected at >= --peak-snr-min (2) sigma (MC errors),
  the peaks are >= --min-sep-kms (100) apart, and
  the blue peak is blueward of systemic (dv_blue < 0).
The single fit's chi2 is recomputed here from its saved parameters on the
same pixels, so the two are compared on identical data.

Errors
------
--n-mc (200) parametric-bootstrap refits of the best double model with the
spectrum's noise, as mc_lya_errors_grating.py does for the single fit. The
peak velocities get the z_sys error added in quadrature, like delta_v_err.

Columns of the output CSV
-------------------------
  ID, z_sys, n_pix, chi2_single, chi2_double, bic_single, bic_double, dbic
  dv_blue, dv_blue_err      km/s, blue peak relative to systemic
  dv_red, dv_red_err        km/s, red peak relative to systemic
  dv_red_err_lya            km/s, the red peak's centre precision alone (no z_sys error)
  lam_red, lam_red_err      AA (observed, air), red peak centre
  z_lya_red, z_lya_red_err  redshift of the red peak (vacuum)
  peak_sep, peak_sep_err    km/s, dv_red - dv_blue
  flux_blue, flux_blue_err, flux_red, flux_red_err   1e-20 erg/s/cm2
  flux_total, flux_total_err    blue + red
  blue_red_ratio, blue_red_ratio_err
  fwhm_blue_kms, fwhm_red_kms (+ _err)   intrinsic (LSF-corrected) widths
  fwhm_red_obs_kms (+ _err)     the red peak as MUSE sees it
  fwhm_lsf_kms, red_resolved    LSF width at the red peak, and whether the red
                                peak is resolved (the single fit's rule, FWHM_int
                                >= 0.5 FWHM_LSF and sigma off its floor)
  double_peak               1 if the double model is preferred (see above)
  double_reason             why not, if 0: dbic, weak_peak, peaks_merged,
                            blue_not_blueward (the "blue" peak is at or
                            redward of systemic, e.g. a narrow core with a
                            broad red wing),
                            no_single_fit, fit_failed
  mu_blue, sigma_blue, mu_red, sigma_red, alpha_red, sigma_lsf_ang
                            the best-fit parameters (AA, air), so other scripts
                            can redraw the model with double_model()
  n_mc_success

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv   z_sys, z_sys_err
  MUSE_catalogs/lya_fits.csv                      best_center and the single fit
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz
  MUSE_subcubes/<ID>/<ID>_lya_nb.fits             pseudo-NB, for the figure

Outputs
-------
  MUSE_catalogs/lya_double_fits.csv                one row per source
  MUSE_subcubes/<ID>/<ID>_lya_double_fit.png       spectrum with both fits, and the NB

Usage
-----
python fit_lya_double_peak.py
python fit_lya_double_peak.py --ids 26917 59658          # replace only these rows
python fit_lya_double_peak.py --n-mc 0 --no-plots        # quick, no errors
python fit_lya_double_peak.py --plots-only               # redraw the figures, no refitting
python fit_lya_double_peak.py --plots-only --plot-halfwidth 60 --ids 31364   # zoomed in
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import curve_fit
from scipy.special import ndtr

import p2_common as pc
import fit_lya_properties_grating as flp

C_KMS = pc.C_KMS
OUT_CSV = f"{pc.MUSE_CAT_DIR}/lya_double_fits.csv"
FIG_NAME = pc.MUSE_FILES["double_fit_png"]
SIG_MAX_BLUE = 15.0     # AA, upper bound on the blue peak's intrinsic sigma
BLUE_RANGE = (-1000.0, 100.0)   # km/s about systemic
RED_MU_WINDOW = 10.0    # AA, red location within best_center +/- this, as the single fit
BLUE_STARTS = (-500.0, -350.0, -250.0, -150.0)   # km/s
RED_SHIFTS = (200.0, 400.0)     # km/s, red start redward of the single peak
K_SINGLE, K_DOUBLE = 4, 7


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=f"{pc.MUSE_CAT_DIR}/lya_fits.csv")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs, rows replaced in --out-csv")
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="half-width about best_center, AA, as the single fit (default 25)")
    p.add_argument("--dbic-min", type=float, default=2.0,
                   help="double preferred if BIC_single - BIC_double exceeds this "
                        "(default 2, see the docstring)")
    p.add_argument("--peak-snr-min", type=float, default=2.0)
    p.add_argument("--min-sep-kms", type=float, default=100.0)
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--plot-halfwidth", type=float, default=150.0,
                   help="half-width of the plotted spectrum about systemic, AA (default 150, "
                        "as fit_lya_properties_grating.py, 60 to zoom in on the peaks)")
    p.add_argument("--plot-bin", type=int, default=2)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plots-only", action="store_true",
                   help="redraw the figures from the saved fits in --out-csv, no refitting, "
                        "the CSV is not changed")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def gauss(x, flux, mu, sig):
    return flux / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def skew_pdf(x, flux, mu, sig, alpha):
    """flux x skew-normal pdf. The same numbers as scipy's skewnorm.pdf used by
    fit_lya_properties_grating.skew_model, about ten times faster, which
    matters for the MC refits."""
    t = (x - mu) / sig
    return flux * 2.0 / sig * np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi) * ndtr(alpha * t)


def skew_lsf(x, flux, mu, sig, alpha, s_lsf):
    """As fit_lya_properties_grating.skew_model_lsf."""
    so, ao = flp.convolved_skew_params(sig, alpha, s_lsf)
    return skew_pdf(x, flux, mu, so, ao)


def skew_peak_fwhm(mu, sig, alpha):
    """Peak and FWHM (AA) of a skew-normal, on the same 4000-point grid as
    fit_lya_properties_grating.compute_peak_from_model and compute_fwhm_from_model."""
    grid = np.linspace(mu - 10 * sig, mu + 10 * sig, 4000)
    prof = skew_pdf(grid, 1.0, mu, sig, alpha)
    i = int(np.argmax(prof))
    idx = np.where(prof >= 0.5 * prof[i])[0]
    return float(grid[i]), float(grid[idx[-1]] - grid[idx[0]])


def double_model(x, fb, mub, sb, fr, mur, sr, ar, s_lsf):
    """Blue intrinsic Gaussian plus red intrinsic skewed Gaussian, both
    convolved with a Gaussian LSF of sigma s_lsf. With fb = 0 it is the single
    fit's model."""
    return (gauss(x, fb, mub, np.hypot(sb, s_lsf))
            + skew_lsf(x, fr, mur, sr, ar, s_lsf))


def bounds_for(lam_sys, centre, sigma_min, alpha_max):
    lo_b, hi_b = [lam_sys * (1 + v / C_KMS) for v in BLUE_RANGE]
    lower = [0.0, lo_b, sigma_min, 0.0, centre - RED_MU_WINDOW, sigma_min, 0.0]
    upper = [np.inf, hi_b, SIG_MAX_BLUE, np.inf, centre + RED_MU_WINDOW, flp.SIGMA_MAX,
             alpha_max]
    return lower, upper


def fit_double(w, f, e, s_lsf, bounds, p0_list):
    """Best of several starts. Returns (popt, pcov, chi2) or None."""
    lower, upper = bounds

    def model(x, fb, mub, sb, fr, mur, sr, ar):
        return double_model(x, fb, mub, sb, fr, mur, sr, ar, s_lsf)

    lo, hi = np.array(lower), np.array(upper)
    span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    best = None
    for p0 in p0_list:
        p0 = np.clip(p0, lo + 1e-6 * span, hi - 1e-6 * span)
        try:
            popt, pcov = curve_fit(model, w, f, p0=p0, sigma=e, absolute_sigma=True,
                                   bounds=(lower, upper), maxfev=20000)
        except Exception:
            continue
        chi2 = float(np.sum(((f - model(w, *popt)) / e) ** 2))
        if best is None or chi2 < best[2]:
            best = (popt, pcov, chi2)
    return best


def red_peak(mur, sr, ar, s_lsf):
    """Peak (air AA) of the LSF-convolved red skewed Gaussian, as the single
    fit's lya_peak_obs."""
    so, ao = flp.convolved_skew_params(sr, ar, s_lsf)
    return skew_peak_fwhm(mur, so, ao)[0]


def starts_from(lam_sys, centre, w, f, single, alpha_max):
    """Starting points. Red from the single fit, with a blue peak at each of
    BLUE_STARTS. Then the single fit's peak taken as the blue peak, with a red
    peak RED_SHIFTS further out. Then the single fit itself with almost no
    blue flux, so the double can never end worse than the single."""
    if single is not None:
        F, mu, sig, alpha, s_lsf = single
        pk = red_peak(mu, sig, alpha, s_lsf)
    else:
        F = max(np.sum(np.clip(f, 0, None)) * np.median(np.diff(w)), 1.0)
        mu, sig, alpha, pk = centre, 2.0, 2.0, centre
    out = []
    for vb in BLUE_STARTS:
        out.append([0.2 * F, lam_sys * (1 + vb / C_KMS), 1.5, 0.85 * F, mu, sig, alpha])
    for dv in RED_SHIFTS:
        out.append([0.7 * F, pk, 1.5, 0.3 * F, pk * (1 + dv / C_KMS), 1.5, 1.0])
    out.append([1e-3 * F, lam_sys * (1 - 300 / C_KMS), 1.5, F, mu, sig, alpha])
    return out


def peak_quantities(popt, lam_sys_vac, s_lsf):
    """Velocities (from vacuum peak wavelengths), separation, ratio, widths,
    and the red peak's redshift."""
    fb, mub, sb, fr, mur, sr, ar = popt
    lam_r = red_peak(mur, sr, ar, s_lsf)
    lam_r_vac = float(pc.air_to_vac(lam_r))
    vb = C_KMS * (float(pc.air_to_vac(mub)) / lam_sys_vac - 1.0)
    vr = C_KMS * (lam_r_vac / lam_sys_vac - 1.0)
    so, ao = flp.convolved_skew_params(sr, ar, s_lsf)
    g = 2 * np.sqrt(2 * np.log(2))
    return dict(dv_blue=vb, dv_red=vr, peak_sep=vr - vb, flux_blue=fb, flux_red=fr,
                flux_total=fb + fr, blue_red_ratio=fb / fr if fr > 0 else np.nan,
                lam_red=lam_r, z_lya_red=lam_r_vac / pc.LYA_REST - 1.0,
                fwhm_blue_kms=g * sb / mub * C_KMS,
                fwhm_red_kms=skew_peak_fwhm(mur, sr, ar)[1] / mur * C_KMS,
                fwhm_red_obs_kms=skew_peak_fwhm(mur, so, ao)[1] / mur * C_KMS)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def load_nb(sid, subcube_dir):
    path = pc.muse_path(sid, "nb", subcube_dir)
    if not os.path.exists(path):
        return None, None
    with fits.open(path) as h:
        nb = np.array(h[1].data, dtype=float)
        pixscale = float(np.mean(proj_plane_pixel_scales(WCS(h[1].header).celestial)) * 3600)
    return nb, pixscale


def plot_double(path, sid, d, wave, flux, var, lam_sys, single, dbl, res, args):
    wd, fd, vd = flp.rebin_for_display(wave, flux, var, args.plot_bin)
    nd = np.sqrt(np.clip(vd, 0, np.inf))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    ax1.step(wd, fd, where="mid", color="black", lw=1.0, alpha=0.6,
             label=f"Flux (bin {args.plot_bin})" if args.plot_bin > 1 else "Flux")
    ax1.fill_between(wd, -nd, nd, color="gray", alpha=0.3, step="mid", label=r"$\pm 1\sigma$")
    ax1.axhline(0, color="k", ls="--", lw=0.8)
    xs = np.linspace(lam_sys - args.plot_halfwidth, lam_sys + args.plot_halfwidth, 2000)
    if single is not None:
        ax1.plot(xs, flp.skew_model_lsf(xs, *single), color="0.35", lw=1.4, ls="--",
                 label="Skewed Gaussian (single)")
    if dbl is not None:
        fb, mub, sb, fr, mur, sr, ar = dbl["popt"]
        s = dbl["s_lsf"]
        ax1.plot(xs, double_model(xs, fb, mub, sb, fr, mur, sr, ar, s), color="crimson",
                 lw=1.8, label="Double fit")
        ax1.plot(xs, gauss(xs, fb, mub, np.hypot(sb, s)), color="royalblue", lw=1.0, ls=":",
                 label="blue peak (Gaussian)")
        ax1.plot(xs, flp.skew_model_lsf(xs, fr, mur, sr, ar, s), color="firebrick", lw=1.0,
                 ls=":", label="red peak (skewed)")
        ax1.axvline(mub, color="royalblue", ls="--", lw=1.0)
        ax1.axvline(res["lam_red"], color="firebrick", ls="--", lw=1.0)
    ax1.axvline(lam_sys, color="orange", ls="--", lw=1.8, label=r"Ly$\alpha$ ($z_{\rm sys}$)")

    txt = []
    if np.isfinite(res.get("lya_snr", np.nan)):
        txt.append(rf"S/N = {res['lya_snr']:.1f}")
    if np.isfinite(res.get("dv_red", np.nan)):
        def pm(k):
            e = res.get(f"{k}_err", np.nan)
            return f"{res[k]:.0f}" + (rf" $\pm$ {e:.0f}" if np.isfinite(e) else "")
        txt += [rf"$\Delta v_{{\rm blue}}$ = {pm('dv_blue')} km/s",
                rf"$\Delta v_{{\rm red}}$ = {pm('dv_red')} km/s",
                rf"separation = {pm('peak_sep')} km/s"]
    if txt:
        ax1.text(0.03, 0.95, "\n".join(txt), transform=ax1.transAxes, va="top", ha="left",
                 fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))
    ax1.set_xlim(lam_sys - args.plot_halfwidth, lam_sys + args.plot_halfwidth)
    inw = (wd >= lam_sys - args.plot_halfwidth) & (wd <= lam_sys + args.plot_halfwidth)
    models = []
    if single is not None:
        models.append(flp.skew_model_lsf(xs, *single))
    if dbl is not None:
        models.append(double_model(xs, *dbl["popt"], dbl["s_lsf"]))
    vals = np.concatenate([fd[inw], nd[inw], -nd[inw]] + models)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = vals.min(), vals.max()
        ax1.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax1.set_xlabel(r"Wavelength (air) [$\AA$]")
    ax1.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    ax1.set_title(f"Source {sid} | double-peak fit | " + rf"$z_{{\rm sys}}={res['z_sys']:.4f}$")
    sec = ax1.secondary_xaxis("top", functions=(lambda l: C_KMS * (l / lam_sys - 1),
                                                lambda v: lam_sys * (1 + v / C_KMS)))
    sec.set_xlabel(r"Velocity from systemic [km s$^{-1}$]", fontsize=9)

    nb, pixscale = load_nb(sid, args.subcube_dir)
    box = 15
    h = box // 2
    cut = np.full((box, box), np.nan)
    if nb is not None:
        xc, yc = int(d["x_pix"]), int(d["y_pix"])
        ys = slice(max(yc - h, 0), min(yc + h + 1, nb.shape[0]))
        xs_ = slice(max(xc - h, 0), min(xc + h + 1, nb.shape[1]))
        cut[ys.start - (yc - h):ys.stop - (yc - h),
            xs_.start - (xc - h):xs_.stop - (xc - h)] = nb[ys, xs_]
    ax2.imshow(cut, origin="lower", cmap="inferno")
    rad = float(d["aperture"]) / pixscale if pixscale else 3.0
    ax2.add_patch(plt.Circle((h, h), rad, edgecolor="cyan", facecolor="none", lw=2.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Pseudo-NB")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per source
# ---------------------------------------------------------------------------

def run_source(sid, z_sys, z_sys_err, frow, args, rng):
    res = dict(ID=sid, z_sys=z_sys, lya_snr=float(frow.get("lya_snr", np.nan)),
               double_peak=0, double_reason="", n_mc_success=0)
    path = pc.muse_path(sid, "spectrum", args.subcube_dir)
    if not os.path.exists(path):
        res["double_reason"] = "no_spectrum"
        return res, None
    d = np.load(path, allow_pickle=True)
    wave, flux, var = (np.asarray(d[k], float) for k in ("wave", "flux", "var"))
    lam_sys = float(pc.lya_obs_air(z_sys))
    lam_sys_vac = pc.LYA_REST * (1 + z_sys)
    centre = float(frow.get("best_center", np.nan))
    if not np.isfinite(centre):
        centre = lam_sys
    sel = ((wave >= centre - args.fit_window) & (wave <= centre + args.fit_window)
           & np.isfinite(flux) & np.isfinite(var) & (var > 0))
    w, f, e = wave[sel], flux[sel], np.sqrt(var[sel])
    n = int(sel.sum())
    res["n_pix"] = n
    s_lsf = float(flp.muse_lsf_sigma(lam_sys))

    # Single fit's chi2 on the same pixels, from its saved parameters
    single = None
    ok_single = str(frow.get("fit_success", "")).lower() in ("true", "1") and \
        all(np.isfinite(float(frow.get(k, np.nan)))
            for k in ("flux_fit", "mu", "sigma", "alpha_skew", "sigma_lsf_ang"))
    if ok_single:
        single = tuple(float(frow[k]) for k in ("flux_fit", "mu", "sigma", "alpha_skew",
                                                "sigma_lsf_ang"))
        chi2_s = float(np.sum(((f - skew_lsf(w, *single)) / e) ** 2))
        res["chi2_single"] = chi2_s
        res["bic_single"] = chi2_s + K_SINGLE * np.log(n)
        s_lsf = single[4]

    bnds = bounds_for(lam_sys, centre, flp.SIGMA_MIN_DEFAULT, flp.ALPHA_MAX_DEFAULT)
    best = fit_double(w, f, e, s_lsf, bnds,
                      starts_from(lam_sys, centre, w, f, single, flp.ALPHA_MAX_DEFAULT))
    if best is None:
        res["double_reason"] = "fit_failed"
        return res, dict(d=d, wave=wave, flux=flux, var=var, lam_sys=lam_sys,
                         single=single, dbl=None)
    popt, pcov, chi2_d = best
    res["chi2_double"] = chi2_d
    res["bic_double"] = chi2_d + K_DOUBLE * np.log(n)
    res["dbic"] = res.get("bic_single", np.nan) - res["bic_double"]
    res.update(peak_quantities(popt, lam_sys_vac, s_lsf))
    res.update(mu_blue=popt[1], sigma_blue=popt[2], mu_red=popt[4], sigma_red=popt[5],
               alpha_red=popt[6], sigma_lsf_ang=s_lsf)
    fwhm_lsf_ang = 2 * np.sqrt(2 * np.log(2)) * s_lsf
    res["fwhm_lsf_kms"] = fwhm_lsf_ang / popt[4] * C_KMS
    res["red_resolved"] = bool(popt[5] > flp.SIGMA_MIN_DEFAULT * (1 + 1e-3) and
                               res["fwhm_red_kms"] >= flp.RESOLVE_FRAC_DEFAULT * res["fwhm_lsf_kms"])

    # MC errors
    keys = ["dv_blue", "dv_red", "peak_sep", "flux_blue", "flux_red", "flux_total",
            "blue_red_ratio", "lam_red", "z_lya_red", "fwhm_blue_kms", "fwhm_red_kms",
            "fwhm_red_obs_kms"]
    if args.n_mc > 0:
        base = double_model(w, *popt, s_lsf)
        draws = {k: [] for k in keys}
        for _ in range(args.n_mc):
            b = fit_double(w, base + rng.normal(0, e), e, s_lsf, bnds, [popt])
            if b is None:
                continue
            q = peak_quantities(b[0], lam_sys_vac, s_lsf)
            for k in keys:
                draws[k].append(q[k])
        res["n_mc_success"] = len(draws["dv_red"])
        if res["n_mc_success"] >= 10:
            for k in keys:
                v = np.asarray(draws[k], float)
                res[f"{k}_err"] = float(np.nanstd(v[np.isfinite(v)], ddof=1))
    else:
        perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        res["flux_blue_err"], res["flux_red_err"] = perr[0], perr[3]
        res["dv_blue_err"] = perr[1] / popt[1] * C_KMS
        res["dv_red_err"] = perr[4] / popt[4] * C_KMS    # location error, approximate
        res["lam_red_err"] = perr[4]
        res["flux_total_err"] = float(np.sqrt(max(pcov[0, 0] + pcov[3, 3] + 2 * pcov[0, 3], 0)))
    # the red peak's centre precision alone, before the z_sys error goes in
    res["dv_red_err_lya"] = res.get("dv_red_err", np.nan)
    if np.isfinite(z_sys_err):
        sys_kms = C_KMS * z_sys_err / (1 + z_sys)
        for k in ("dv_blue", "dv_red"):
            if np.isfinite(res.get(f"{k}_err", np.nan)):
                res[f"{k}_err"] = float(np.hypot(res[f"{k}_err"], sys_kms))

    # Decision
    snr_b = res["flux_blue"] / res.get("flux_blue_err", np.nan)
    snr_r = res["flux_red"] / res.get("flux_red_err", np.nan)
    if single is None:
        res["double_reason"] = "no_single_fit"
    elif res["peak_sep"] < args.min_sep_kms:
        res["double_reason"] = "peaks_merged"
    elif not res["dv_blue"] < 0:
        res["double_reason"] = "blue_not_blueward"
    elif not res["dbic"] > args.dbic_min:
        res["double_reason"] = "dbic"
    elif not (snr_b >= args.peak_snr_min and snr_r >= args.peak_snr_min):
        res["double_reason"] = "weak_peak"
    else:
        res["double_peak"] = 1
    return res, dict(d=d, wave=wave, flux=flux, var=var, lam_sys=lam_sys, single=single,
                     dbl=dict(popt=popt, s_lsf=s_lsf))


PAR_COLS = ("flux_blue", "mu_blue", "sigma_blue", "flux_red", "mu_red", "sigma_red", "alpha_red")


def plots_only(a):
    """Redraw <ID>_lya_double_fit.png from the parameters saved in --out-csv."""
    out = os.path.abspath(a.out_csv)
    if not os.path.exists(out):
        raise SystemExit(f"No {out}. Run the fit first")
    dbl_df = pd.read_csv(out)
    miss = [c for c in PAR_COLS + ("sigma_lsf_ang",) if c not in dbl_df.columns]
    if miss:
        raise SystemExit(f"{out} has no {miss}. It was made by an older version, rerun the fit")
    dbl_df["ID"] = dbl_df["ID"].astype(int)
    if a.ids:
        dbl_df = dbl_df[dbl_df["ID"].isin(a.ids)]
    zs = pc.load_sample(a.zsys, ids=a.ids, need=("ID", "z_sys")).set_index("ID")
    fits_df = pd.read_csv(a.fits_csv)
    fits_df["ID"] = fits_df["ID"].astype(int)
    fits_df = fits_df.set_index("ID")
    n = 0
    for _, r in dbl_df.iterrows():
        sid = int(r["ID"])
        path = pc.muse_path(sid, "spectrum", a.subcube_dir)
        if sid not in zs.index or not os.path.exists(path):
            print(f"  {sid:>6d}  no z_sys or spectrum, skipped")
            continue
        d = np.load(path, allow_pickle=True)
        wave, flux, var = (np.asarray(d[k], float) for k in ("wave", "flux", "var"))
        z_sys = float(zs.loc[sid, "z_sys"])
        single = None
        if sid in fits_df.index:
            frow = fits_df.loc[sid]
            keys = ("flux_fit", "mu", "sigma", "alpha_skew", "sigma_lsf_ang")
            if str(frow.get("fit_success", "")).lower() in ("true", "1") and \
                    all(np.isfinite(float(frow.get(k, np.nan))) for k in keys):
                single = tuple(float(frow[k]) for k in keys)
        par = [float(r[c]) for c in PAR_COLS]
        dbl = (dict(popt=par, s_lsf=float(r["sigma_lsf_ang"]))
               if np.all(np.isfinite(par + [float(r["sigma_lsf_ang"])])) else None)
        res = r.to_dict()
        res["z_sys"] = z_sys
        png = f"{os.path.abspath(a.subcube_dir)}/{sid}/{sid}_{FIG_NAME}"
        plot_double(png, sid, d, wave, flux, var, float(pc.lya_obs_air(z_sys)), single, dbl,
                    res, a)
        n += 1
    print(f"\n  redrew {n} figures, +/-{a.plot_halfwidth:g} AA, bin {a.plot_bin}")
    print(f"Wrote  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME}")


def main():
    a = parse_args()
    print("fit_lya_double_peak.py")
    if a.plots_only:
        for p_ in (a.out_csv, a.zsys, a.fits_csv):
            print(f"  reading  {os.path.abspath(p_)}")
        print(f"  writing  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME}  (figures only)")
        plots_only(a)
        return
    for p_ in (a.zsys, a.fits_csv):
        print(f"  reading  {os.path.abspath(p_)}")
    print(f"  reading  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_lya_spectrum.npz, <ID>_lya_nb.fits")
    print(f"  writing  {os.path.abspath(a.out_csv)}")
    if not a.no_plots:
        print(f"  writing  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME}")
    print(f"  double preferred if dBIC > {a.dbic_min:g}, both peaks >= {a.peak_snr_min:g} sigma, "
          f"peaks >= {a.min_sep_kms:g} km/s apart and the blue peak blueward of systemic, "
          f"{a.n_mc} MC refits\n")

    sample = pc.load_sample(a.zsys, ids=a.ids, need=("ID", "z_sys"))
    fits_df = pd.read_csv(a.fits_csv)
    fits_df["ID"] = fits_df["ID"].astype(int)
    fits_df = fits_df.set_index("ID")
    rng = np.random.default_rng(a.seed)

    rows = []
    for _, s in sample.iterrows():
        sid = int(s["ID"])
        if sid not in fits_df.index:
            print(f"  {sid:>6d}  not in {os.path.basename(a.fits_csv)}, skipped")
            continue
        frow = fits_df.loc[sid]
        zerr = float(s["z_sys_err"]) if "z_sys_err" in s and pd.notna(s["z_sys_err"]) else np.nan
        res, plot = run_source(sid, float(s["z_sys"]), zerr, frow, a, rng)
        rows.append(res)
        msg = f"  {sid:>6d}  S/N {res['lya_snr']:5.1f}"
        if np.isfinite(res.get("dv_red", np.nan)):
            msg += (f"  dBIC {res.get('dbic', np.nan):6.1f}  blue {res['dv_blue']:6.0f}  "
                    f"red {res['dv_red']:6.0f}  sep {res['peak_sep']:5.0f} km/s  "
                    f"B/R {res['blue_red_ratio']:.2f}")
        msg += "  DOUBLE" if res["double_peak"] == 1 else f"  single ({res['double_reason']})"
        print(msg, flush=True)
        if plot is not None and not a.no_plots:
            png = f"{os.path.abspath(a.subcube_dir)}/{sid}/{sid}_{FIG_NAME}"
            plot_double(png, sid, plot["d"], plot["wave"], plot["flux"], plot["var"],
                        plot["lam_sys"], plot["single"], plot["dbl"], res, a)

    cols = ["ID", "z_sys", "lya_snr", "n_pix", "chi2_single", "chi2_double", "bic_single",
            "bic_double", "dbic", "dv_blue", "dv_blue_err", "dv_red", "dv_red_err",
            "dv_red_err_lya", "lam_red", "lam_red_err", "z_lya_red", "z_lya_red_err",
            "peak_sep", "peak_sep_err", "flux_blue", "flux_blue_err", "flux_red",
            "flux_red_err", "flux_total", "flux_total_err", "blue_red_ratio",
            "blue_red_ratio_err", "fwhm_blue_kms", "fwhm_blue_kms_err", "fwhm_red_kms",
            "fwhm_red_kms_err", "fwhm_red_obs_kms", "fwhm_red_obs_kms_err", "fwhm_lsf_kms",
            "red_resolved", "mu_blue", "sigma_blue", "mu_red", "sigma_red", "alpha_red",
            "sigma_lsf_ang",
            "double_peak", "double_reason", "n_mc_success"]
    df = pd.DataFrame(rows).reindex(columns=cols)
    out = os.path.abspath(a.out_csv)
    if a.ids and os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old[~old["ID"].astype(int).isin(df["ID"])], df], ignore_index=True)
    df = df.sort_values("ID").reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    dbl = df[df["double_peak"] == 1]
    print(f"\n  fitted         {int(df['dv_red'].notna().sum())} of {len(df)}")
    print(f"  double preferred {len(dbl)}" + (f"  {dbl['ID'].tolist()}" if len(dbl) else ""))
    for r, n in df.loc[df["double_peak"] != 1, "double_reason"].value_counts().items():
        print(f"    single, {r:<14} {n}")
    print(f"\nWrote  {out}")
    if not a.no_plots:
        print(f"Wrote  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME}")


if __name__ == "__main__":
    main()