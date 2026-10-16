#!/usr/bin/env python3
"""
plot_lya_vs_systemic.py

Per-source figures comparing the MUSE Lya line with the NIRSpec line that set
z_sys, for every source in lya_catalog.csv, and contact-sheet PDFs of the
detections, the non-detections and the close pairs.

Per source, in MUSE_subcubes/<ID>/
-----------------------------------
  <ID>_lya_systemic_panels.png
      Left    MUSE Lya spectrum with the skewed-Gaussian fit and an MC
              1 sigma band, red dashed line at systemic Lya, grey dashed line
              at the Lya peak, Delta_v and Sigma_Lya in the corner.
      Right   NIRSpec spectrum of the z_sys line ([OIII] or Halpha) with its
              local continuum removed and the LiMe model. Where the [OIII]
              doublet was fitted jointly both lines get a systemic line and
              the panel is centred between them.
      Each panel is normalised to the peak of its own model, with a velocity
      axis on top (v = 0 at systemic Lya on the left, at systemic [OIII] 5007
      or Halpha on the right).
  <ID>_lya_systemic_overlay.png
      Both lines on one velocity axis about systemic, each normalised to its
      model peak, with dashed lines at v = 0 and at v = Delta_v.

PDFs, in plots/lya_systemic_pdfs/
---------------------------------
  lya_systemic_<panels|overlay>_<tag>_<detections|non_detections|close_pairs>.pdf

<tag> is the detection rule the catalogue was built with (det_rule, e.g. p95
or p90_dverr100), so PDFs for different rules sit side by side.

The sets come from lya_catalog.csv (build_lya_catalog.py, p2_common.
lya_pdf_groups). Detections and non-detections are the sources with
in_analysis == 1 split on lya_det. The close-pairs PDFs hold every source
with pair_status == close_pair, detected or not, with its group number in
the caption. Interlopers are in no PDF. Sources not tested for Lya (e.g.
65863) still get their figures but are in no PDF, and are listed on the
first page of each. Every PDF is sorted by S/N, highest first. The figures
do not depend on the detection rule, so after a change to it --pdf-only
remakes the PDFs in a few seconds without refitting.

Where the lines come from
-------------------------
Lya. The saved science fit (lya_fits_mc.csv), evaluated with
fit_lya_properties_grating.skew_model_lsf at mu, sigma, alpha_skew, flux_fit
and sigma_lsf_ang, i.e. the LSF-convolved model compared with the data. The
grey band is the same model-mode bootstrap as mc_lya_errors_grating.py with
--n-mc refits (display only). Its peak is checked against lya_peak_wave.
Where the Lya fit failed there is no model or peak line, and the spectrum is
normalised to its brightest pixel (after display binning) in the Lya band,
-300/+1200 km/s, with a floor of 3 sigma so pure noise is not stretched.
Where the catalogue prefers the double fit (lya_model double) the model is
the double-peak model from lya_double_fits.csv instead, with no grey band,
and the Delta_v line marks the red peak.

Systemic line. lime_<line>_fits.csv keeps z, its error and the S/N but not
the Gaussian parameters, so the fit is repeated here with lime_fit_lines.py's
own functions (load_spectrum, band_df, ha_blend_band, ha_blend_cfg,
frame_value) and the band settings, input redshift (z_used) and method
recorded in lime_<line>_fits.csv for the z_sys grating. The order of calls
mirrors fit_line. The repeated fit is checked against the catalogue z_sys and
a warning printed if they differ by more than --refit-tol (1 km/s). The
dashed systemic line always sits at the catalogue z_sys. The model is drawn
as lime_fit_lines.plot_fit draws it (profile_flux, center, sigma per
component on the local linear continuum m_cont, n_cont of the main line).
For Halpha fitted as the Halpha + [NII] blend the model includes [NII].

Wavelength frames
-----------------
MUSE is in air and NIRSpec in vacuum. The MUSE wavelengths and Lya model are
moved to vacuum before plotting, so velocities follow Delta_v,
v = c (z_lambda - z_sys) / (1 + z_sys).

Binning
-------
NIRSpec at native sampling. MUSE rebinned by an integer factor so its pixel
width in km/s roughly matches NIRSpec at the line (--muse-bin overrides).
Display only, the fits use the full-resolution data.

Environment
-----------
The LiMe refit needs env312. --pdf-only needs no LiMe and runs in either.

Inputs
------
  MUSE_catalogs/lya_catalog.csv                   build_lya_catalog.py
  jwst_catalogs/primer_minerva_in_muse_zsys.csv   z_sys_line, z_sys_snr
  MUSE_catalogs/lya_fits_mc.csv                   Lya fit parameters
  MUSE_catalogs/lya_detection_cuts_<tag>.csv      the rule quoted in the PDFs
  jwst_catalogs/lime_<line>_fits.csv              z_used, method, band settings
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz
  jwst_spectra_pm/<G>_<F>/<ID>_<G>_<F>_spectra_lime.fits

Usage
-----
python plot_lya_vs_systemic.py --ids 23070 50803        # test, figures only
python plot_lya_vs_systemic.py                          # all sources
python plot_lya_vs_systemic.py --pdf-only               # PDFs only, after a new catalogue
python plot_lya_vs_systemic.py --pdf-only --catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_p90.csv
"""

import argparse
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import p2_common as pc
import fit_lya_properties_grating as flp
from make_lya_fit_pdfs import load_png, rule_text, cuts_for

C_KMS = pc.C_KMS
LYA_REST = pc.LYA_REST
P2 = pc.P2_ROOT
CAT = pc.MUSE_CAT_DIR

GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}

LINE_TEX = {
    "OIII": r"[O$\,$III]$\,\lambda5007$",
    "Ha": r"H$\alpha$",
}

# spec.fit.continuum call inside lime_fit_lines.fit_line, mirrored because it
# is written inline there rather than as a constant.
CONT_DEGREES = [3, 4]
CONT_THRESH = [3, 2]

A4 = (8.27, 11.69)

# ---------------------------------------------------------------------------
# Style, MNRAS serif as before
# ---------------------------------------------------------------------------
MNRAS_COL_WIDTH_IN = 3.46
MNRAS_PAGE_WIDTH_IN = 7.09

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "ytick.right": True,
})

COL_DATA = "black"
COL_NOISE_FILL = "#b58aa3"      # mauve 1 sigma noise band
COL_NOISE_EDGE = "#8a5a76"
COL_SYS = "#f08080"             # pale red systemic line
COL_PEAK = "0.35"               # grey Lya peak line
COL_MODEL = "0.2"
COL_MODEL_BAND = "0.6"
COL_OV_LYA = "#7aa6e8"          # overlay, pastel blue
COL_OV_SYS = "#79c68f"          # overlay, pastel green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def v_from_wave(wave_vac, lam_sys_vac):
    """Velocity about systemic, same convention as Delta_v."""
    return C_KMS * (np.asarray(wave_vac, dtype=float) / lam_sys_vac - 1.0)


def wave_from_v(v, lam_sys_vac):
    return lam_sys_vac * (1.0 + np.asarray(v, dtype=float) / C_KMS)


def rebin(wave, flux, err, nbin):
    """Integer rebin for display. Mean flux, error of the mean."""
    nbin = int(nbin)
    if nbin <= 1:
        return wave, flux, err
    n = (wave.size // nbin) * nbin
    if n < nbin:
        return wave, flux, err
    w = wave[:n].reshape(-1, nbin)
    f = flux[:n].reshape(-1, nbin)
    e = err[:n].reshape(-1, nbin)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean(w, axis=1), np.nanmean(f, axis=1),
                np.sqrt(np.nansum(e ** 2, axis=1)) / nbin)


def pixel_kms(wave, centre):
    """Median pixel width in km/s within +/-50 AA of centre."""
    sel = np.abs(wave - centre) < 50.0
    if sel.sum() < 3:
        sel = slice(None)
    return C_KMS * np.nanmedian(np.diff(wave[sel])) / centre


def y_limits(flux_list, err_list, pad_frac=0.12):
    vals = []
    for f, e in zip(flux_list, err_list):
        vals.append(np.asarray(f))
        if e is not None:
            vals += [np.asarray(e), -np.asarray(e)]
    vals = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([])
    lo = min(np.nanmin(vals), -0.2) if vals.size else -0.2
    hi = max(np.nanmax(vals), 1.1) if vals.size else 1.1
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def load_lya(npz_path, frow, z_sys, n_mc, rng, fit_window, sigma_min, alpha_max,
             dv_blue, dv_red, drow=None):
    """MUSE spectrum and the science Lya fit, in vacuum wavelength.

    With a successful fit the model is flp.skew_model_lsf at the saved
    parameters and everything is normalised to its peak. Where the catalogue
    prefers the double fit (drow, its row of lya_double_fits.csv) the model is
    the double-peak model instead, with no MC band. Without a fit there is no
    model, and the normalisation is set later from the data
    (normalise_no_fit) once the display binning is known.
    """
    d = np.load(npz_path)
    wave_air = np.asarray(d["wave"], dtype=float)
    flux = np.asarray(d["flux"], dtype=float)
    var = np.asarray(d["var"], dtype=float)
    err = np.sqrt(np.clip(var, 0.0, np.inf))
    out = dict(wave=flp.air_to_vac(wave_air), flux=flux, err=err, grid=None, model=None,
               lo=None, hi=None, n_mc=0, peak_check_kms=np.nan, has_fit=False, norm=1.0)

    if drow is not None:
        from fit_lya_double_peak import double_model
        par = [float(drow[k]) for k in ("flux_blue", "mu_blue", "sigma_blue", "flux_red",
                                         "mu_red", "sigma_red", "alpha_red", "sigma_lsf_ang")]
        grid_air = np.linspace(par[1] - 8 * max(par[2], par[7]) - 5,
                               par[4] + 12 * max(par[5], par[7]) + 5, 1500)
        best = double_model(grid_air, *par)
        peak = float(np.nanmax(best))
        out.update(flux=flux / peak, err=err / peak, grid=flp.air_to_vac(grid_air),
                   model=best / peak, has_fit=True, norm=peak, double=True)
        return out

    ok = frow is not None and str(frow.get("fit_success", False)).lower() in ("true", "1") and np.all(
        np.isfinite([float(frow[k]) for k in ("mu", "sigma", "alpha_skew", "flux_fit",
                                              "sigma_lsf_ang")]))
    if not ok:
        lam = pc.LYA_REST * (1 + z_sys)
        out["band_vac"] = (lam * (1 - dv_blue / C_KMS), lam * (1 + dv_red / C_KMS))
        return out


    mu, sig, alpha, ftot, s_lsf = (float(frow[k]) for k in
                                   ("mu", "sigma", "alpha_skew", "flux_fit", "sigma_lsf_ang"))
    centre = float(frow["best_center"]) if np.isfinite(frow["best_center"]) else mu
    sig_obs, alpha_obs = flp.convolved_skew_params(sig, alpha, s_lsf)
    if str(frow.get("peak_used", "obs")) == "int":
        peak_air = flp.compute_peak_from_model(mu, sig, alpha)
    else:
        peak_air = flp.compute_peak_from_model(mu, sig_obs, alpha_obs)
    out["peak_check_kms"] = C_KMS * (peak_air - float(frow["lya_peak_wave"])) / peak_air

    grid_air = np.linspace(mu - 12 * sig_obs, mu + 12 * sig_obs, 1500)
    best = flp.skew_model_lsf(grid_air, ftot, mu, sig, alpha, s_lsf)
    peak = float(np.nanmax(best))

    if n_mc > 0:
        models = []
        sel = (wave_air >= centre - fit_window) & (wave_air <= centre + fit_window)
        base = flp.skew_model_lsf(wave_air, ftot, mu, sig, alpha, s_lsf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_mc):
                fake = flux.copy()
                fake[sel] = base[sel] + rng.normal(0.0, err[sel])
                fit = flp.fit_skewed_gaussian(wave_air, fake, var, centre, fit_window=fit_window,
                                              sigma_min=sigma_min, alpha_max=alpha_max)
                if fit["fit_success"]:
                    models.append(flp.skew_model_lsf(grid_air, fit["flux_fit"], fit["mu"],
                                                     fit["sigma"], fit["alpha_skew"],
                                                     fit["sigma_lsf_ang"]))
        out["n_mc"] = len(models)
        if len(models) >= 10:
            lo, hi = np.nanpercentile(np.array(models), [16, 84], axis=0)
            out["lo"], out["hi"] = lo / peak, hi / peak

    out.update(flux=flux / peak, err=err / peak, grid=flp.air_to_vac(grid_air),
               model=best / peak, has_fit=True, norm=peak)
    return out


def normalise_no_fit(lya):
    """No Lya fit. Scale the binned data to its brightest pixel in the Lya
    band, but never by less than 3 sigma, so noise is not blown up to 1."""
    lo, hi = lya["band_vac"]
    sel = (lya["wave_d"] >= lo) & (lya["wave_d"] <= hi) & np.isfinite(lya["flux_d"])
    if not sel.any():
        norm = 1.0
    else:
        norm = max(float(np.nanmax(lya["flux_d"][sel])),
                   3.0 * float(np.nanmedian(lya["err_d"][sel])))
    norm = norm if np.isfinite(norm) and norm > 0 else 1.0
    for k in ("flux", "err", "flux_d", "err_d"):
        lya[k] = lya[k] / norm
    lya["norm"] = norm


# ---------------------------------------------------------------------------
# NIRSpec systemic line
# ---------------------------------------------------------------------------
def import_lime():
    try:
        import lime_fit_lines as lfl
    except ImportError as e:
        raise SystemExit(f"Could not import lime_fit_lines ({e}). The systemic refit needs "
                         "LiMe, so run this in env312, or use --pdf-only to remake the PDFs")
    return lfl


def comp_rest(lfl):
    return {lfl.L_O3_5007: lfl.OIII_5007, lfl.L_O3_4959: lfl.OIII_4959,
            lfl.L_HA: lfl.HA, lfl.L_NII_RED: lfl.NII_RED, lfl.L_NII_BLUE: lfl.NII_BLUE}


def refit_systemic(lfl, line, path, lrow):
    """Repeat the lime_fit_lines.py fit of one line in one spectrum.

    The method (joint doublet or single 5007, Halpha single or blend), the
    input redshift and the band settings come from the lime_<line>_fits.csv
    row, so nothing is re-decided. Returns the LiMe Spectrum, the data, the
    main-line label and rest wavelength, the model components and the rest
    wavelengths that get a systemic marker.
    """
    a = SimpleNamespace(line_margin=float(lrow["line_margin"]),
                        cont_gap=float(lrow["cont_gap"]),
                        cont_width=float(lrow["cont_width"]))
    z = float(lrow["z_used"])
    method = str(lrow["method"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec, wave, flux, err = lfl.load_spectrum(path, z)
        spec.fit.continuum(degree_list=CONT_DEGREES, emis_threshold=CONT_THRESH)
        if line == "OIII":
            ref, rest = lfl.L_O3_5007, lfl.OIII_5007
            if method == "joint_4959_5007":
                band = lfl.band_df(lfl.L_O3_5007, lfl.OIII_4959, lfl.OIII_5007, lfl.OIII_5007, a)
                label = lfl.L_O3_BLEND
                cfg = {lfl.L_O3_BLEND: f"{lfl.L_O3_4959}+{lfl.L_O3_5007}",
                       f"{lfl.L_O3_4959}_amp": {"expr": f"{lfl.L_O3_5007}_amp/{lfl.OIII_RATIO_THEORY}"},
                       f"{lfl.L_O3_4959}_kinem": lfl.L_O3_5007}
                comps, sys_rest = [lfl.L_O3_5007, lfl.L_O3_4959], [lfl.OIII_4959, lfl.OIII_5007]
            elif method == "single_5007":
                band = lfl.band_df(lfl.L_O3_5007, lfl.OIII_5007, lfl.OIII_5007, lfl.OIII_5007, a)
                label, cfg = lfl.L_O3_5007, None
                comps, sys_rest = [lfl.L_O3_5007], [lfl.OIII_5007]
            else:
                raise ValueError(f"unknown [OIII] method {method}")
            kw = {"bands": band, "cont_source": lfl.CONT_SOURCE}
            if cfg is not None:
                kw["fit_cfg"] = cfg
            spec.fit.bands(label, **kw)
        elif line == "Ha":
            ref, rest = lfl.L_HA, lfl.HA
            band = lfl.band_df(lfl.L_HA, lfl.HA, lfl.HA, lfl.HA, a)
            spec.fit.bands(lfl.L_HA, bands=band, cont_source=lfl.CONT_SOURCE)
            comps, sys_rest = [lfl.L_HA], [lfl.HA]
            if method == "blend_NII":
                spec.fit.bands(lfl.L_HA_BLEND, bands=lfl.ha_blend_band(a),
                               fit_cfg=lfl.ha_blend_cfg(), cont_source=lfl.CONT_SOURCE)
                comps = [lfl.L_HA, lfl.L_NII_BLUE, lfl.L_NII_RED]
        else:
            raise ValueError(f"z_sys line {line} not supported")
    return spec, wave, flux, err, ref, rest, comps, sys_rest


def load_systemic(lfl, line, path, lrow, z_sys, n_mc, rng):
    """NIRSpec data minus LiMe's local continuum, and the LiMe model, both
    normalised to the peak of the main-line Gaussian."""
    spec, wave, flux, err, ref, rest, comps, sys_rest = refit_systemic(lfl, line, path, lrow)
    fv = lfl.frame_value
    m, n = fv(spec, ref, "m_cont"), fv(spec, ref, "n_cont")
    if np.isfinite(m) and np.isfinite(n):
        cont = m * wave + n
    else:
        c0 = fv(spec, ref, "cont")
        cont = np.full_like(wave, c0 if np.isfinite(c0) else 0.0)

    pars = [(fv(spec, c, "profile_flux"), fv(spec, c, "center"), fv(spec, c, "sigma"))
            for c in comps]
    pars = [p for p in pars if np.all(np.isfinite(p)) and p[2] > 0]
    f0, c0_, s0 = (fv(spec, ref, k) for k in ("profile_flux", "center", "sigma"))
    if not (np.isfinite(f0) and np.isfinite(c0_) and np.isfinite(s0) and s0 > 0 and f0 > 0):
        raise ValueError("LiMe fit of the main line has no usable flux, centre or sigma")
    amp = f0 / (np.sqrt(2 * np.pi) * s0)
    z_refit = c0_ / rest - 1.0

    rmap = comp_rest(lfl)
    lam_all = sorted(rmap[c] * (1 + z_sys) for c in comps)
    grid = np.linspace(lam_all[0] * (1 - 4000 / C_KMS), lam_all[-1] * (1 + 4000 / C_KMS), 4000)
    best = sum(lfl.gaussian(grid, f, c, s) for f, c, s in pars)

    lo = hi = None
    if n_mc > 0:
        # Display band from the errors of the main line, the other components
        # scaled with it as the fit ties imply.
        fe, ce, se = (fv(spec, ref, k) for k in ("profile_flux_err", "center_err", "sigma_err"))
        fe, ce, se = (x if np.isfinite(x) else 0.0 for x in (fe, ce, se))
        draws = []
        for _ in range(n_mc):
            f_, c_, s_ = rng.normal(f0, fe), rng.normal(c0_, ce), abs(rng.normal(s0, se))
            draws.append(sum(lfl.gaussian(grid, f_ * f / f0, c_ * c / c0_, s_ * s / s0)
                             for f, c, s in pars))
        lo, hi = np.nanpercentile(np.array(draws), [16, 84], axis=0)
        lo, hi = lo / amp, hi / amp

    return dict(wave=wave, flux=(flux - cont) / amp, err=err / amp, grid=grid,
                model=best / amp, lo=lo, hi=hi, z_refit=z_refit, rest=rest,
                lam_sys_lines=sorted(r * (1 + z_sys) for r in sys_rest),
                blend=line == "Ha" and len(comps) > 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def add_top_velocity(ax, lam_sys):
    top = ax.secondary_xaxis("top", functions=(
        lambda w: v_from_wave(w, lam_sys), lambda v: wave_from_v(v, lam_sys)))
    top.set_xlabel(r"Velocity [km s$^{-1}$]")
    top.tick_params(direction="in")
    return top


def draw_panel(ax, s, lam_sys, vrange, text_lines, peak_wave=None,
               label_systemic=True, extra_sys=(), text_x=0.04, text_ha="left"):
    """One spectrum panel in wavelength with a velocity axis on top."""
    w_lo, w_hi = wave_from_v(vrange[0], lam_sys), wave_from_v(vrange[1], lam_sys)
    inw = (s["wave_d"] >= w_lo) & (s["wave_d"] <= w_hi)

    ax.fill_between(s["wave_d"], -s["err_d"], s["err_d"], step="mid",
                    color=COL_NOISE_FILL, alpha=0.45, lw=0.6,
                    edgecolor=COL_NOISE_EDGE, zorder=1)
    ax.axhline(0.0, color="k", ls="--", lw=0.6, zorder=2)
    ax.step(s["wave_d"], s["flux_d"], where="mid", color=COL_DATA, lw=0.9, zorder=3)
    if s["lo"] is not None:
        ax.fill_between(s["grid"], s["lo"], s["hi"], color=COL_MODEL_BAND,
                        alpha=0.5, lw=0, zorder=4)
    if s["model"] is not None:
        ax.plot(s["grid"], s["model"], color=COL_MODEL, ls="--", lw=1.1, zorder=5)

    for lam in (lam_sys, *extra_sys):
        ax.axvline(lam, color=COL_SYS, ls="--", lw=1.2, zorder=6)
    if peak_wave is not None and np.isfinite(peak_wave):
        ax.axvline(peak_wave, color=COL_PEAK, ls="--", lw=1.2, zorder=6)

    ax.set_xlim(w_lo, w_hi)
    ylo, yhi = y_limits([s["flux_d"][inw]], [s["err_d"][inw]])
    ax.set_ylim(ylo, yhi)
    if label_systemic:
        ax.text(lam_sys - 0.012 * (w_hi - w_lo), ylo + 0.62 * (yhi - ylo), "systemic",
                rotation=90, ha="right", va="center", color="0.45", fontsize=8, zorder=7,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.text(text_x, 0.95, "\n".join(text_lines), transform=ax.transAxes, va="top",
            ha=text_ha, multialignment="left", fontsize=8.5, linespacing=1.5, zorder=8)
    ax.set_xlabel(r"Observed wavelength (vacuum) [$\rm \AA$]")
    add_top_velocity(ax, lam_sys)


def dv_text(src):
    if not np.isfinite(src["delta_v"]):
        return r"no Ly$\alpha$ fit"
    lab = r"\Delta v_{\rm red}" if src.get("double") else r"\Delta v"
    if np.isfinite(src["delta_v_err"]):
        return (rf"${lab} = {src['delta_v']:.0f} \pm {src['delta_v_err']:.0f}$"
                r" km s$^{-1}$")
    return rf"${lab} = {src['delta_v']:.0f}$ km s$^{{-1}}$"


def figure_panels(src, lya, sysl, out_path, vr_lya, vr_sys, dpi, doublet_pad):
    fig, axes = plt.subplots(1, 2, figsize=(MNRAS_PAGE_WIDTH_IN, 2.9))
    draw_panel(axes[0], lya, src["lam_lya_sys"], vr_lya,
               [f"ID {src['ID']}", dv_text(src), rf"S/N = {src['lya_snr']:.1f}"],
               peak_wave=src["lam_lya_peak"])
    axes[0].set_ylabel("Normalised flux")
    axes[0].text(0.96, 0.95, r"MUSE Ly$\alpha$", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8.5)

    lams = sysl["lam_sys_lines"]
    extra = [l for l in lams if abs(l - src["lam_line_sys"]) > 1e-6 * src["lam_line_sys"]]
    info = [f"ID {src['ID']}", rf"$z_{{\rm sys}} = {src['z_sys']:.4f}$",
            rf"S/N = {src['z_sys_snr']:.1f}"]
    if extra:
        # [OIII] doublet. The clear space is between the lines, so the text
        # goes there, centred, and the label drops the 5007.
        v_l = v_from_wave(np.array(lams), src["lam_line_sys"])
        vr = [float(v_l.min()) - doublet_pad, float(v_l.max()) + doublet_pad]
        x_mid = (0.5 * (v_l.min() + v_l.max()) - vr[0]) / (vr[1] - vr[0])
        tex_pair = src["tex"].split(r"$\,\lambda")[0]
        draw_panel(axes[1], sysl, src["lam_line_sys"], vr,
                   info + [f"NIRSpec {tex_pair}", src["grating"]],
                   label_systemic=False, extra_sys=extra, text_x=x_mid, text_ha="center")
    else:
        draw_panel(axes[1], sysl, src["lam_line_sys"], vr_sys, info, label_systemic=False)
        tex = src["tex"] + (r" + [N$\,$II]" if sysl["blend"] else "")
        axes[1].text(0.96, 0.95, f"NIRSpec {tex}\n{src['grating']}",
                     transform=axes[1].transAxes, ha="right", va="top", fontsize=8.5,
                     linespacing=1.4)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_overlay(src, lya, sysl, out_path, vrange, show_fits, dpi):
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.8))
    v_lya = v_from_wave(lya["wave_d"], src["lam_lya_sys"])
    v_sys = v_from_wave(sysl["wave_d"], src["lam_line_sys"])

    ax.axhline(0.0, color="0.5", lw=0.5, zorder=1)
    ax.step(v_lya, lya["flux_d"], where="mid", color=COL_OV_LYA, lw=1.4,
            label=r"Ly$\alpha$ (MUSE)", zorder=3)
    ax.step(v_sys, sysl["flux_d"], where="mid", color=COL_OV_SYS, lw=1.4,
            label=f"{src['tex']} (NIRSpec)", zorder=2)
    if show_fits:
        if lya["model"] is not None:
            ax.plot(v_from_wave(lya["grid"], src["lam_lya_sys"]), lya["model"],
                    color=COL_OV_LYA, lw=0.9, alpha=0.6, zorder=4)
        ax.plot(v_from_wave(sysl["grid"], src["lam_line_sys"]), sysl["model"],
                color=COL_OV_SYS, lw=0.9, alpha=0.6, zorder=4)

    ax.set_xlim(*vrange)
    m_lya = (v_lya >= vrange[0]) & (v_lya <= vrange[1])
    m_sys = (v_sys >= vrange[0]) & (v_sys <= vrange[1])
    ylo, yhi = y_limits([lya["flux_d"][m_lya], sysl["flux_d"][m_sys]], [None, None])
    yhi = yhi + 0.15 * (yhi - ylo)
    ax.set_ylim(ylo, yhi)

    dx = 0.012 * (vrange[1] - vrange[0])
    ytxt = yhi - 0.04 * (yhi - ylo)
    ax.axvline(0.0, color=COL_OV_SYS, ls="--", lw=1.1, zorder=5)
    ax.text(-dx, ytxt, "systemic", color=COL_OV_SYS, fontsize=7.5, ha="right", va="top")
    if np.isfinite(src["delta_v"]):
        ax.axvline(src["delta_v"], color=COL_OV_LYA, ls="--", lw=1.1, zorder=5)
        ax.text(src["delta_v"] + dx, ytxt, r"Ly$\alpha$ peak", color=COL_OV_LYA,
                fontsize=7.5, ha="left", va="top")

    ax.set_title(f"ID {src['ID']}   " + rf"$z_{{\rm sys}} = {src['z_sys']:.4f}$   "
                 + dv_text(src), fontsize=8, pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1.5, columnspacing=1.2)
    ax.set_xlabel(r"Velocity relative to systemic [km s$^{-1}$]")
    ax.set_ylabel("Normalised flux")
    ax.tick_params(top=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# One source
# ---------------------------------------------------------------------------
def make_source(sid, row, frow, lrow, lfl, a):
    """Both figures for one source. Returns (status, message), status is
    "ok", "warn" (LiMe refit off the catalogue z_sys) or "fail"."""
    line, g = str(row["z_sys_line"]), str(row["z_sys_grating"])
    if line not in LINE_TEX:
        return "fail", f"z_sys line {line} not supported"
    if lrow is None:
        return "fail", f"no fitted {g} row in lime_{line}_fits.csv"
    gfull = f"{g}_{GRATING_FILTER[g]}"
    lime_path = os.path.join(a.spectra_root, gfull, f"{sid}_{gfull}_spectra_lime.fits")
    npz_path = pc.muse_path(sid, "spectrum", a.subcube_dir)
    for p in (lime_path, npz_path):
        if not os.path.exists(p):
            return "fail", f"missing {p}"

    z_sys = float(row["z_sys"])
    rng = np.random.default_rng(a.seed + sid)
    drow = None
    if row.get("lya_model") == "double" and a.double_fits is not None and sid in a.double_fits.index:
        drow = a.double_fits.loc[sid]
    lya = load_lya(npz_path, frow, z_sys, a.n_mc, rng, a.fit_window, a.sigma_min,
                   a.alpha_max, a.dv_blue, a.dv_red, drow=drow)
    try:
        sysl = load_systemic(lfl, line, lime_path, lrow, z_sys, a.n_mc, rng)
    except Exception as e:
        return "fail", f"LiMe refit failed: {e}"

    dv_refit = C_KMS * (sysl["z_refit"] - z_sys) / (1.0 + z_sys)
    warn = abs(dv_refit) > a.refit_tol
    lam_lya_sys = LYA_REST * (1.0 + z_sys)
    lam_line_sys = sysl["rest"] * (1.0 + z_sys)
    delta_v = float(row["delta_v"]) if lya["has_fit"] else np.nan

    v_pix_nir = pixel_kms(sysl["wave"], lam_line_sys)
    v_pix_muse = pixel_kms(lya["wave"], lam_lya_sys)
    mbin = a.muse_bin if a.muse_bin is not None else max(1, int(round(v_pix_nir / v_pix_muse)))
    lya["wave_d"], lya["flux_d"], lya["err_d"] = rebin(lya["wave"], lya["flux"], lya["err"], mbin)
    sysl["wave_d"], sysl["flux_d"], sysl["err_d"] = sysl["wave"], sysl["flux"], sysl["err"]
    if not lya["has_fit"]:
        normalise_no_fit(lya)

    src = dict(ID=sid, z_sys=z_sys, tex=LINE_TEX[line], grating=gfull,
               lya_snr=float(row["lya_snr"]), z_sys_snr=float(row["z_sys_snr"]),
               delta_v=delta_v, delta_v_err=float(row["delta_v_err"]),
               lam_lya_sys=lam_lya_sys, lam_line_sys=lam_line_sys,
               lam_lya_peak=wave_from_v(delta_v, lam_lya_sys) if np.isfinite(delta_v) else None,
               double=bool(lya.get("double", False)))

    p1 = pc.muse_path(sid, "systemic_panels_png", a.subcube_dir)
    p2 = pc.muse_path(sid, "systemic_overlay_png", a.subcube_dir)
    figure_panels(src, lya, sysl, p1, a.lya_vrange, a.sys_vrange, a.dpi, a.doublet_pad)
    figure_overlay(src, lya, sysl, p2, a.overlay_vrange, not a.no_overlay_fits, a.dpi)

    msg = (f"{line} {gfull}, LiMe refit {dv_refit:+.2f} km/s from z_sys"
           + (f" WARNING above {a.refit_tol:g}" if warn else "")
           + (f", Lya peak check {lya['peak_check_kms']:+.2f} km/s, MC {lya['n_mc']}/{a.n_mc}"
              if lya["has_fit"] else ", no Lya fit")
           + f", MUSE bin x{mbin}\n         {p1}\n         {p2}")
    return ("warn" if warn else "ok"), msg


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
TITLES = {"detections": "detections", "non_detections": "non-detections",
          "close_pairs": "close pairs (not in the analysis)"}


def caption(r, with_group=False):
    parts = [f"ID {int(r['ID'])}"]
    if with_group:
        parts.append(f"group {int(r['close_pair_group'])}")
    parts += [rf"$\Sigma$ = {r['lya_snr']:.2f}", f"z_sys = {r['z_sys']:.4f}"]
    if pd.notna(r["delta_v"]):
        e = (f" $\\pm$ {r['delta_v_err']:.0f}" if pd.notna(r.get("delta_v_err")) else "")
        if pd.notna(r.get("lya_centre_err")):
            e += f" (centre $\\pm$ {r['lya_centre_err']:.0f})"
        dbl = r.get("lya_model") == "double"
        lab = r"$\Delta v_{\rm red}$" if dbl else r"$\Delta v$"
        parts.append(rf"{lab} = {r['delta_v']:.0f}{e} km/s" + (" (double)" if dbl else ""))
    parts.append(pc.det_label(r))
    return "    ".join(parts)


def summary_page(pdf, title, subtitle, rows, untested, missing_note, with_reason):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold", va="top")
    fig.text(0.08, 0.92, subtitle, fontsize=9, va="top")
    fig.text(0.08, 0.895, f"{len(rows)} sources, sorted by S/N", fontsize=9, va="top")
    lines = []
    for _, r in rows.iterrows():
        extra = f"  {r['det_reason']}" if with_reason and r.get("det_reason") else ""
        lines.append(f"{int(r['ID']):>6d}  {r['lya_snr']:6.2f}{extra}")
    ncol = 3
    per_col = int(np.ceil(len(lines) / ncol)) if lines else 0
    for c in range(ncol):
        chunk = lines[c * per_col:(c + 1) * per_col]
        fig.text(0.08 + c * 0.3, 0.86, "    ID     S/N\n" + "\n".join(chunk),
                 family="monospace", fontsize=7.5, va="top")
    foot = []
    if untested:
        foot.append("Not tested for Lya, in no PDF: " + ", ".join(map(str, untested)))
    if missing_note:
        foot.append(missing_note)
    if foot:
        fig.text(0.08, 0.06, "\n".join(foot), fontsize=8, va="bottom", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, kind, title, subtitle, rows, untested, a, with_group, with_reason):
    """kind 'panels' (4 per page, one column) or 'overlay' (8 per page)."""
    key = "systemic_panels_png" if kind == "panels" else "systemic_overlay_png"
    nrow, ncol = (4, 1) if kind == "panels" else (4, 2)
    per_page = nrow * ncol
    missing = [int(i) for i in rows["ID"]
               if not os.path.exists(pc.muse_path(int(i), key, a.subcube_dir))]
    note = f"No figure (see the run log): {', '.join(map(str, missing))}" if missing else ""
    with PdfPages(path) as pdf:
        summary_page(pdf, title, subtitle, rows, untested, note, with_reason)
        for start in range(0, len(rows), per_page):
            chunk = rows.iloc[start:start + per_page]
            fig, axes = plt.subplots(nrow, ncol, figsize=A4)
            axes = np.atleast_1d(axes).ravel()
            for ax, (_, r) in zip(axes, chunk.iterrows()):
                png = pc.muse_path(int(r["ID"]), key, a.subcube_dir)
                if os.path.exists(png):
                    ax.imshow(load_png(png, a.width_px))
                else:
                    ax.text(0.5, 0.5, f"no figure\n{png}", ha="center", va="center",
                            fontsize=6, transform=ax.transAxes)
                ax.set_title(caption(r, with_group), fontsize=6 if kind == "overlay" else 7.5,
                             loc="left")
                ax.set_axis_off()
            for ax in axes[len(chunk):]:
                ax.set_axis_off()
            fig.tight_layout(h_pad=0.6)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return missing


def make_pdfs(cat, a):
    cat = cat.copy()
    cat["det_reason"] = cat["det_reason"].fillna("")
    groups, untested, interlopers = pc.lya_pdf_groups(cat)
    tag, cuts = cuts_for(cat, a.catalog, a.cuts)
    rule = rule_text(cuts)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"\nPDFs  ({rule_text(cuts, tex=False)})")
    for kind in ("panels", "overlay"):
        for name, rows in groups.items():
            path = os.path.join(os.path.abspath(a.outdir),
                                f"lya_systemic_{kind}_{tag}_{name}.pdf")
            title = f"Lya vs systemic ({kind}), {TITLES[name]} ({tag})"
            missing = write_pdf(path, kind, title, rule, rows, untested, a,
                                with_group=name == "close_pairs",
                                with_reason=name != "detections")
            print(f"  {kind:<8} {name:<15} {len(rows):>3d} sources  {path}")
            if missing:
                print(f"           WARNING no figure for {missing}")
    if interlopers:
        print(f"  interlopers, in no PDF: {interlopers}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.LYA_CATALOG)
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits_mc.csv")
    p.add_argument("--double-csv", default=f"{CAT}/lya_double_fits.csv",
                   help="fit_lya_double_peak.py output. Sources the catalogue marks "
                        "lya_model double are drawn with the double-peak model")
    p.add_argument("--cuts", default=None,
                   help="default lya_detection_cuts_<tag>.csv next to the catalogue")
    p.add_argument("--lime-dir", default=f"{P2}/jwst_catalogs",
                   help="holds lime_<line>_fits.csv")
    p.add_argument("--spectra-root", default=f"{P2}/jwst_spectra_pm")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outdir", default=f"{P2}/plots/lya_systemic_pdfs")
    p.add_argument("--ids", type=int, nargs="+", default=None)
    p.add_argument("--pdf-only", action="store_true",
                   help="only remake the PDFs from the existing figures (no LiMe needed)")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--lya-vrange", type=float, nargs=2, default=[-1500, 1500])
    p.add_argument("--sys-vrange", type=float, nargs=2, default=[-3500, 3500],
                   help="right panel range for Halpha, km/s")
    p.add_argument("--doublet-pad", type=float, default=1500.0,
                   help="km/s beyond 4959 and 5007 for the [OIII] doublet panel")
    p.add_argument("--overlay-vrange", type=float, nargs=2, default=[-1500, 1500])
    p.add_argument("--muse-bin", type=int, default=None)
    p.add_argument("--dv-blue", type=float, default=300.0,
                   help="Lya band for the no-fit normalisation, as the S/N search")
    p.add_argument("--dv-red", type=float, default=1200.0)
    p.add_argument("--n-mc", type=int, default=100,
                   help="refits for the grey model bands, 0 turns them off")
    p.add_argument("--fit-window", type=float, default=25.0)
    p.add_argument("--sigma-min", type=float, default=flp.SIGMA_MIN_DEFAULT)
    p.add_argument("--alpha-max", type=float, default=flp.ALPHA_MAX_DEFAULT)
    p.add_argument("--refit-tol", type=float, default=1.0)
    p.add_argument("--no-overlay-fits", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width-px", type=int, default=2000,
                   help="figures are shrunk to this width in the PDFs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    print("plot_lya_vs_systemic.py")
    for k in ("catalog", "zsys", "fits_csv", "double_csv", "lime_dir", "spectra_root",
              "subcube_dir", "outdir"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
        print(f"  {k:<13} {getattr(a, k)}")
    print(f"  figures       {a.subcube_dir}/<ID>/<ID>_lya_systemic_panels.png, "
          "<ID>_lya_systemic_overlay.png")

    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    zs = pd.read_csv(a.zsys)[["ID", "z_sys_line", "z_sys_snr"]]
    zs["ID"] = zs["ID"].astype(int)
    cat = cat.merge(zs, on="ID", how="left")
    if a.ids:
        miss = sorted(set(a.ids) - set(cat["ID"]))
        if miss:
            print(f"  not in the catalogue: {miss}")
        cat = cat[cat["ID"].isin(a.ids)]
    print(f"  {len(cat)} sources")

    if not a.pdf_only:
        lfl = import_lime()
        fits = pd.read_csv(a.fits_csv)
        fits["ID"] = fits["ID"].astype(int)
        fits = fits.set_index("ID")
        a.double_fits = None
        if os.path.exists(a.double_csv):
            dd = pd.read_csv(a.double_csv)
            dd["ID"] = dd["ID"].astype(int)
            if "alpha_red" in dd.columns:
                a.double_fits = dd.set_index("ID")
            else:
                print(f"  WARNING {a.double_csv} has no fit parameters, rerun "
                      "fit_lya_double_peak.py. Double sources drawn with the single fit")
        n_dbl = int((cat.get("lya_model") == "double").sum()) if "lya_model" in cat else 0
        print(f"  {n_dbl} sources drawn with the double-peak model (lya_model double)")
        lime = {}
        for line in sorted(cat["z_sys_line"].dropna().unique()):
            path = os.path.join(a.lime_dir, f"lime_{line}_fits.csv")
            print(f"  reading       {path}")
            d = pd.read_csv(path)
            d["ID"] = d["ID"].astype(int)
            lime[line] = d[d["status"] == "fitted"].set_index(["ID", "grating"])
        print("")

        done, failed, warned = [], [], []
        for _, row in cat.sort_values("ID").iterrows():
            sid = int(row["ID"])
            frow = fits.loc[sid] if sid in fits.index else None
            key = (sid, str(row["z_sys_grating"]))
            line = str(row["z_sys_line"])
            lrow = lime[line].loc[key] if line in lime and key in lime[line].index else None
            if isinstance(lrow, pd.DataFrame):
                lrow = lrow.iloc[0]
            try:
                status, msg = make_source(sid, row, frow, lrow, lfl, a)
            except Exception as e:
                status, msg = "fail", f"FAILED {e}"
            print(f"  {sid}  {msg}", flush=True)
            if status == "fail":
                failed.append(sid)
            else:
                done.append(sid)
                if status == "warn":
                    warned.append(sid)
        print(f"\n{len(done)} sources plotted (2 figures each)")
        if failed:
            print(f"  no figures   {failed}")
        if warned:
            print(f"  LiMe refit off the catalogue z_sys by more than {a.refit_tol:g} km/s "
                  f"(refitted by hand?)  {warned}")

    if not a.no_pdf:
        if a.ids:
            print("\nPDFs not remade for an --ids run, use --pdf-only once all figures exist")
        else:
            make_pdfs(cat, a)


if __name__ == "__main__":
    main()