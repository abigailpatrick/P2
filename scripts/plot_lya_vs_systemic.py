#!/usr/bin/env python3
"""
Per-source figures comparing the MUSE Lya line with the NIRSpec systemic line.

For each source two figures are written to  <outroot>/<ID>/ :

  <ID>_lya_systemic_panels.png
      Left panel   MUSE Lya spectrum with the skewed-Gaussian fit and its MC
                   1 sigma band, red dashed line at systemic Lya, grey dashed
                   line at the Lya peak, and Delta_v written in the corner.
      Right panel  NIRSpec continuum-subtracted spectrum of the line that set
                   z_sys ([OIII] or Ha, as chosen in
                   delta_v_from_best_zsys_line.csv) with its LiMe fit and 1
                   sigma band, and z_sys written in the corner. For [OIII]
                   where the tied doublet was fitted, both 4959 and 5007 get a
                   red dashed line at their systemic wavelength and the panel
                   is centred on the midpoint of the pair (--doublet-pad sets
                   how far it extends beyond each line).
      Both panels are normalised to the peak of their own fitted model, with a
      velocity axis on top where v = 0 is systemic Lya (left) and systemic
      [OIII] 5007 or Ha (right).

  <ID>_lya_systemic_overlay.png
      Both lines in one panel on a shared velocity axis about systemic, each
      normalised to its fitted peak, with a dashed line at systemic (v = 0)
      and a dashed line at the Lya peak (v = Delta_v).

Inputs (all from the existing P2 pipeline)
------------------------------------------
  delta_v_from_best_zsys_line.csv   z_sys, z_sys_line, z_sys_snr, delta_v_kms
  lya_properties_mc.csv             skewed-Gaussian parameters (mu, sigma,
                                    alpha_skew, flux_fit, best_center), lya_snr
  systemic_redshifts_by_JELS_ID.csv grating of the chosen systemic line
  <line>_results_by_JELS_ID.csv     DJA z of that grating (band placement)
  {ID}_spectrum.npz                 MUSE aperture spectrum (air wavelengths)
  <ID>_<grating>_spectra_lime.fits  NIRSpec spectrum (vacuum wavelengths)

Where the lines come from
-------------------------
Lya. The plotted model is the saved science fit. fit_lya_properties_grating's
own skew_model is evaluated at mu, sigma, alpha_skew and flux_fit from
lya_properties_mc.csv, and its peak is checked against lya_peak_wave.

Systemic line. The LiMe scripts save only z, z_err and S/N, not the
continuum-subtracted spectrum or the Gaussian parameters. So the fit is
repeated here by importing lime_OIII_jointfit.py and lime_lines_jointfit.py
and calling their own functions in the same order as fit_one() and
fit_one_line() (load_lime_spectrum, the band builders, band_in_range, the
fit_cfg builders, get_5007_row / get_line_row). The only thing mirrored rather
than imported is the continuum call, degree_list=[3, 4] and
emis_threshold=[3, 2], which is hardcoded inside those functions. The repeated
fit is checked against the catalogue z_sys and a warning is printed if they
differ by more than --refit-tol (default 1 km/s). A mismatch means the source
was refitted by hand with overrides, and the dashed systemic line always stays
at the catalogue z_sys.

The grey bands are display aids only. For Lya they use the same model-mode
bootstrap as mc_lya_errors_grating.py. For the systemic line they are drawn
from the LiMe parameter errors.

Wavelength frames
-----------------
MUSE is in air, NIRSpec in vacuum. The MUSE wavelengths and Lya model are
converted to vacuum before plotting, so velocities follow the same convention
as Delta_v in the catalogue,  v = c (z_lambda - z_sys) / (1 + z_sys).

Binning
-------
NIRSpec is plotted at native sampling. MUSE is rebinned by an integer factor
chosen so its pixel width in km/s roughly matches the NIRSpec pixel width at
the line (override with --muse-bin). Rebinning is for display only, the fits
use full-resolution data.

Usage
-----
python plot_lya_vs_systemic.py --id 48121
python plot_lya_vs_systemic.py --group a
python plot_lya_vs_systemic.py                 # every source with a z_sys line
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KMS = 299792.458
LYA_REST = 1215.67

P2 = "/ceph/cephfs/apatrick/P2"

GRATING_FULL = {
    "G235M": "G235M_F170LP",
    "G235H": "G235H_F170LP",
    "G395M": "G395M_F290LP",
    "G395H": "G395H_F290LP",
}

# Display names for the systemic lines. The fitting itself (labels, rest
# wavelengths, band geometry, fit_cfg) is imported from the LiMe scripts.
LINE_TEX = {
    "OIII": r"[O$\,$III]$\,\lambda5007$",
    "Ha": r"H$\alpha$",
    "Hbeta": r"H$\beta$",
    "NII": r"[N$\,$II]$\,\lambda6584$",
    "OII": r"[O$\,$II]$\,\lambda3729$",
}

# Continuum call hardcoded inside fit_one() / fit_one_line() of both LiMe
# scripts. Mirrored here because it is not exposed as a constant there.
CONT_DEGREES = [3, 4]
CONT_THRESH = [3, 2]


# ---------------------------------------------------------------------------
# Style. Serif MNRAS sizing as in plot_lya_science.py, colours after the
# reference figures.
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

COL_OV_LYA = "#7aa6e8"          # overlay figure, pastel blue
COL_OV_SYS = "#79c68f"          # overlay figure, pastel green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def import_pipeline(scripts_dir):
    """Import the three science scripts so every fit and model comes from
    the same code that produced the analysis catalogues."""
    sys.path.insert(0, os.path.abspath(scripts_dir))
    import fit_lya_properties_grating as flp
    import lime_OIII_jointfit as oj
    import lime_lines_jointfit as lj
    return flp, oj, lj


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
    with np.errstate(invalid="ignore"):
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
        vals.append(f)
        if e is not None:
            vals.append(e)
            vals.append(-e)
    vals = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in vals])
    lo, hi = min(np.nanmin(vals), -0.2), max(np.nanmax(vals), 1.1)
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# MUSE Lya
# ---------------------------------------------------------------------------

def load_lya(npz_path, prow, flp, n_mc, rng, fit_window, sigma_min, alpha_max):
    """MUSE spectrum and the science Lya fit, in vacuum wavelength.

    The plotted model is the saved science fit, flp.skew_model evaluated at
    mu, sigma, alpha_skew and flux_fit from lya_properties_mc.csv. Nothing is
    refitted for the model itself. The optional grey 1 sigma band uses the
    same model-mode bootstrap as mc_lya_errors_grating.py (best model plus
    noise, refit with flp.fit_skewed_gaussian and identical settings).
    """
    d = np.load(npz_path)
    wave_air = np.asarray(d["wave"], dtype=float)
    flux = np.asarray(d["flux"], dtype=float)
    var = np.asarray(d["var"], dtype=float)
    err = np.sqrt(np.clip(var, 0.0, np.inf))

    mu, sig, alpha, ftot = (float(prow[k]) for k in
                            ("mu", "sigma", "alpha_skew", "flux_fit"))
    centre = float(prow["best_center"]) if np.isfinite(prow["best_center"]) else mu

    # Consistency check against the catalogue peak wavelength.
    peak_air = flp.compute_peak_from_model(mu, sig, alpha)
    peak_diff = C_KMS * (peak_air - float(prow["lya_peak_wave"])) / peak_air

    grid_air = np.linspace(mu - 12 * sig, mu + 12 * sig, 1500)
    best = flp.skew_model(grid_air, ftot, mu, sig, alpha)
    peak = np.nanmax(best)

    lo = hi = None
    n_ok = 0
    if n_mc > 0:
        models = []
        sel = (wave_air >= centre - fit_window) & (wave_air <= centre + fit_window)
        base = flp.skew_model(wave_air, ftot, mu, sig, alpha)
        for _ in range(n_mc):
            fake = flux.copy()
            fake[sel] = base[sel] + rng.normal(0.0, err[sel])
            fit = flp.fit_skewed_gaussian(wave_air, fake, var, centre,
                                          fit_window=fit_window,
                                          sigma_min=sigma_min,
                                          alpha_max=alpha_max)
            if fit["fit_success"]:
                models.append(flp.skew_model(grid_air, fit["flux_fit"], fit["mu"],
                                             fit["sigma"], fit["alpha_skew"]))
        n_ok = len(models)
        if n_ok >= 10:
            lo, hi = np.nanpercentile(np.array(models), [16, 84], axis=0)
            lo, hi = lo / peak, hi / peak

    return dict(
        wave=flp.air_to_vac(wave_air), flux=flux / peak, err=err / peak,
        grid=flp.air_to_vac(grid_air), model=best / peak, lo=lo, hi=hi,
        n_mc=n_ok, peak_check_kms=peak_diff,
    )


# ---------------------------------------------------------------------------
# NIRSpec systemic line
# ---------------------------------------------------------------------------

def run_lime_fit(lime_path, line_key, z_dja, oj, lj):
    """Repeat the science LiMe fit for one line in one grating.

    Same call sequence as fit_one() in lime_OIII_jointfit.py and
    fit_one_line() in lime_lines_jointfit.py, built from their own functions:
    load_lime_spectrum, the band builders, band_in_range, the fit_cfg
    builders, and get_5007_row / get_line_row with centre_to_redshift for z.
    Only the figure-writing calls are left out.

    Returns (spec, wave, ref_label, component_labels, z_line, snr).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if line_key == "OIII":
            spec, wave = oj.load_lime_spectrum(lime_path, z_dja)
            zf = 1.0 + z_dja
            spec.fit.continuum(degree_list=CONT_DEGREES, emis_threshold=CONT_THRESH)
            doublet_band = oj.build_doublet_band()
            if oj.band_in_range(doublet_band.iloc[0], wave, zf):
                spec.fit.bands(oj.OIII_BLEND_LABEL, bands=doublet_band,
                               fit_cfg=oj.build_joint_fit_cfg())
                comps = [oj.OIII_4959_LABEL, oj.OIII_5007_LABEL]
            else:
                spec.fit.bands(oj.OIII_5007_LABEL, bands=oj.single_5007_band())
                comps = [oj.OIII_5007_LABEL]
            centre, _, snr = oj.get_5007_row(spec)
            return (spec, wave, oj.OIII_5007_LABEL, comps,
                    oj.centre_to_redshift(centre), snr)

        line = next((l for l in lj.LINES if l["name"] == line_key), None)
        if line is None:
            raise ValueError(f"{line_key} not in lime_lines_jointfit.LINES")
        spec, wave = lj.load_lime_spectrum(lime_path, z_dja)
        spec.fit.continuum(degree_list=CONT_DEGREES, emis_threshold=CONT_THRESH)
        if line["kind"] == "single":
            band = lj.single_band_df(line["label"], line["rest_vac"])
            spec.fit.bands(line["label"], bands=band)
            ref, rest, comps = line["label"], line["rest_vac"], [line["label"]]
        else:
            band = lj.doublet_band_df(line["label_blend"], line["rest_vac_other"],
                                      line["rest_vac_ref"])
            spec.fit.bands(line["label_blend"], bands=band,
                           fit_cfg=lj.build_doublet_fit_cfg(line))
            ref, rest = line["label_ref"], line["rest_vac_ref"]
            comps = [line["label_other"], line["label_ref"]]
        centre, _, snr = lj.get_line_row(spec, ref)
        return spec, wave, ref, comps, lj.centre_to_redshift(centre, rest), snr


def rest_wavelength(line_key, oj, lj):
    if line_key == "OIII":
        return oj.OIII_5007_VAC
    line = next(l for l in lj.LINES if l["name"] == line_key)
    return line["rest_vac"] if line["kind"] == "single" else line["rest_vac_ref"]


def component_rest(label, oj, lj):
    """Vacuum rest wavelength for a LiMe component label, taken from the
    constants in the two LiMe scripts."""
    if label == oj.OIII_5007_LABEL:
        return oj.OIII_5007_VAC
    if label == oj.OIII_4959_LABEL:
        return oj.OIII_4959_VAC
    for line in lj.LINES:
        if line["kind"] == "single" and line["label"] == label:
            return line["rest_vac"]
        if line["kind"] == "doublet":
            if line["label_ref"] == label:
                return line["rest_vac_ref"]
            if line["label_other"] == label:
                return line["rest_vac_other"]
    raise KeyError(f"No rest wavelength for {label}")


def load_systemic(lime_path, line_key, z_dja, z_sys, n_mc, rng, oj, lj,
                  cont_mode="local"):
    """Continuum-subtracted NIRSpec spectrum and the LiMe line model,
    normalised to the peak of the reference-line Gaussian.

    The model is LiMe's own gaussian_model evaluated at each component's
    fitted amp, center and sigma from spec.frame. The subtracted continuum
    is the local linear continuum (m_cont, n_cont) LiMe fitted under the line,
    so data and model share one baseline. cont_mode="global" subtracts the
    polynomial continuum from spec.fit.continuum instead, which can look
    flatter far from the line.
    """
    from lime.fitting.lines import gaussian_model

    spec, _, ref, comps, z_refit, snr = run_lime_fit(lime_path, line_key,
                                                     z_dja, oj, lj)
    with fits.open(lime_path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave, flux, err = wave[good], flux[good], err[good]

    fr = spec.frame
    r0 = fr.loc[ref]
    if cont_mode == "global":
        cw = np.ma.filled(np.ma.asarray(spec.wave, dtype=float), np.nan)
        cc = np.ma.filled(np.ma.asarray(spec.cont, dtype=float), np.nan) * spec.norm_flux
        ok = np.isfinite(cw) & np.isfinite(cc)
        cont = np.interp(wave, cw[ok], cc[ok])
    else:
        cont = float(r0["m_cont"]) * wave + float(r0["n_cont"])
    amp, mu, sig = float(r0["amp"]), float(r0["center"]), float(r0["sigma"])

    pars = [(float(fr.loc[c, "amp"]), float(fr.loc[c, "center"]),
             float(fr.loc[c, "sigma"])) for c in comps]

    lam_sys = rest_wavelength(line_key, oj, lj) * (1.0 + z_sys)
    # Systemic wavelength of every fitted component, e.g. both [OIII] lines.
    lam_comps = sorted(component_rest(c, oj, lj) * (1.0 + z_sys) for c in comps)
    grid = np.linspace(lam_comps[0] * (1 - 4000 / C_KMS),
                       lam_comps[-1] * (1 + 4000 / C_KMS), 4000)
    best = sum(gaussian_model(grid, a, m, s) for a, m, s in pars)

    lo = hi = None
    if n_mc > 0:
        # Approximate band from the LiMe errors on the reference line, other
        # components scaled with it as their fit ties imply.
        ae = float(r0["amp_err"]) if np.isfinite(r0["amp_err"]) else 0.0
        me = float(r0["center_err"]) if np.isfinite(r0["center_err"]) else 0.0
        se = float(r0["sigma_err"]) if np.isfinite(r0["sigma_err"]) else 0.0
        draws = []
        for _ in range(n_mc):
            a_, m_, s_ = (rng.normal(amp, ae), rng.normal(mu, me),
                          abs(rng.normal(sig, se)))
            draws.append(sum(gaussian_model(grid, a_ * a / amp, m_ * m / mu,
                                            s_ * s / sig) for a, m, s in pars))
        lo, hi = np.nanpercentile(np.array(draws), [16, 84], axis=0)
        lo, hi = lo / amp, hi / amp

    return dict(
        wave=wave, flux=(flux - cont) / amp, err=err / amp,
        grid=grid, model=best / amp, lo=lo, hi=hi,
        z_refit=z_refit, snr=snr, rest=rest_wavelength(line_key, oj, lj),
        lam_comps=lam_comps,
    )


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
    """One spectrum panel in wavelength space with a velocity axis on top.

    lam_sys sets v = 0 on the top axis. extra_sys are further systemic
    wavelengths (e.g. [OIII] 4959) that also get a red dashed line.
    """
    w_lo, w_hi = wave_from_v(vrange[0], lam_sys), wave_from_v(vrange[1], lam_sys)
    inw = (s["wave_d"] >= w_lo) & (s["wave_d"] <= w_hi)

    ax.fill_between(s["wave_d"], -s["err_d"], s["err_d"], step="mid",
                    color=COL_NOISE_FILL, alpha=0.45, lw=0.6,
                    edgecolor=COL_NOISE_EDGE, zorder=1)
    ax.axhline(0.0, color="k", ls="--", lw=0.6, zorder=2)
    ax.step(s["wave_d"], s["flux_d"], where="mid", color=COL_DATA, lw=0.9,
            zorder=3)

    if s["lo"] is not None:
        ax.fill_between(s["grid"], s["lo"], s["hi"], color=COL_MODEL_BAND,
                        alpha=0.5, lw=0, zorder=4)
    ax.plot(s["grid"], s["model"], color=COL_MODEL, ls="--", lw=1.1, zorder=5)

    for lam in (lam_sys, *extra_sys):
        ax.axvline(lam, color=COL_SYS, ls="--", lw=1.2, zorder=6)
    if peak_wave is not None and np.isfinite(peak_wave):
        ax.axvline(peak_wave, color=COL_PEAK, ls="--", lw=1.2, zorder=6)

    ax.set_xlim(w_lo, w_hi)
    ylo, yhi = y_limits([s["flux_d"][inw]], [s["err_d"][inw]])
    ax.set_ylim(ylo, yhi)

    if label_systemic:
        ax.text(lam_sys - 0.012 * (w_hi - w_lo), ylo + 0.62 * (yhi - ylo),
                "systemic", rotation=90, ha="right", va="center",
                color="0.45", fontsize=8, zorder=7,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none",
                          alpha=0.7))

    ax.text(text_x, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            va="top", ha=text_ha, multialignment="left", fontsize=8.5,
            linespacing=1.5, zorder=8)
    ax.set_xlabel(r"Observed wavelength (vacuum) [$\rm \AA$]")
    add_top_velocity(ax, lam_sys)


def systemic_vrange(sysl, lam_ref, vr_single, pad):
    """Velocity window for the systemic panel, about lam_ref (v = 0).

    One component: vr_single as given. Several components (the [OIII]
    doublet): from the bluest line minus pad to the reddest line plus pad, so
    the window is centred on the midpoint and both lines sit equidistant from
    the panel centre.
    """
    lams = sysl["lam_comps"]
    if len(lams) < 2:
        return vr_single
    v = v_from_wave(np.array(lams), lam_ref)
    return [float(v.min()) - pad, float(v.max()) + pad]


def figure_panels(src, lya, sysl, out_path, vr_lya, vr_sys, dpi, doublet_pad):
    fig, axes = plt.subplots(1, 2, figsize=(MNRAS_PAGE_WIDTH_IN, 2.9))

    dv_txt = (rf"$\Delta v = {src['delta_v']:.0f} \pm {src['delta_v_err']:.0f}$"
              r" km s$^{-1}$")
    draw_panel(axes[0], lya, src["lam_lya_sys"], vr_lya,
               [f"ID {src['ID']}", dv_txt, rf"S/N = {src['lya_snr']:.1f}"],
               peak_wave=src["lam_lya_peak"])
    axes[0].set_ylabel("Normalised flux")
    axes[0].text(0.96, 0.95, r"MUSE Ly$\alpha$", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8.5)

    extra = [l for l in sysl["lam_comps"]
             if abs(l - src["lam_line_sys"]) > 1e-6 * src["lam_line_sys"]]
    vr = systemic_vrange(sysl, src["lam_line_sys"], vr_sys, doublet_pad)
    info = [f"ID {src['ID']}", rf"$z_{{\rm sys}} = {src['z_sys']:.4f}$",
            rf"S/N = {src['z_sys_snr']:.1f}"]
    label = f"NIRSpec {src['tex']}\n{src['grating']}"
    if extra:
        # Doublet. The only clear space is between the two lines, so the
        # info and the instrument label go there as one block, centred in the
        # gap. Both [OIII] lines are shown, so the label drops the 5007.
        v_lines = v_from_wave(np.array(sysl["lam_comps"]), src["lam_line_sys"])
        x_mid = (0.5 * (v_lines.min() + v_lines.max()) - vr[0]) / (vr[1] - vr[0])
        tex_pair = src["tex"].split(r"$\,\lambda")[0]
        draw_panel(axes[1], sysl, src["lam_line_sys"], vr,
                   info + [f"NIRSpec {tex_pair}", src["grating"]],
                   label_systemic=False, extra_sys=extra, text_x=x_mid,
                   text_ha="center")
    else:
        draw_panel(axes[1], sysl, src["lam_line_sys"], vr, info,
                   label_systemic=False)
        axes[1].text(0.96, 0.95, label, transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=8.5, linespacing=1.4)

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
        ax.plot(v_from_wave(lya["grid"], src["lam_lya_sys"]), lya["model"],
                color=COL_OV_LYA, lw=0.9, alpha=0.6, zorder=4)
        ax.plot(v_from_wave(sysl["grid"], src["lam_line_sys"]), sysl["model"],
                color=COL_OV_SYS, lw=0.9, alpha=0.6, zorder=4)

    ax.set_xlim(*vrange)
    m_lya = (v_lya >= vrange[0]) & (v_lya <= vrange[1])
    m_sys = (v_sys >= vrange[0]) & (v_sys <= vrange[1])
    ylo, yhi = y_limits([lya["flux_d"][m_lya], sysl["flux_d"][m_sys]],
                        [None, None])
    yhi = yhi + 0.15 * (yhi - ylo)
    ax.set_ylim(ylo, yhi)

    ax.axvline(0.0, color=COL_OV_SYS, ls="--", lw=1.1, zorder=5)
    ax.axvline(src["delta_v"], color=COL_OV_LYA, ls="--", lw=1.1, zorder=5)

    # Labels just inside the top of the panel, systemic to the left of its
    # line and the Lya peak to the right of its line so they never collide.
    dx = 0.012 * (vrange[1] - vrange[0])
    ytxt = yhi - 0.04 * (yhi - ylo)
    ax.text(-dx, ytxt, "systemic", color=COL_OV_SYS, fontsize=7.5,
            ha="right", va="top")
    ax.text(src["delta_v"] + dx, ytxt, r"Ly$\alpha$ peak", color=COL_OV_LYA,
            fontsize=7.5, ha="left", va="top")

    ax.set_title(
        f"ID {src['ID']}   " + rf"$z_{{\rm sys}} = {src['z_sys']:.4f}$   "
        + rf"$\Delta v = {src['delta_v']:.0f} \pm {src['delta_v_err']:.0f}$ km s$^{{-1}}$",
        fontsize=8, pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.2)
    ax.set_xlabel(r"Velocity relative to systemic [km s$^{-1}$]")
    ax.set_ylabel("Normalised flux")
    ax.tick_params(top=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Per-source Lya vs systemic-line figures.")
    p.add_argument("--id", type=int, default=None, help="Run one source only.")
    p.add_argument("--group", default=None,
                   help="Restrict to IDs in lya_group_<group>.csv (e.g. a).")
    p.add_argument("--group-dir", default=f"{P2}/MUSE_catalogs")
    p.add_argument("--delta-v-csv",
                   default=f"{P2}/MUSE_catalogs/delta_v_from_best_zsys_line.csv")
    p.add_argument("--properties-csv",
                   default=f"{P2}/MUSE_catalogs/lya_properties_mc.csv")
    p.add_argument("--systemic-csv",
                   default=f"{P2}/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv")
    p.add_argument("--catalog-dir", default=f"{P2}/jwst_catalogs",
                   help="Holds <line>_results_by_JELS_ID.csv.")
    p.add_argument("--muse-spec-dir", default=f"{P2}/MUSE_subcubes/dataproducts",
                   help="Holds {ID}_spectrum.npz.")
    p.add_argument("--spectra-root", default=f"{P2}/jwst_spectra",
                   help="Holds <grating>/<ID>_<grating>_spectra_lime.fits.")
    p.add_argument("--outroot", default=f"{P2}/plots",
                   help="Figures go to <outroot>/<ID>/.")
    p.add_argument("--scripts-dir", default=here,
                   help="Directory holding fit_lya_properties_grating.py, "
                        "lime_OIII_jointfit.py and lime_lines_jointfit.py.")
    p.add_argument("--lya-vrange", type=float, nargs=2, default=[-1500, 1500],
                   help="Left panel velocity range about systemic, km/s.")
    p.add_argument("--sys-vrange", type=float, nargs=2, default=[-3500, 3500],
                   help="Right panel velocity range for single lines (Ha), km/s.")
    p.add_argument("--doublet-pad", type=float, default=1500.0,
                   help="For the [OIII] doublet, km/s shown beyond 4959 on the "
                        "blue side and beyond 5007 on the red side. The panel "
                        "is centred on the midpoint of the two lines.")
    p.add_argument("--overlay-vrange", type=float, nargs=2, default=[-1500, 1500])
    p.add_argument("--muse-bin", type=int, default=None,
                   help="MUSE display rebin factor. Default matches NIRSpec pixel km/s.")
    p.add_argument("--nirspec-bin", type=int, default=1)
    p.add_argument("--sys-cont", choices=["local", "global"], default="local",
                   help="NIRSpec continuum to subtract. local is the linear "
                        "continuum LiMe fitted under the line (matches the "
                        "model baseline), global is the polynomial from "
                        "spec.fit.continuum.")
    p.add_argument("--n-mc", type=int, default=100,
                   help="Realisations for the grey 1 sigma model bands. "
                        "0 turns the bands off and speeds the run up.")
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="Lya fit half-window, AA. Must match the science run.")
    p.add_argument("--sigma-min", type=float, default=1.0)
    p.add_argument("--alpha-max", type=float, default=15.0)
    p.add_argument("--refit-tol", type=float, default=1.0,
                   help="Warn if the repeated LiMe fit differs from catalogue "
                        "z_sys by more than this, km/s. It should agree "
                        "exactly unless the source was refitted by hand.")
    p.add_argument("--no-overlay-fits", action="store_true",
                   help="Leave the model curves off the overlay figure.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    flp, oj, lj = import_pipeline(args.scripts_dir)
    # Point the LiMe modules at the chosen catalogue directory.
    oj.CATALOG_DIR = lj.CATALOG_DIR = os.path.abspath(args.catalog_dir)

    paths = {k: os.path.abspath(getattr(args, k)) for k in
             ("delta_v_csv", "properties_csv", "systemic_csv", "catalog_dir",
              "muse_spec_dir", "spectra_root", "outroot")}
    print("[CONFIG]")
    for k, v in paths.items():
        print(f"  {k:<15} {v}")
    print("")

    dv = pd.read_csv(paths["delta_v_csv"])
    props = pd.read_csv(paths["properties_csv"])
    systemic = pd.read_csv(paths["systemic_csv"])
    for df in (dv, props, systemic):
        df["ID"] = df["ID"].astype(int)
    props = props.set_index("ID")
    systemic = systemic.set_index("ID")

    ids = dv.loc[dv["z_sys_line"].notna(), "ID"].tolist()
    if args.group is not None:
        gpath = os.path.abspath(os.path.join(args.group_dir,
                                             f"lya_group_{args.group}.csv"))
        keep = set(pd.read_csv(gpath)["ID"].astype(int))
        ids = [i for i in ids if i in keep]
        print(f"group {args.group}: {gpath}")
    if args.id is not None:
        ids = [args.id]
    dv = dv.set_index("ID")

    results_cache, dja_cache = {}, {}
    written, skipped, warned = [], [], []

    for sid in ids:
        print(f"--- ID {sid} ---")
        if sid not in dv.index or pd.isna(dv.loc[sid, "z_sys_line"]):
            print("  [SKIP] no systemic line in delta_v csv")
            skipped.append(sid)
            continue
        drow = dv.loc[sid]
        line = str(drow["z_sys_line"])
        if line not in LINE_TEX:
            print(f"  [SKIP] systemic line {line} not supported here")
            skipped.append(sid)
            continue
        if sid not in props.index or not bool(props.loc[sid, "fit_success"]):
            print("  [SKIP] no successful Lya fit")
            skipped.append(sid)
            continue
        prow = props.loc[sid]

        g_short = systemic.loc[sid, f"z_{line}_grating"] if sid in systemic.index else np.nan
        if pd.isna(g_short):
            print(f"  [SKIP] no grating recorded for {line}")
            skipped.append(sid)
            continue
        g_full = GRATING_FULL[str(g_short)]
        lime_path = os.path.join(paths["spectra_root"], g_full,
                                 f"{sid}_{g_full}_spectra_lime.fits")
        npz_path = os.path.join(paths["muse_spec_dir"], f"{sid}_spectrum.npz")
        missing = [p for p in (lime_path, npz_path) if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] missing {missing}")
            skipped.append(sid)
            continue

        z_sys = float(drow["z_sys"])
        # DJA z of that grating places the LiMe band, as in the science run.
        # Read from the DJA match catalogue with the pipeline's own function,
        # falling back to the z_<grating> column of the line results CSV
        # (which holds the same value).
        z_band = np.nan
        cpath = oj.catalogue_path(g_full)
        if g_full not in dja_cache:
            dja_cache[g_full] = Table.read(cpath) if os.path.exists(cpath) else None
        if dja_cache[g_full] is not None:
            zd = oj.get_dja_redshift(dja_cache[g_full], sid)
            z_band = np.nan if zd is None else zd
        if line not in results_cache:
            rp = os.path.join(paths["catalog_dir"], f"{line}_results_by_JELS_ID.csv")
            results_cache[line] = (pd.read_csv(rp).assign(ID=lambda d: d["ID"].astype(int))
                                   .set_index("ID") if os.path.exists(rp) else None)
        res = results_cache[line]
        if not np.isfinite(z_band) and res is not None and sid in res.index and f"z_{g_short}" in res.columns:
            z_band = pd.to_numeric(res.loc[sid, f"z_{g_short}"], errors="coerce")
        if not np.isfinite(z_band):
            z_band = z_sys

        try:
            lya = load_lya(npz_path, prow, flp, args.n_mc, rng, args.fit_window,
                           args.sigma_min, args.alpha_max)
            sysl = load_systemic(lime_path, line, float(z_band), z_sys,
                                 args.n_mc, rng, oj, lj, args.sys_cont)
        except Exception as exc:
            print(f"  [SKIP] fit or load failed: {exc}")
            skipped.append(sid)
            continue

        dv_refit = C_KMS * (sysl["z_refit"] - z_sys) / (1.0 + z_sys)
        print(f"  line {line} ({g_full}), z_sys {z_sys:.6f}, "
              f"LiMe fit {sysl['z_refit']:.6f} ({dv_refit:+.2f} km/s)")
        print(f"  Lya model peak vs catalogue lya_peak_wave "
              f"{lya['peak_check_kms']:+.2f} km/s")
        if abs(dv_refit) > args.refit_tol:
            print(f"  [WARN] LiMe fit differs from catalogue z_sys by "
                  f"{dv_refit:+.1f} km/s. The source may have been refitted "
                  f"with manual overrides. The systemic dashed line stays at "
                  f"the catalogue z_sys.")
            warned.append(sid)

        lam_lya_sys = LYA_REST * (1.0 + z_sys)
        lam_line_sys = sysl["rest"] * (1.0 + z_sys)
        delta_v = float(drow["delta_v_kms"])

        # Display binning. MUSE matched to NIRSpec pixel width in km/s.
        v_pix_nir = pixel_kms(sysl["wave"], lam_line_sys) * max(args.nirspec_bin, 1)
        v_pix_muse = pixel_kms(lya["wave"], lam_lya_sys)
        mbin = (args.muse_bin if args.muse_bin is not None
                else max(1, int(round(v_pix_nir / v_pix_muse))))
        lya["wave_d"], lya["flux_d"], lya["err_d"] = rebin(
            lya["wave"], lya["flux"], lya["err"], mbin)
        sysl["wave_d"], sysl["flux_d"], sysl["err_d"] = rebin(
            sysl["wave"], sysl["flux"], sysl["err"], args.nirspec_bin)
        print(f"  pixel widths: NIRSpec {v_pix_nir:.0f} km/s, MUSE "
              f"{v_pix_muse:.0f} km/s, MUSE rebin x{mbin}, Lya MC {lya['n_mc']}/{args.n_mc}")

        src = dict(
            ID=sid, z_sys=z_sys, line=line, tex=LINE_TEX[line],
            grating=g_full, lya_snr=float(prow["lya_snr"]),
            z_sys_snr=float(drow["z_sys_snr"]), delta_v=delta_v,
            delta_v_err=float(drow["delta_v_err_kms"]),
            lam_lya_sys=lam_lya_sys, lam_line_sys=lam_line_sys,
            lam_lya_peak=wave_from_v(delta_v, lam_lya_sys),
        )

        outdir = os.path.join(paths["outroot"], str(sid))
        os.makedirs(outdir, exist_ok=True)
        p1 = os.path.join(outdir, f"{sid}_lya_systemic_panels.png")
        p2 = os.path.join(outdir, f"{sid}_lya_systemic_overlay.png")
        figure_panels(src, lya, sysl, p1, args.lya_vrange, args.sys_vrange,
                      args.dpi, args.doublet_pad)
        figure_overlay(src, lya, sysl, p2, args.overlay_vrange,
                       not args.no_overlay_fits, args.dpi)
        print(f"  saved  {p1}")
        print(f"  saved  {p2}")
        written.append(sid)

    print("")
    print("=" * 60)
    print(f"sources requested   {len(ids)}")
    print(f"figures written     {len(written)}  (2 per source)")
    print(f"skipped             {len(skipped)}  {skipped if skipped else ''}")
    print(f"refit warnings      {len(warned)}  {warned if warned else ''}")
    print(f"output root         {paths['outroot']}")
    print("=" * 60)


if __name__ == "__main__":
    main()