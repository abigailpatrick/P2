#!/usr/bin/env python3
"""
Systemic-anchored composite Lya spectra in bins of any master-catalogue column,
for measuring the shape of the line (velocity offset, blue-to-red ratio, peak
separation) rather than its strength.

Each bin is stacked with up to three normalisations (--norms).

  f1500   every spectrum divided by its UV continuum flux density at rest
          1500 AA (f1500_nJy from fit_muv_beta.py, sources with good_muv).
          The composite is in units of the continuum, so its integral over
          rest wavelength is a stacked rest-frame EW in AA. This is the main
          stack. Lya non-detections contribute their measured near-zero flux.
  halpha  every spectrum divided by its Halpha flux (--ha-col, default the
          observed, uncorrected ha_flux_best), only sources with Halpha
          S/N > --ha-snr-min. The composite is Lya per unit Halpha, per rest
          AA, so its integral divided by 8.7 is a stacked escape fraction.
          With uncorrected Halpha this fesc has no dust or slit-loss
          correction. Swap --ha-col once the corrections are final.
  none    no normalisation. Rest-frame flux density f_obs (1+z) in units of
          1e-20 erg/s/cm2/AA, so the brightest sources dominate. Not run by
          default, add it with --norms f1500 halpha none.

Per source, before stacking
---------------------------
  1. Read the MUSE aperture spectrum {ID}_spectrum.npz
     (ap_extract_specs_grating.py, air wavelengths, MUSE flux units).
  2. Pixels with zero or non-finite variance (the AO gap, the cube edges) are
     treated as missing. A source with any missing pixel within
     +- --core-window km/s of systemic is dropped (the AO-gap sources).
  3. Convolve to a common resolution, --target-lsf-kms (default 190 km/s
     FWHM), with a Gaussian of width sqrt(target^2 - LSF^2), where LSF is the
     MUSE LSF at the observed Lya wavelength (Bacon et al. 2017, the same
     function as fit_lya_properties_grating.py). The MUSE LSF runs from about
     185 km/s at z = 3 to about 90 km/s at z = 6, so without this the bins
     would have different resolution.
  4. Air to vacuum (Ciddor 1996), shift to the rest frame with z_sys and
     interpolate onto a common rest-frame grid of --bin AA (default 0.2 AA,
     about 49 km/s at Lya).
  5. Normalise (f1500 or halpha, above).

Per bin
-------
  mean    sigma-clipped mean per pixel (--clip-sigma, default 3, about the
          median with a MAD scatter). Linear, so integrals keep their meaning.
  median  per-pixel median, kept as a check.
  Errors  bootstrap over sources (--nboot). The 16th-84th percentile band is
          drawn on the plots and every measurement below is repeated on every
          bootstrap realisation, so the measurements get errors too.

Measurements on each composite (lya_stack_measurements.csv)
----------------------------------------------------------
  red_peak_kms       peak of a single LSF-convolved skewed Gaussian fitted over
                     --red-fit-window (the same model as the individual fits,
                     with the common LSF). red_peak_int_kms is the peak of the
                     intrinsic profile and fwhm_int_kms its FWHM.
  red_centroid_kms   flux-weighted mean velocity over --centroid-window
                     (default 0 to +1000 km/s). Model-free, robust at low S/N.
  blue_red_ratio     integral over --blue-window (-800 to 0) divided by the
                     integral over --red-window (0 to +800), as Hayes et al.
                     (2023).
  peak_sep_kms       red minus blue peak from a two-component fit (blue
                     skewed Gaussian with alpha <= 0, red with alpha >= 0,
                     both LSF-convolved). Only quoted when the blue component
                     flux has bootstrap S/N > --blue-snr-min (default 3).
                     Otherwise blank, and blue_flux_3sig_limit gives 3x the
                     bootstrap scatter of the blue-window integral.
  integral           integral over --int-window (default -1000 to +1000 km/s).
                     f1500 stack: ew_rest_aa. halpha stack: lya_ha_ratio, and
                     fesc_stack = lya_ha_ratio / 8.7.
Each quantity has _p16 and _p84 columns from the bootstrap.

Sample
------
From the master catalogue (build_master_catalog.py): in_muse, is_primary and
a z_sys. Then dropped, with the reason written to the members CSV:
  |zsys_mismatch_kms| > --mismatch-tol-kms (default 300). Those sources'
      continuum subtraction and extraction were made with a z far from the
      current z_sys, so their Lya may sit in the wrong part of the mask.
  missing pixels within the core window (AO gap).
  no spectrum file.

Bins
----
--splits takes any master column with a split value, e.g.
  --splits z_sys:3.5 M_UV:-19
which makes, for each, a "< value" and a ">= value" bin. An "all" stack is
always made too. For M_UV only sources with good_muv are binned.

Outputs (all to --outdir, full paths printed)
---------------------------------------------
  lya_stack_members.csv                 one row per candidate source, with
                                        its bins, normalisers and why it was
                                        dropped, if it was
  lya_stack_<norm>_<bin>.csv            composite: dv_kms, wave_rest, mean,
                                        median, their p16/p84, n_spec
  lya_stack_measurements.csv            one row per norm, bin and combiner
  lya_stack_<norm>_<split>.png          the two bins of a split, plus an
                                        overlay with each scaled to its peak
  lya_stack_<norm>_all.png

Usage
-----
python stack_lya_systematic.py
python stack_lya_systematic.py --splits z_sys:3.5 M_UV:-19 beta:-2 --nboot 1000
python stack_lya_systematic.py --norms none --nboot 100
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import ndtr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as pc                      # noqa: E402
import fit_lya_properties_grating as flp    # noqa: E402  LSF, air/vac, skew model

C_KMS = pc.C_KMS
LYA = pc.LYA_REST
C_AA_S = 2.99792458e18          # speed of light in AA/s
NJY_CGS = 1e-32                 # erg/s/cm2/Hz per nJy
CASE_B = 8.7                    # intrinsic Lya/Halpha
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

NORMS = ["f1500", "halpha", "none"]
NORM_YLABEL = {
    "f1500": r"$f_\lambda / f_{\lambda,1500}$",
    "halpha": r"$f_{\lambda,{\rm rest}} / F_{\rm H\alpha}$ (Å$^{-1}$)",
    "none": r"$f_{\lambda,{\rm rest}}$ ($10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)",
}
NICE = {"z_sys": r"z_{\rm sys}", "M_UV": r"M_{\rm UV}", "beta": r"\beta"}


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Systemic-anchored Lya stacks.")
    p.add_argument("--master-csv",
                   default=f"{pc.P2_ROOT}/master_catalog/p2_master_catalog.csv")
    p.add_argument("--spec-dir",
                   default=f"{pc.P2_ROOT}/MUSE_subcubes/dataproducts",
                   help="Directory of {ID}_spectrum.npz files.")
    p.add_argument("--outdir", default=f"{pc.P2_ROOT}/plots/stacks")
    p.add_argument("--splits", nargs="*", default=["z_sys:3.5", "M_UV:-19"],
                   help="column:value pairs from the master catalogue.")
    p.add_argument("--target-lsf-kms", type=float, default=190.0,
                   help="Common resolution (FWHM) every spectrum is smoothed to.")
    p.add_argument("--bin", type=float, default=0.2, help="Rest-frame bin, AA.")
    p.add_argument("--vmin", type=float, default=-2000.0)
    p.add_argument("--vmax", type=float, default=2000.0)
    p.add_argument("--core-window", type=float, default=800.0,
                   help="Drop a source with missing pixels within +- this (km/s).")
    p.add_argument("--mismatch-tol-kms", type=float, default=300.0,
                   help="Drop sources whose MUSE products used a z this far "
                        "from the current z_sys. Negative to switch off.")
    p.add_argument("--norms", nargs="+", choices=NORMS, default=["f1500", "halpha"],
                   help="Normalisations to run. 'none' stacks the rest-frame flux "
                        "density with no normalisation, so bright sources dominate.")
    p.add_argument("--ha-col", default="ha_flux_best")
    p.add_argument("--ha-snr-col", default="ha_flux_best_snr")
    p.add_argument("--ha-snr-min", type=float, default=3.0)
    p.add_argument("--muse-flux-unit", type=float, default=1e-20,
                   help="erg/s/cm2/AA per MUSE flux unit.")
    p.add_argument("--clip-sigma", type=float, default=3.0)
    p.add_argument("--min-n", type=int, default=3,
                   help="Composite pixels with fewer spectra are blanked.")
    p.add_argument("--nboot", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--red-fit-window", type=float, nargs=2, default=[-200.0, 1200.0])
    p.add_argument("--double-fit-window", type=float, nargs=2, default=[-1000.0, 1200.0])
    p.add_argument("--blue-window", type=float, nargs=2, default=[-800.0, 0.0])
    p.add_argument("--red-window", type=float, nargs=2, default=[0.0, 800.0])
    p.add_argument("--centroid-window", type=float, nargs=2, default=[0.0, 1000.0])
    p.add_argument("--int-window", type=float, nargs=2, default=[-1000.0, 1000.0])
    p.add_argument("--blue-snr-min", type=float, default=3.0)
    p.add_argument("--alpha-max", type=float, default=15.0)
    return p.parse_args()


# ============================================================
# Per-source preparation
# ============================================================

def v_to_rest(v):
    return LYA * (1.0 + np.asarray(v, float) / C_KMS)


def kms(lam):
    return C_KMS * (np.asarray(lam, float) - LYA) / LYA


def load_spectrum(path):
    d = np.load(path)
    w = np.asarray(d["wave"], float)
    f = np.asarray(d["flux"], float)
    var = np.asarray(d["var"], float)
    good = np.isfinite(f) & np.isfinite(var) & (var > 0)
    return w, f, good


def core_has_gap(w_air, good, z, core_kms):
    """True if a missing pixel falls within +-core_kms of systemic Lya."""
    lya_air = flp.vac_to_air(LYA * (1.0 + z))
    half = lya_air * core_kms / C_KMS
    sel = (w_air >= lya_air - half) & (w_air <= lya_air + half)
    return bool(np.any(~good[sel])) or sel.sum() == 0


def smooth_to_target(w_air, f, good, z, target_kms):
    """Gaussian-smooth so the effective LSF at Lya is target_kms FWHM.

    Normalised convolution, so missing pixels neither leak in nor bias the
    result. Returns the smoothed flux and the native LSF FWHM in km/s.
    """
    lya_air = flp.vac_to_air(LYA * (1.0 + z))
    sig_lsf = float(flp.muse_lsf_sigma(lya_air))                  # AA
    sig_tgt = target_kms * FWHM_TO_SIGMA * lya_air / C_KMS         # AA
    lsf_kms = sig_lsf / FWHM_TO_SIGMA / lya_air * C_KMS
    if sig_tgt <= sig_lsf:
        return np.where(good, f, np.nan), lsf_kms
    dpix = float(np.median(np.diff(w_air)))
    k = np.sqrt(sig_tgt ** 2 - sig_lsf ** 2) / dpix
    num = gaussian_filter1d(np.where(good, f, 0.0), k, mode="constant")
    den = gaussian_filter1d(good.astype(float), k, mode="constant")
    out = np.where(den > 0.5, num / np.where(den > 0, den, 1.0), np.nan)
    out[~good] = np.nan
    return out, lsf_kms


def to_rest_grid(w_air, f, z, grid):
    """Interpolate onto the rest-frame vacuum grid. Grid points closer than
    one native pixel to a missing pixel are left blank."""
    w_rest = flp.air_to_vac(w_air) / (1.0 + z)
    ok = np.isfinite(f)
    if ok.sum() < 2:
        return np.full(grid.size, np.nan)
    out = np.interp(grid, w_rest[ok], f[ok], left=np.nan, right=np.nan)
    bad = w_rest[~ok]
    if bad.size:
        step = float(np.median(np.diff(w_rest)))
        idx = np.clip(np.searchsorted(bad, grid), 0, bad.size - 1)
        near = np.minimum(np.abs(grid - bad[idx]),
                          np.abs(grid - bad[np.maximum(idx - 1, 0)]))
        out[near < step] = np.nan
    return out


def f1500_flambda(f1500_njy, z):
    """Observed-frame f_lambda (erg/s/cm2/AA) at rest 1500 AA."""
    lam = 1500.0 * (1.0 + z)
    return f1500_njy * NJY_CGS * C_AA_S / lam ** 2


# ============================================================
# Stacking
# ============================================================

def combine(specs, clip_sigma, min_n):
    """Sigma-clipped mean and median per pixel, plus the number of spectra."""
    n = np.sum(np.isfinite(specs), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped = sigma_clip(specs, sigma=clip_sigma, maxiters=5, axis=0,
                             cenfunc="median", stdfunc="mad_std", masked=True)
        mean = np.ma.mean(clipped, axis=0).filled(np.nan)
        median = np.nanmedian(specs, axis=0)
    mean[n < min_n] = np.nan
    median[n < min_n] = np.nan
    return mean, median, n


# ============================================================
# Measurements
# ============================================================

def skew_lsf(x, ftot, mu, sigma, alpha, sig_lsf):
    """Same model as fit_lya_properties_grating.skew_model_lsf (skew-normal
    convolved with a Gaussian LSF, exact), written with numpy so the many
    bootstrap fits run quickly. skew-normal pdf = 2/s phi(t) Phi(alpha t)."""
    s, al = flp.convolved_skew_params(sigma, alpha, sig_lsf)
    t = (x - mu) / s
    return ftot * 2.0 / s * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi) * ndtr(al * t)


def integrate(v, y, lo, hi, dlam):
    sel = (v >= lo) & (v <= hi)
    if sel.sum() == 0 or np.mean(np.isfinite(y[sel])) < 0.8:
        return np.nan
    return float(np.nansum(y[sel]) * dlam)


def fit_single(wave, y, err, sig_lsf, window, alpha_max, p_start=None):
    """One LSF-convolved skewed Gaussian.
    Returns (peak_obs_kms, peak_int_kms, fwhm_int_kms, params)."""
    v = kms(wave)
    sel = (v >= window[0]) & (v <= window[1]) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    nan = (np.nan, np.nan, np.nan, None)
    if sel.sum() < 6:
        return nan
    w, f, e = wave[sel], y[sel], err[sel]
    ipk = int(np.nanargmax(f))
    p0 = [max(f[ipk], 1e-30) * np.sqrt(2 * np.pi), w[ipk] - 0.3, 1.0, 2.0]
    if p_start is not None:
        p0 = list(p_start)
    lo = [0.0, v_to_rest(window[0]), 0.05, 0.0]
    hi = [np.inf, v_to_rest(window[1]), 10.0, alpha_max]
    p0 = list(np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9))

    def model(x, a, mu, s, al):
        return skew_lsf(x, a, mu, s, al, sig_lsf)
    try:
        popt, _ = curve_fit(model, w, f, p0=p0, sigma=e, absolute_sigma=True,
                            bounds=(lo, hi), maxfev=20000)
    except Exception:
        return nan
    a, mu, s, al = popt
    s_obs, al_obs = flp.convolved_skew_params(s, al, sig_lsf)
    pk_obs = flp.compute_peak_from_model(mu, s_obs, al_obs)
    pk_int = flp.compute_peak_from_model(mu, s, al)
    fwhm = flp.compute_fwhm_from_model(mu, s, al)
    return float(kms(pk_obs)), float(kms(pk_int)), float(fwhm / LYA * C_KMS), popt


def fit_double(wave, y, err, sig_lsf, window, alpha_max, p_start=None):
    """Blue (alpha <= 0) plus red (alpha >= 0) skewed Gaussians, LSF-convolved.
    Returns (blue_flux, blue_peak_kms, red_peak_kms, params)."""
    v = kms(wave)
    sel = (v >= window[0]) & (v <= window[1]) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    nan = (np.nan, np.nan, np.nan, None)
    if sel.sum() < 10:
        return nan
    w, f, e = wave[sel], y[sel], err[sel]
    amp = max(np.nanmax(f), 1e-30) * np.sqrt(2 * np.pi)
    p0 = [0.2 * amp, v_to_rest(-300), 1.0, -2.0, amp, v_to_rest(150), 1.0, 2.0]
    if p_start is not None:
        p0 = list(p_start)
    lo = [0.0, v_to_rest(-1000), 0.05, -alpha_max, 0.0, v_to_rest(-300), 0.05, 0.0]
    hi = [np.inf, v_to_rest(0), 10.0, 0.0, np.inf, v_to_rest(1000), 10.0, alpha_max]
    p0 = list(np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9))

    def model(x, ab, mb, sb, alb, ar, mr, sr, alr):
        return (skew_lsf(x, ab, mb, sb, alb, sig_lsf)
                + skew_lsf(x, ar, mr, sr, alr, sig_lsf))
    try:
        popt, _ = curve_fit(model, w, f, p0=p0, sigma=e, absolute_sigma=True,
                            bounds=(lo, hi), maxfev=40000)
    except Exception:
        return nan
    ab, mb, sb, alb, ar, mr, sr, alr = popt
    sbo, albo = flp.convolved_skew_params(sb, alb, sig_lsf)
    sro, alro = flp.convolved_skew_params(sr, alr, sig_lsf)
    return (float(ab), float(kms(flp.compute_peak_from_model(mb, sbo, albo))),
            float(kms(flp.compute_peak_from_model(mr, sro, alro))), popt)


def measure(wave, v, y, err, a, sig_lsf, dlam, starts=(None, None)):
    """All measurements on one composite.
    Returns (dict, (single-fit params, double-fit params)). Passing the main
    composite's params as starts makes the bootstrap refits converge fast."""
    out = {}
    pk, pk_int, fwhm, popt = fit_single(wave, y, err, sig_lsf, a.red_fit_window,
                                        a.alpha_max, starts[0])
    out["red_peak_kms"], out["red_peak_int_kms"], out["fwhm_int_kms"] = pk, pk_int, fwhm
    sel = (v >= a.centroid_window[0]) & (v <= a.centroid_window[1]) & np.isfinite(y)
    tot = np.sum(y[sel])
    out["red_centroid_kms"] = (float(np.sum(v[sel] * y[sel]) / tot)
                               if sel.sum() and tot > 0 else np.nan)
    blue = integrate(v, y, *a.blue_window, dlam)
    red = integrate(v, y, *a.red_window, dlam)
    out["blue_int"], out["red_int"] = blue, red
    out["blue_red_ratio"] = blue / red if np.isfinite(red) and red > 0 else np.nan
    bf, bpk, rpk, popt2 = fit_double(wave, y, err, sig_lsf, a.double_fit_window,
                                     a.alpha_max, starts[1])
    out["blue_flux_fit"], out["blue_peak_kms"], out["red_peak2_kms"] = bf, bpk, rpk
    out["peak_sep_kms"] = rpk - bpk if np.isfinite(rpk) and np.isfinite(bpk) else np.nan
    out["integral"] = integrate(v, y, *a.int_window, dlam)
    return out, (popt, popt2)


# ============================================================
# Plotting
# ============================================================

def bin_label(col, op, val):
    name = NICE.get(col, col.replace("_", r"\_"))
    sym = "<" if op == "lt" else r"\geq"
    return rf"${name} {sym} {val:g}$"


def draw_panel(ax, v, res, colour, label, a, sig_lsf, ylabel):
    ax.fill_between(v, res["mean_p16"], res["mean_p84"], step="mid",
                    color=colour, alpha=0.25, lw=0)
    ax.step(v, res["mean"], where="mid", color=colour, lw=1.3, label="clipped mean")
    ax.step(v, res["median"], where="mid", color=colour, lw=0.9, ls="--",
            alpha=0.8, label="median")
    if res["popt"] is not None:
        vf = np.linspace(*a.red_fit_window, 400)
        ax.plot(vf, flp.skew_model_lsf(v_to_rest(vf), *res["popt"], sig_lsf),
                color="k", lw=0.9, alpha=0.8, label="skewed Gaussian fit")
    ax.axvline(0, color="0.3", ls=":", lw=1)
    ax.axhline(0, color="0.6", lw=0.6)
    m = res["meas"]["mean"]
    txt = (f"{label}\nN = {res['n']}\n"
           rf"$\Delta v_{{\rm peak}}$ = {m['red_peak_kms']:.0f} km s$^{{-1}}$" "\n"
           f"B/R = {m['blue_red_ratio']:.2f}")
    ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlim(a.vmin, a.vmax)


def plot_split(v, results, keys, labels, colours, norm, a, sig_lsf, out_png):
    nb = len(keys)
    nrow = nb + (1 if nb > 1 else 0)
    fig, axes = plt.subplots(nrow, 1, figsize=(5.5, 2.6 * nrow), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, k, lab, col in zip(axes, keys, labels, colours):
        draw_panel(ax, v, results[k], col, lab, a, sig_lsf, NORM_YLABEL[norm])
    if nb > 1:
        ax = axes[-1]
        core = np.abs(v) <= 1000
        for k, lab, col in zip(keys, labels, colours):
            y = results[k]["mean"]
            s = np.nanmax(y[core]) if np.any(np.isfinite(y[core])) else 1.0
            ax.step(v, y / s, where="mid", color=col, lw=1.3, label=lab)
        ax.axvline(0, color="0.3", ls=":", lw=1)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("scaled to peak")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_xlim(a.vmin, a.vmax)
    axes[0].legend(fontsize=7, loc="upper right", frameon=False)
    axes[-1].set_xlabel(r"$\Delta v$ (km s$^{-1}$)")
    axes[0].set_title(f"{norm} normalised, smoothed to {a.target_lsf_kms:g} km/s",
                      fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    a = parse_args()
    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print("[CONFIG]")
    print(f"  master csv     {os.path.abspath(a.master_csv)}")
    print(f"  spectra        {os.path.abspath(a.spec_dir)}")
    print(f"  output dir     {outdir}")
    print(f"  splits         {' '.join(a.splits)}")
    print(f"  common LSF     {a.target_lsf_kms:g} km/s FWHM, bin {a.bin:g} AA rest "
          f"({a.bin / LYA * C_KMS:.0f} km/s)")
    print(f"  Halpha         {a.ha_col}, S/N > {a.ha_snr_min:g}")
    print(f"  bootstrap      {a.nboot}\n")

    m = pd.read_csv(a.master_csv)
    m["ID"] = m["ID"].astype(int)
    prim = m["is_primary"].astype(str).str.lower().isin(["true", "1"])
    cand = m[(m["in_muse"] == 1) & prim & m["z_sys"].notna()].copy().reset_index(drop=True)
    print(f"[SAMPLE] in MUSE, primary, with z_sys: {len(cand)}")

    grid = np.arange(v_to_rest(a.vmin) - a.bin, v_to_rest(a.vmax) + 2 * a.bin, a.bin)
    v = kms(grid)
    sig_lsf_rest = a.target_lsf_kms * FWHM_TO_SIGMA * LYA / C_KMS

    reasons, specs, lsfs = [], {}, {}
    for _, r in cand.iterrows():
        sid, z = int(r["ID"]), float(r["z_sys"])
        mm = r.get("zsys_mismatch_kms", np.nan)
        path = os.path.join(a.spec_dir, f"{sid}_spectrum.npz")
        if a.mismatch_tol_kms >= 0 and np.isfinite(mm) and abs(mm) > a.mismatch_tol_kms:
            reasons.append(f"MUSE products made {mm:+.0f} km/s from z_sys")
            continue
        if not os.path.exists(path):
            reasons.append("no spectrum file")
            continue
        w, f, good = load_spectrum(path)
        if core_has_gap(w, good, z, a.core_window):
            reasons.append(f"missing pixels within +-{a.core_window:g} km/s (AO gap)")
            continue
        fs, lsf_kms = smooth_to_target(w, f, good, z, a.target_lsf_kms)
        specs[sid] = to_rest_grid(w, fs * a.muse_flux_unit, z, grid)  # erg/s/cm2/AA, observed f_lambda
        lsfs[sid] = lsf_kms
        reasons.append("")
    cand["dropped"] = reasons
    cand["native_lsf_kms"] = cand["ID"].map(lsfs)
    for _, r in cand[cand["dropped"] != ""].iterrows():
        print(f"  dropped {int(r['ID'])}: {r['dropped']}")
    over = cand[cand["native_lsf_kms"] > a.target_lsf_kms]
    if len(over):
        print(f"  [WARN] native LSF above the target for {over['ID'].tolist()}, not smoothed")

    use = cand[cand["dropped"] == ""].copy().reset_index(drop=True)
    print(f"[SAMPLE] stacked candidates: {len(use)}\n")

    # Normalisers. Spectra are observed-frame f_lambda in erg/s/cm2/AA.
    #   f1500:  f_obs / f_cont,obs(1500(1+z))         dimensionless
    #   halpha: f_obs (1+z) / F_Ha = f_obs / (F_Ha/(1+z))   per rest AA
    good_uv = use["good_muv"].astype(str).str.lower().isin(["true", "1"])
    use["norm_f1500"] = np.where(good_uv & (use["f1500_nJy"] > 0),
                                 f1500_flambda(use["f1500_nJy"], use["z_sys"]), np.nan)
    has_ha = (use[a.ha_col] > 0) & (use[a.ha_snr_col] > a.ha_snr_min)
    use["norm_halpha"] = np.where(has_ha, use[a.ha_col] / (1.0 + use["z_sys"]), np.nan)
    # none: rest-frame f_lambda = f_obs (1+z), in units of 1e-20 erg/s/cm2/AA
    use["norm_none"] = 1e-20 / (1.0 + use["z_sys"])

    bins = {"all": [("all", "all", np.ones(len(use), bool))]}
    for s in a.splits:
        col, val = s.split(":")
        val = float(val)
        if col not in use.columns:
            print(f"[WARN] split column {col} not in the master, skipped")
            continue
        x = pd.to_numeric(use[col], errors="coerce")
        ok = x.notna()
        if col == "M_UV":
            ok &= good_uv
        bins[col] = [(f"{col}_lt{val:g}", bin_label(col, "lt", val), (ok & (x < val)).values),
                     (f"{col}_ge{val:g}", bin_label(col, "ge", val), (ok & (x >= val)).values)]
        use[f"bin_{col}"] = np.where(ok, np.where(x < val, f"lt{val:g}", f"ge{val:g}"), "")

    rows = []
    colours = ["#2b6cb0", "#c05621"]
    for norm in a.norms:
        nrm = use[f"norm_{norm}"].values
        print(f"[{norm}] sources with a normaliser: {int(np.isfinite(nrm).sum())} of {len(use)}")
        for split, blist in bins.items():
            results, keys, labels = {}, [], []
            for key, label, sel in blist:
                sel = sel & np.isfinite(nrm)
                ids = use.loc[sel, "ID"].astype(int).tolist()
                if len(ids) < a.min_n:
                    print(f"  [WARN] {norm} {key}: only {len(ids)} sources, skipped")
                    continue
                S = np.array([specs[i] for i in ids]) / nrm[sel][:, None]
                mean, median, nspec = combine(S, a.clip_sigma, a.min_n)
                bm = np.empty((a.nboot, grid.size))
                bd = np.empty((a.nboot, grid.size))
                for b in range(a.nboot):
                    idx = rng.integers(0, len(ids), len(ids))
                    bm[b], bd[b], _ = combine(S[idx], a.clip_sigma, a.min_n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = {"mean": mean, "median": median, "n": len(ids),
                           "mean_p16": np.nanpercentile(bm, 16, axis=0),
                           "mean_p84": np.nanpercentile(bm, 84, axis=0),
                           "median_p16": np.nanpercentile(bd, 16, axis=0),
                           "median_p84": np.nanpercentile(bd, 84, axis=0),
                           "meas": {}, "popt": None}
                    for comb, comp, boots in (("mean", mean, bm), ("median", median, bd)):
                        err = np.nanstd(boots, axis=0)
                        meas, popts = measure(grid, v, comp, err, a, sig_lsf_rest, a.bin)
                        if comb == "mean":
                            res["popt"] = popts[0]
                        bmeas = pd.DataFrame([measure(grid, v, boots[b], err, a,
                                                      sig_lsf_rest, a.bin, popts)[0]
                                              for b in range(a.nboot)])
                        res["meas"][comb] = meas
                        row = {"norm": norm, "split": split, "bin": key,
                               "combiner": comb, "n_sources": len(ids)}
                        for q, val in meas.items():
                            row[q] = val
                            has = bmeas[q].notna().any()
                            row[q + "_p16"] = float(np.nanpercentile(bmeas[q], 16)) if has else np.nan
                            row[q + "_p84"] = float(np.nanpercentile(bmeas[q], 84)) if has else np.nan
                        bf_sd = float(np.nanstd(bmeas["blue_flux_fit"]))
                        row["blue_snr"] = meas["blue_flux_fit"] / bf_sd if bf_sd > 0 else np.nan
                        row["blue_flux_3sig_limit"] = 3.0 * float(np.nanstd(bmeas["blue_int"]))
                        if not (row["blue_snr"] > a.blue_snr_min):
                            for q in ("peak_sep_kms", "peak_sep_kms_p16", "peak_sep_kms_p84"):
                                row[q] = np.nan
                        if norm == "f1500":
                            row["ew_rest_aa"] = row["integral"]
                        elif norm == "halpha":
                            row["lya_ha_ratio"] = row["integral"]
                            for s in ("", "_p16", "_p84"):
                                row["fesc_stack" + s] = row["integral" + s] / CASE_B
                        row["ids"] = ";".join(str(i) for i in ids)
                        rows.append(row)
                results[key] = res
                keys.append(key)
                labels.append(label)
                out_csv = os.path.join(outdir, f"lya_stack_{norm}_{key}.csv")
                pd.DataFrame({"dv_kms": v, "wave_rest": grid, "mean": mean,
                              "mean_p16": res["mean_p16"], "mean_p84": res["mean_p84"],
                              "median": median, "median_p16": res["median_p16"],
                              "median_p84": res["median_p84"], "n_spec": nspec}
                             ).to_csv(out_csv, index=False)
                mm = res["meas"]["mean"]
                print(f"  {norm:6s} {key:14s} N={len(ids):3d}  "
                      f"peak {mm['red_peak_kms']:6.0f}  centroid {mm['red_centroid_kms']:6.0f}  "
                      f"B/R {mm['blue_red_ratio']:5.2f}")
                print(f"         written  {out_csv}")
            if keys:
                out_png = os.path.join(outdir, f"lya_stack_{norm}_{split}.png")
                plot_split(v, results, keys, labels,
                           colours if len(keys) > 1 else ["0.15"],
                           norm, a, sig_lsf_rest, out_png)
                print(f"         written  {out_png}")
        print("")

    meas_csv = os.path.join(outdir, "lya_stack_measurements.csv")
    front = ["norm", "split", "bin", "combiner", "n_sources"]
    md = pd.DataFrame(rows)
    md = md[front + [c for c in md.columns if c not in front + ["ids"]] + ["ids"]]
    md.to_csv(meas_csv, index=False)

    keep = ["ID", "z_sys", "z_sys_line", "M_UV", "good_muv", "f1500_nJy",
            a.ha_col, a.ha_snr_col, "lya_det_98", "manual", "zsys_mismatch_kms",
            "native_lsf_kms", "dropped"]
    mem = cand[[c for c in keep if c in cand.columns]].copy()
    mem["used_f1500"] = mem["ID"].isin(use.loc[use["norm_f1500"].notna(), "ID"])
    mem["used_halpha"] = mem["ID"].isin(use.loc[use["norm_halpha"].notna(), "ID"])
    for c in [c for c in use.columns if c.startswith("bin_")]:
        mem[c] = mem["ID"].map(use.set_index("ID")[c])
    mem_csv = os.path.join(outdir, "lya_stack_members.csv")
    mem.to_csv(mem_csv, index=False)

    print(f"written  {meas_csv}")
    print(f"written  {mem_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()