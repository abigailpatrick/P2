#!/usr/bin/env python
"""Fit [OIII], Hbeta, Halpha (+[NII]) and [OII] in every NIRSpec grating
spectrum of the PRIMER + MINERVA sample with LiMe 2.4.3 (conda env312).

Replaces lime_OIII_jointfit.py, lime_lines_jointfit.py and
lime_OIII_singlereview.py, for the PRIMER + MINERVA sample (Isaac IDs) and
the spectra in jwst_spectra_pm/.

Changes from the old scripts
----------------------------
- Every fit uses the flank continuum (LiMe "adjacent", a straight line fitted
  to both flanks with their errors). LiMe's default draws the continuum
  through the two end pixels of the line region, so two noisy pixels could
  set it. Fluxes and widths come from the same fit, so there is no refit.
- Detection follows the LiMe paper (Fernandez et al. 2024, Sect. 5): the
  fitted amplitude over the flux scatter in the adjacent bands, A/noise,
  must be >= 5 (its detection boundary for lines about as wide as the
  resolution). The fitted FWHM must cover >= 1 pixel (rejects single-pixel
  spikes) and the centre error must be <= 2.5 Angstrom. The column snr holds
  A/noise. LiMe's own snr_line is kept as snr_lime for reference only.
- A line is fitted only if the spectrum has data at its centre, so a line in
  the detector gap or past the end of the spectrum is not fitted. [OIII]
  needs 5007 itself: a source whose 5007 falls in the gap gets no [OIII] fit,
  even if 4959 is on the spectrum.
- Halpha z comes from the Halpha + [NII] blend.

Lines
-----
  OIII   [OIII] 4959,5007 fitted as one blend, 4959 tied to 5007 in
         kinematics and amplitude (ratio 2.98). 5007 alone when 4959 is off
         the spectrum or in the gap. z from 5007.
  Hbeta  single Gaussian. z, flux.
  Ha     Halpha blended with [NII] 6548, 6584 (kinematics tied to Halpha,
         6584/6548 flux ratio 3.05). z, S/N, success, Halpha flux, FWHM and
         [NII] 6584 flux (method blend_NII). A single-Gaussian Halpha fit is
         kept as z_single, z_single_err, snr_single for comparison, and gives
         z only when the blend band is off the spectrum (method single).
  OII    [OII] 3726,3729 blend, kinematics tied, both amplitudes free (the
         ratio depends on density). z from 3729.
Every fit uses the DJA redshift of that grating to place the band, unless
--z is given. z is always computed from the fitted observed centre and the
vacuum rest wavelength.

Success: A/noise >= --snr-min (5) AND FWHM >= --min-fwhm-pix (1 pixel) AND
centre error <= --centre-err-max (2.5 Angstrom). Fluxes and widths are stored
for every fit, whatever the flag.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse.csv    IDs, gratings and DJA z per grating
  jwst_spectra_pm/<G>_<F>/<ID>_<G>_<F>_spectra_lime.fits
                                              from make_lime_spectra_format.py

Outputs
-------
  lime_fits/<ID>/<line>_fits/<ID>_<G>_<F>_<line>_contsub.png
  lime_fits/<ID>/<line>_fits/<ID>_<G>_<F>_<line>_fit.png
      data, components, total model, continuum and residuals on a linear
      axis, for the fit that gave z (the [NII] blend for Halpha)
  jwst_catalogs/lime_<line>_fits.csv
      one row per (ID, grating) with a spectrum: status (fitted,
      out_of_range, failed), z_dja, z_used, method, z, z_err, snr,
      centre_err_AA, success, the flux and width columns, the band settings
      and the figure paths.
  jwst_catalogs/lime_<line>_summary.csv
      one row per ID: gratings_fitted, n_success, the best successful fit
      (highest snr) as z_<line>, z_<line>_err, z_<line>_snr,
      z_<line>_grating, and for Hbeta and Ha the best flux (highest flux
      S/N), and for Ha the best FWHM (Halpha flux S/N >= 5, H grating
      preferred, then highest S/N).
build_zsys_catalog.py reads the summaries.

Batch and review modes
----------------------
Without --ids every source is fitted and the CSVs of the lines fitted are
written fresh. With --ids only those sources (and --gratings, if given) are
refitted, and only their rows are replaced in the existing CSVs. The
overrides below are for this review of individual sources.

  --z FLOAT            absolute redshift used to place the bands (one ID only)
  --line-margin FLOAT  rest-frame Angstrom the line region extends beyond
                       each outer line (15)
  --cont-gap FLOAT     gap between the line region and each flank (5)
  --cont-width FLOAT   width of each continuum flank (20)
  --force-single       [OIII] only, fit 5007 alone even when 4959 is in range
  --dry-run            fit and plot, print, write no CSVs

Usage
-----
conda activate env312
python lime_fit_lines.py                                  # all sources, all lines
python lime_fit_lines.py --lines OIII Ha                  # selected lines
python lime_fit_lines.py --ids 57866                      # one source, patch rows
python lime_fit_lines.py --ids 57866 --lines OIII --gratings G235M --z 5.9412
python lime_fit_lines.py --ids 57866 --lines OIII --force-single --dry-run
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lime


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
P2 = "/ceph/cephfs/apatrick/P2"
CATALOG = f"{P2}/jwst_catalogs/primer_minerva_in_muse.csv"
SPECTRA_ROOT = f"{P2}/jwst_spectra_pm"
FIG_ROOT = f"{P2}/lime_fits"
CSV_DIR = f"{P2}/jwst_catalogs"

GRATINGS = ["G235M", "G235H", "G395M", "G395H"]
GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}

# ----------------------------------------------------------------------
# Fit settings, unchanged from the previous scripts
# ----------------------------------------------------------------------
LINE_MARGIN = 15.0
CONT_GAP = 5.0
CONT_WIDTH = 20.0
SNR_MIN = 5.0          # A / sigma_noise, LiMe paper detection boundary
CENTRE_ERR_MAX_AA = 2.5
MIN_FWHM_PIX = 1.0     # fitted FWHM in pixels (NIRSpec samples a resolution element with ~2 pixels)

C_KMS = 299792.458
GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
OIII_RATIO_THEORY = 2.98
NII_FLUX_RATIO = 3.05
R_SCALE = 1.3
FWHM_SNR_MIN = 5.0
CONT_SOURCE = "adjacent"   # flank continuum for every fit, see fit_line

# Vacuum rest wavelengths, Angstrom, and LiMe labels.
OIII_5007, OIII_4959 = 5006.843, 4958.911
HB = 4862.683
HA, NII_RED, NII_BLUE = 6564.632, 6585.270, 6549.860
OII_3729, OII_3726 = 3729.875, 3727.092

L_O3_5007, L_O3_4959, L_O3_BLEND = "O3_5007A", "O3_4959A", "O3_5007A_b"
L_HB = "H1_4861A"
L_HA, L_NII_RED, L_NII_BLUE, L_HA_BLEND = "H1_6563A", "N2_6584A", "N2_6548A", "H1_6563A_b"
L_O2_3729, L_O2_3726, L_O2_BLEND = "O2_3729A", "O2_3726A", "O2_3729A_b"

LINES = ["OIII", "Hbeta", "Ha", "OII"]

FLUX_COLS = ["flux", "flux_err", "flux_snr"]
HA_COLS = ["z_single", "z_single_err", "snr_single"] + FLUX_COLS + ["fwhm_obs", "fwhm_obs_err", "fwhm_inst", "fwhm_int",
                       "fwhm_int_err", "unresolved", "NII6584_flux", "NII6584_flux_err"]
BASE_COLS = ["ID", "grating", "status", "reason", "z_dja", "z_used", "method",
             "z", "z_err", "snr", "snr_lime", "fwhm_pix", "centre_err_AA", "success",
             "line_margin", "cont_gap", "cont_width", "reviewed",
             "spectrum", "contsub_png", "fit_png"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=CATALOG)
    p.add_argument("--spectra-root", default=SPECTRA_ROOT)
    p.add_argument("--fig-root", default=FIG_ROOT)
    p.add_argument("--csv-dir", default=CSV_DIR)
    p.add_argument("--lines", nargs="+", default=LINES, choices=LINES)
    p.add_argument("--ids", nargs="+", type=int, default=None)
    p.add_argument("--gratings", nargs="+", default=GRATINGS, choices=GRATINGS)
    p.add_argument("--z", type=float, default=None)
    p.add_argument("--line-margin", type=float, default=LINE_MARGIN)
    p.add_argument("--cont-gap", type=float, default=CONT_GAP)
    p.add_argument("--cont-width", type=float, default=CONT_WIDTH)
    p.add_argument("--snr-min", type=float, default=SNR_MIN)
    p.add_argument("--centre-err-max", type=float, default=CENTRE_ERR_MAX_AA)
    p.add_argument("--min-fwhm-pix", type=float, default=MIN_FWHM_PIX)
    p.add_argument("--force-single", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.z is not None and (a.ids is None or len(a.ids) != 1):
        p.error("--z needs exactly one --ids")
    return a


def fits_csv(a, line):
    return os.path.abspath(os.path.join(a.csv_dir, f"lime_{line}_fits.csv"))


def summary_csv(a, line):
    return os.path.abspath(os.path.join(a.csv_dir, f"lime_{line}_summary.csv"))


def gf(g):
    return f"{g}_{GRATING_FILTER[g]}"


# ----------------------------------------------------------------------
# Spectrum and bands
# ----------------------------------------------------------------------
def load_spectrum(path, z):
    with fits.open(path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], float)
        flux = np.asarray(tab["FLUX"], float)
        err = np.asarray(tab["ERR"], float)
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave, flux, err = wave[good], flux[good], err[good]
    norm = np.nanmedian(np.abs(flux[flux > 0])) if np.any(flux > 0) else 1e-20
    spec = lime.Spectrum(input_wave=wave, input_flux=flux, input_err=err,
                         redshift=z, units_wave="AA", units_flux="FLAM",
                         norm_flux=norm)
    return spec, wave, flux, err


def band_df(label, lo_rest, hi_rest, centre, a):
    w3 = lo_rest - a.line_margin
    w4 = hi_rest + a.line_margin
    w2, w5 = w3 - a.cont_gap, w4 + a.cont_gap
    w1, w6 = w2 - a.cont_width, w5 + a.cont_width
    return pd.DataFrame({"wavelength": [centre], "w1": [w1], "w2": [w2], "w3": [w3],
                         "w4": [w4], "w5": [w5], "w6": [w6]}, index=[label])


def band_in_range(band_row, wave, zf):
    return (float(band_row["w1"]) * zf >= np.nanmin(wave)
            and float(band_row["w6"]) * zf <= np.nanmax(wave))


def ha_blend_band(a):
    band = band_df(L_HA_BLEND, NII_BLUE, NII_RED, HA, a)
    for lab, wl in ((L_NII_BLUE, NII_BLUE), (L_HA, HA), (L_NII_RED, NII_RED)):
        band.loc[lab, "wavelength"] = wl
        for w in ("w1", "w2", "w3", "w4", "w5", "w6"):
            band.loc[lab, w] = band.loc[L_HA_BLEND, w]
    band["units_wave"] = "Angstrom"
    return band


def ha_blend_cfg():
    amp_ratio = NII_FLUX_RATIO * NII_BLUE / NII_RED
    return {L_HA_BLEND: f"{L_NII_BLUE}+{L_HA}+{L_NII_RED}",
            f"{L_NII_RED}_kinem": L_HA,
            f"{L_NII_BLUE}_kinem": L_HA,
            f"{L_NII_BLUE}_amp": {"expr": f"{L_NII_RED}_amp/{amp_ratio:.6f}"}}


def frame_value(spec, label, col):
    if label not in spec.frame.index or col not in spec.frame.columns:
        return np.nan
    try:
        v = float(spec.frame.loc[label, col])
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def line_row(spec, label):
    """Centre, centre error and LiMe snr_line (profile flux S/N if absent)."""
    centre = frame_value(spec, label, "center")
    centre_err = frame_value(spec, label, "center_err")
    snr = frame_value(spec, label, "snr_line")
    if not np.isfinite(snr):
        f, fe = frame_value(spec, label, "profile_flux"), frame_value(spec, label, "profile_flux_err")
        snr = f / fe if np.isfinite(f) and np.isfinite(fe) and fe > 0 else np.nan
    return centre, centre_err, snr


def line_flux(spec, label):
    f, fe = frame_value(spec, label, "profile_flux"), frame_value(spec, label, "profile_flux_err")
    return f, fe, (f / fe if np.isfinite(f) and np.isfinite(fe) and fe > 0 else np.nan)


# ----------------------------------------------------------------------
# Instrumental resolution (msaexp curves, located on disk, not imported)
# ----------------------------------------------------------------------
_R_CURVES = {}
_R_WARNED = []


def instrumental_fwhm_kms(grating, wave_obs):
    if not np.isfinite(wave_obs):
        return np.nan
    key = grating.lower()
    if key not in _R_CURVES:
        spec = importlib.util.find_spec("msaexp")
        if spec is None or not spec.submodule_search_locations:
            if not _R_WARNED:
                print("    WARNING msaexp not installed, FWHM_inst and FWHM_int left blank")
                _R_WARNED.append(1)
            return np.nan
        path = os.path.join(list(spec.submodule_search_locations)[0], "data",
                            f"jwst_nirspec_{key}_disp.fits")
        tab = Table.read(path)
        _R_CURVES[key] = (np.asarray(tab["WAVELENGTH"], float) * 1e4, np.asarray(tab["R"], float))
    w, r = _R_CURVES[key]
    return C_KMS / (np.interp(wave_obs, w, r) * R_SCALE)


def intrinsic_fwhm(obs, obs_err, inst):
    if not (np.isfinite(obs) and np.isfinite(inst)):
        return np.nan, np.nan, False
    if obs <= inst:
        return np.nan, np.nan, True
    val = np.sqrt(obs ** 2 - inst ** 2)
    return val, (obs * obs_err / val if np.isfinite(obs_err) else np.nan), False


def gaussian(x, flux, centre, sigma):
    return flux / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def plot_fit(spec, wave, flux, err, band_row, comps, cont_label, markers, z_fit,
             title, png):
    """Data, fitted components, total model, continuum and residuals.

    band_row   the band (w1..w6, rest frame) used for the fit, placed at z_fit
    comps      [(LiMe label, legend name, colour)], the first is the main line
    cont_label LiMe label whose local continuum (m_cont, n_cont) is drawn
    markers    rest wavelengths marked with dash-dot lines at z_fit
    Linear y axis, scaled to the data in the band.
    """
    zf = 1.0 + z_fit
    lo, hi = float(band_row["w1"]) * zf, float(band_row["w6"]) * zf
    sel = (wave >= lo) & (wave <= hi)
    x, y, e = wave[sel], flux[sel], err[sel]
    if x.size < 3:
        return
    m, n = frame_value(spec, cont_label, "m_cont"), frame_value(spec, cont_label, "n_cont")
    xf = np.linspace(x.min(), x.max(), 2000)
    if np.isfinite(m) and np.isfinite(n):
        cont, cont_f = m * x + n, m * xf + n
    else:
        c0 = frame_value(spec, cont_label, "cont")
        c0 = c0 if np.isfinite(c0) else 0.0
        cont, cont_f = np.full_like(x, c0), np.full_like(xf, c0)
    total_f, total_d, curves = cont_f.copy(), cont.copy(), []
    for i, (lab, name, col) in enumerate(comps):
        f, c, s = (frame_value(spec, lab, k) for k in ("profile_flux", "center", "sigma"))
        if np.isfinite(f) and np.isfinite(c) and np.isfinite(s) and s > 0:
            total_f += gaussian(xf, f, c, s)
            total_d += gaussian(x, f, c, s)
            curves.append((name, col, gaussian(xf, f, c, s), i == 0))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.step(x, y, where="mid", color="k", lw=1, label="data", zorder=2)
    ax.fill_between(x, y - e, y + e, step="mid", color="0.8", lw=0, zorder=1)
    if len(curves) > 1:
        ax.plot(xf, total_f, color="tab:orange", lw=2.5, alpha=0.6, label="total model", zorder=3)
    ax.plot(xf, cont_f, color="0.4", ls=":", lw=1, label="continuum", zorder=4)
    for name, col, g, main in curves:
        ax.plot(xf, cont_f + g, color=col, ls="-" if main else "--", lw=1.2, label=name, zorder=5)
    for wl in markers:
        ax.axvline(wl * zf, color="0.7", lw=0.6, ls="-.", zorder=0)
    for a_, b_ in (("w1", "w2"), ("w5", "w6")):
        for axx in (ax, axr):
            axx.axvspan(float(band_row[a_]) * zf, float(band_row[b_]) * zf,
                        color="0.5", alpha=0.12, lw=0)
    # y range from the data and model, ignoring single wild pixels
    ylo = min(np.nanpercentile(y - e, 1), np.nanmin(cont_f))
    yhi = max(np.nanpercentile(y + e, 99.5), np.nanmax(total_f))
    pad = 0.08 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(r"$f_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.legend(fontsize=8, loc="upper right")
    res = (y - total_d) / e
    axr.step(x, res, where="mid", color="k", lw=1)
    axr.axhline(0, color="tab:orange", lw=1)
    # residual axis set by the bulk of pixels, so single spikes run off the panel
    r_lim = (float(np.clip(np.nanpercentile(np.abs(res), 95) * 1.6, 3.0, 25.0))
             if np.isfinite(res).any() else 3.0)
    axr.set_ylim(-r_lim, r_lim)
    axr.set_ylabel(r"resid / $\sigma$")
    axr.set_xlabel(r"observed wavelength [$\AA$]")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def covered(wave, centre_obs, half_width):
    """True if the spectrum has data across centre_obs +/- half_width.

    load_spectrum drops non-finite pixels, so a detector gap or the end of the
    spectrum shows up as missing pixels. Needs at least 80 per cent of the
    pixels expected from the local pixel spacing.
    """
    if not (np.nanmin(wave) <= centre_obs - half_width and centre_obs + half_width <= np.nanmax(wave)):
        return False
    near = np.abs(wave - centre_obs) <= 3 * half_width
    dl = np.median(np.diff(wave[near])) if near.sum() > 2 else np.median(np.diff(wave))
    n = np.sum(np.abs(wave - centre_obs) <= half_width)
    return n >= 0.8 * (2 * half_width / dl)


def amp_over_noise(spec, label, wave, flux, band_row, zf):
    """A / sigma_noise and the fitted FWHM in pixels for one line.

    A is the fitted Gaussian peak height above the continuum,
    profile_flux / (sqrt(2 pi) sigma). sigma_noise is the standard deviation
    of the flux in the two adjacent continuum bands (w1-w2, w5-w6) about the
    fitted linear continuum, estimated robustly from the median absolute
    deviation of each flank. These are the two quantities the LiMe paper
    (Fernandez et al. 2024, Sect. 5) uses for its detection boundary and its
    flux accuracy tests.
    """
    f = frame_value(spec, label, "profile_flux")
    sig = frame_value(spec, label, "sigma")
    m, n = frame_value(spec, label, "m_cont"), frame_value(spec, label, "n_cont")
    blue = (wave >= float(band_row["w1"]) * zf) & (wave <= float(band_row["w2"]) * zf)
    red = (wave >= float(band_row["w5"]) * zf) & (wave <= float(band_row["w6"]) * zf)
    if not (np.isfinite(f) and np.isfinite(sig) and sig > 0 and np.isfinite(m) and np.isfinite(n)
            and blue.sum() + red.sum() > 3):
        return np.nan, np.nan
    # Robust standard deviation (1.4826 x median absolute deviation) of each
    # flank about its own median after removing the fitted continuum slope,
    # averaged over the two flanks. Equal to the standard deviation for
    # Gaussian noise, but not inflated by a hot pixel or by the fitted
    # continuum sitting a little off one flank.
    mads = []
    for sel in (blue, red):
        if sel.sum() > 3:
            r = flux[sel] - (m * wave[sel] + n)
            mads.append(1.4826 * np.median(np.abs(r - np.median(r))))
    noise = float(np.mean(mads))
    amp = f / (np.sqrt(2 * np.pi) * sig)
    c = frame_value(spec, label, "center")
    near = np.abs(wave - c) <= 20 * sig
    pix = np.median(np.diff(wave[near])) if near.sum() > 2 else np.median(np.diff(wave))
    return amp / noise, GAUSS_FWHM * sig / pix


def z_from(spec, ref, rest, a, wave, flux, band_row, zf):
    """z, z_err, A/noise, centre_err, success, LiMe's snr_line, FWHM in pixels.

    A line is detected and its redshift used when
      A / sigma_noise >= --snr-min (5)   the LiMe detection boundary for lines
                                         about as wide as the resolution
      FWHM >= --min-fwhm-pix (1 pixel)   not a single-pixel spike
      centre error <= --centre-err-max (2.5 Angstrom)
    """
    centre, centre_err, snr_lime = line_row(spec, ref)
    z = centre / rest - 1.0
    z_err = centre_err / rest if np.isfinite(centre_err) else np.nan
    an, fwhm_pix = amp_over_noise(spec, ref, wave, flux, band_row, zf)
    ok = bool(np.isfinite(an) and an >= a.snr_min
              and np.isfinite(fwhm_pix) and fwhm_pix >= a.min_fwhm_pix
              and np.isfinite(centre_err) and centre_err <= a.centre_err_max)
    return z, z_err, an, centre_err, ok, snr_lime, fwhm_pix


# ----------------------------------------------------------------------
# One fit
# ----------------------------------------------------------------------
def fit_line(line, src_id, g, path, z, a):
    """Fit one line in one spectrum. Returns a dict of results.

    Every fit uses the flank continuum (LiMe "adjacent", a straight line
    fitted to w1-w2 and w5-w6 with their errors). LiMe's default ("central")
    draws the continuum through the first and last pixel of the line region,
    so two noisy pixels can set it. Detection uses z_from. The flux
    and width come from the same fit.

    A line counts as covered only if the spectrum has data at its centre, so
    a line in the detector gap is not fitted. For [OIII] the redshift needs
    5007: if 5007 is not covered the source has no [OIII] fit, and the blend
    is used only when 4959 is covered too.

    Halpha: z comes from the Halpha + [NII] blend whenever the blend band is
    on the spectrum (method blend_NII). The single-Gaussian fit is kept as
    z_single, z_single_err, snr_single, and gives z only when the blend band
    is off the spectrum (method single).
    """
    out = {"status": "fitted", "reason": "", "method": "", "z_used": z}
    spec, wave, flux, err = load_spectrum(path, z)
    zf = 1.0 + z
    hw = 0.5 * a.line_margin * zf  # half-width for the coverage check, observed AA
    blend_band = None

    if line == "OIII":
        if not covered(wave, OIII_5007 * zf, hw):
            return {"status": "out_of_range", "reason": "[OIII] 5007 not covered (edge or detector gap)"}
        doublet = band_df(L_O3_5007, OIII_4959, OIII_5007, OIII_5007, a)
        single = band_df(L_O3_5007, OIII_5007, OIII_5007, OIII_5007, a)
        if (band_in_range(doublet.iloc[0], wave, zf) and covered(wave, OIII_4959 * zf, hw)
                and not a.force_single):
            out["method"], band, label = "joint_4959_5007", doublet, L_O3_BLEND
            cfg = {L_O3_BLEND: f"{L_O3_4959}+{L_O3_5007}",
                   f"{L_O3_4959}_amp": {"expr": f"{L_O3_5007}_amp/{OIII_RATIO_THEORY}"},
                   f"{L_O3_4959}_kinem": L_O3_5007}
            comps = [(L_O3_5007, "[OIII] 5007", "tab:red"), (L_O3_4959, "[OIII] 4959", "tab:blue")]
        elif band_in_range(single.iloc[0], wave, zf):
            out["method"], band, label, cfg = "single_5007", single, L_O3_5007, None
            comps = [(L_O3_5007, "[OIII] 5007", "tab:red")]
        else:
            return {"status": "out_of_range"}
        ref, rest, markers = L_O3_5007, OIII_5007, (OIII_4959, OIII_5007)
    elif line == "Hbeta":
        band, label, cfg, ref, rest = band_df(L_HB, HB, HB, HB, a), L_HB, None, L_HB, HB
        out["method"], comps, markers = "single", [(L_HB, r"H$\beta$", "tab:red")], (HB,)
    elif line == "Ha":
        band, label, cfg, ref, rest = band_df(L_HA, HA, HA, HA, a), L_HA, None, L_HA, HA
        out["method"], comps, markers = "single", [(L_HA, r"H$\alpha$", "tab:red")], (HA,)
        hb = ha_blend_band(a)
        if band_in_range(hb.loc[L_HA_BLEND], wave, zf):
            blend_band = hb
    elif line == "OII":
        band = band_df(L_O2_BLEND, OII_3726, OII_3729, OII_3729, a)
        label, ref, rest = L_O2_BLEND, L_O2_3729, OII_3729
        cfg = {L_O2_BLEND: f"{L_O2_3726}+{L_O2_3729}", f"{L_O2_3726}_kinem": L_O2_3729}
        out["method"] = "blend_3726_3729"
        comps = [(L_O2_3729, "[OII] 3729", "tab:red"), (L_O2_3726, "[OII] 3726", "tab:blue")]
        markers = (OII_3726, OII_3729)
    if not band_in_range(band.iloc[0], wave, zf):
        return {"status": "out_of_range"}
    if not covered(wave, rest * zf, hw):
        return {"status": "out_of_range", "reason": f"{line} not covered (edge or detector gap)"}

    fdir = os.path.join(a.fig_root, str(src_id), f"{line}_fits")
    os.makedirs(fdir, exist_ok=True)
    stem = f"{src_id}_{gf(g)}_{line}"
    out["contsub_png"] = os.path.abspath(os.path.join(fdir, f"{stem}_contsub.png"))
    out["fit_png"] = os.path.abspath(os.path.join(fdir, f"{stem}_fit.png"))
    title_id = f"ID {src_id}  {gf(g)}  {line}"

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=out["contsub_png"])
    kw = {"bands": band, "cont_source": CONT_SOURCE}
    if cfg is not None:
        kw["fit_cfg"] = cfg
    spec.fit.bands(label, **kw)
    zz = z_from(spec, ref, rest, a, wave, flux, band.iloc[0], zf)

    if line == "Ha":
        out["z_single"], out["z_single_err"], out["snr_single"] = zz[0], zz[1], zz[2]
        if blend_band is not None:
            try:
                spec.fit.bands(L_HA_BLEND, bands=blend_band, fit_cfg=ha_blend_cfg(),
                               cont_source=CONT_SOURCE)
                band = blend_band.loc[[L_HA_BLEND]]
                zz = z_from(spec, L_HA, HA, a, wave, flux, band.iloc[0], zf)
                out["method"] = "blend_NII"
                comps = [(L_HA, r"H$\alpha$", "tab:red"), (L_NII_BLUE, "[NII] 6548", "tab:blue"),
                         (L_NII_RED, "[NII] 6584", "tab:blue")]
                markers = (NII_BLUE, HA, NII_RED)
            except Exception as e:
                out["reason"] = f"Ha blend failed, z from single fit: {e}"
                print(f"    Ha blend FAILED {e}")
        else:
            out["reason"] = "Ha+[NII] blend out of range, z from single fit, no [NII]"

    (out["z"], out["z_err"], out["snr"], out["centre_err_AA"], out["success"],
     out["snr_lime"], out["fwhm_pix"]) = zz
    if line == "Ha" and out["method"] == "blend_NII":
        dv = C_KMS * (out["z"] - out["z_single"]) / (1 + out["z_single"])
        print(f"    Ha    single           z {out['z_single']:.6f}  A/noise {out['snr_single']:.2f}   "
              f"blend - single {dv:+.0f} km/s")
    print(f"    {line:<5} {out['method']:<16} z {out['z']:.6f} +/- {out['z_err']:.6f}  "
          f"A/noise {out['snr']:.2f}  FWHM {out['fwhm_pix']:.1f} pix  centre_err {out['centre_err_AA']:.3g} AA  "
          f"success {out['success']}")

    if line in ("Hbeta", "Ha"):
        out["flux"], out["flux_err"], out["flux_snr"] = line_flux(spec, ref if line == "Hbeta" else L_HA)
    if line == "Ha" and out["method"] == "blend_NII":
        out["NII6584_flux"], out["NII6584_flux_err"], _ = line_flux(spec, L_NII_RED)
    if line == "Ha":
        sv, sve = frame_value(spec, L_HA, "sigma_vel"), frame_value(spec, L_HA, "sigma_vel_err")
        out["fwhm_obs"], out["fwhm_obs_err"] = GAUSS_FWHM * sv, GAUSS_FWHM * sve
        out["fwhm_inst"] = instrumental_fwhm_kms(g, frame_value(spec, L_HA, "center"))
        out["fwhm_int"], out["fwhm_int_err"], out["unresolved"] = intrinsic_fwhm(
            out["fwhm_obs"], out["fwhm_obs_err"], out["fwhm_inst"])

    z_plot = out["z"] if np.isfinite(out["z"]) else z
    title = (f"{title_id}  {out['method']}\n"
             f"z {out['z']:.5f} ± {out['z_err']:.5f}   A/noise {out['snr']:.1f}   "
             f"centre err {out['centre_err_AA']:.2f} Å   success {out['success']}")
    if line in ("Hbeta", "Ha"):
        title += f"\n{line} flux {out['flux']:.3e} ± {out['flux_err']:.2e} (S/N {out['flux_snr']:.1f})"
        if line == "Ha" and out["method"] == "blend_NII":
            title += f"   [NII]6584 {out['NII6584_flux']:.3e} ± {out['NII6584_flux_err']:.2e}"
        print(f"    {line} flux {out['flux']:.3e} +/- {out['flux_err']:.2e} (S/N {out['flux_snr']:.1f})")
    try:
        plot_fit(spec, wave, flux, err, band.iloc[0], comps, ref if line != "Ha" else L_HA,
                 markers, z_plot, title, out["fit_png"])
    except Exception as e:
        print(f"    plot failed: {e}")
    print(f"    figure   {out['fit_png']}")
    return out


# ----------------------------------------------------------------------
# Summary per line
# ----------------------------------------------------------------------
def summarise(df, line):
    rows = []
    for sid, grp in df.groupby("ID"):
        r = {"ID": int(sid),
             "gratings_fitted": ",".join(grp.loc[grp["status"] == "fitted", "grating"]),
             "n_success": int(grp["success"].fillna(False).astype(bool).sum())}
        ok = grp[grp["success"].fillna(False).astype(bool)]
        if len(ok):
            b = ok.loc[ok["snr"].idxmax()]
            r.update({f"z_{line}": b["z"], f"z_{line}_err": b["z_err"],
                      f"z_{line}_snr": b["snr"], f"z_{line}_grating": b["grating"]})
        else:
            r.update({f"z_{line}": np.nan, f"z_{line}_err": np.nan,
                      f"z_{line}_snr": np.nan, f"z_{line}_grating": ""})
        if line in ("Hbeta", "Ha") and "flux_snr" in grp:
            fl = grp[np.isfinite(pd.to_numeric(grp["flux_snr"], errors="coerce"))]
            if len(fl):
                b = fl.loc[fl["flux_snr"].astype(float).idxmax()]
                r.update({f"{line}_flux_best": b["flux"], f"{line}_flux_best_err": b["flux_err"],
                          f"{line}_flux_best_snr": b["flux_snr"],
                          f"{line}_flux_best_grating": b["grating"]})
        if line == "Ha" and "fwhm_obs" in grp:
            c = grp[(pd.to_numeric(grp["flux_snr"], errors="coerce") >= FWHM_SNR_MIN)
                    & np.isfinite(pd.to_numeric(grp["fwhm_obs"], errors="coerce"))]
            if len(c):
                c = c.assign(_h=c["grating"].str.endswith("H"))
                b = c.sort_values(["_h", "flux_snr"], ascending=False).iloc[0]
                for k in ("fwhm_obs", "fwhm_obs_err", "fwhm_inst", "fwhm_int", "fwhm_int_err"):
                    r[f"Ha_{k}_best"] = b[k]
                r["Ha_fwhm_unresolved_best"] = b["unresolved"]
                r["Ha_fwhm_best_grating"] = b["grating"]
        rows.append(r)
    return pd.DataFrame(rows)


def write_line_csvs(a, line, new):
    path, spath = fits_csv(a, line), summary_csv(a, line)
    cols = BASE_COLS + (HA_COLS if line == "Ha" else FLUX_COLS if line == "Hbeta" else [])
    new = new.reindex(columns=cols)
    if a.ids is not None and os.path.exists(path):
        old = pd.read_csv(path)
        key_new = set(zip(new["ID"], new["grating"]))
        keep = [(i, g) not in key_new for i, g in zip(old["ID"], old["grating"])]
        df = pd.concat([old[keep], new], ignore_index=True)
    else:
        df = new
    df = df.sort_values(["ID", "grating"]).reset_index(drop=True)
    summ = summarise(df, line)
    if a.dry_run:
        return df, summ
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    summ.to_csv(spath, index=False)
    print(f"  written  {path}")
    print(f"  written  {spath}")
    return df, summ


# ----------------------------------------------------------------------
def main():
    a = parse_args()
    print("lime_fit_lines.py")
    print(f"  catalogue      {os.path.abspath(a.catalog)}")
    print(f"  spectra        {os.path.abspath(a.spectra_root)}")
    print(f"  figures        {os.path.abspath(a.fig_root)}/<ID>/<line>_fits/")
    print(f"  csv dir        {os.path.abspath(a.csv_dir)}")
    print(f"  lines          {', '.join(a.lines)}")
    print(f"  gratings       {', '.join(a.gratings)}")
    print(f"  success        A/noise >= {a.snr_min:g}, FWHM >= {a.min_fwhm_pix:g} pix, "
          f"centre_err <= {a.centre_err_max:g} AA")
    print(f"  band           margin {a.line_margin:g}, gap {a.cont_gap:g}, width {a.cont_width:g} AA")
    if a.z is not None:
        print(f"  z override     {a.z}")
    if a.force_single:
        print("  OIII           single 5007 forced")
    if a.dry_run:
        print("  DRY RUN, no CSVs written")

    cat = pd.read_csv(a.catalog)
    if a.ids is not None:
        missing = sorted(set(a.ids) - set(cat["ID"]))
        if missing:
            print(f"  IDs not in the catalogue: {missing}")
        cat = cat[cat["ID"].isin(a.ids)]
    print(f"  {len(cat)} sources")

    results = {line: [] for line in a.lines}
    for _, row in cat.iterrows():
        sid = int(row["ID"])
        for g in a.gratings:
            if int(row.get(g, 0)) != 1:
                continue
            path = os.path.abspath(os.path.join(a.spectra_root, gf(g),
                                                f"{sid}_{gf(g)}_spectra_lime.fits"))
            z_dja = float(row[f"z_{g}"])
            z = a.z if a.z is not None else z_dja
            print(f"\n--- ID {sid}  {gf(g)}  DJA z {z_dja:.5f}"
                  + (f"  using z {z:.5f}" if a.z is not None else "") + " ---")
            print(f"    spectrum {path}")
            for line in a.lines:
                base = {"ID": sid, "grating": g, "z_dja": z_dja, "spectrum": path,
                        "line_margin": a.line_margin, "cont_gap": a.cont_gap,
                        "cont_width": a.cont_width, "reviewed": a.ids is not None}
                if not os.path.exists(path):
                    print(f"    {line:<5} no LiMe spectrum")
                    results[line].append({**base, "status": "failed",
                                          "reason": "no _spectra_lime.fits"})
                    continue
                try:
                    r = fit_line(line, sid, g, path, z, a)
                except Exception as e:
                    print(f"    {line:<5} FAILED {e}")
                    r = {"status": "failed", "reason": str(e)}
                if r["status"] == "out_of_range":
                    print(f"    {line:<5} out of range")
                results[line].append({**base, **r})

    print("\n" + "=" * 60)
    for line in a.lines:
        new = pd.DataFrame(results[line])
        if new.empty:
            print(f"{line}: nothing fitted")
            continue
        df, summ = write_line_csvs(a, line, new)
        st = new["status"].value_counts().to_dict()
        print(f"{line}: this run {st}, successful fits {int(new['success'].fillna(False).astype(bool).sum())}")
        print(f"  sources with a successful {line} fit (whole file): "
              f"{int(summ[f'z_{line}'].notna().sum())} of {len(summ)}")
    print(f"figures under {os.path.abspath(a.fig_root)}")
    print("=" * 60)


if __name__ == "__main__":
    main()