#!/usr/bin/env python
"""Fit Hbeta, Halpha, [NII] 6584, and the [OII] 3726,3729 doublet across all
four gratings and build per-line result CSVs.

Companion to lime_OIII_jointfit.py. Same loader, same band geometry, same
success criteria, generalised over a LINES config list instead of hardcoding
[OIII]. Written for LiMe 2.4.3.

Per source, per grating, per line
----------------------------------
1. Work out whether the line (or, for OII, the padded doublet band) falls
   inside that spectrum's observed wavelength range at the DJA redshift. If
   not, skip that line for that grating.
2. Fit a local continuum, then fit the line: a single Gaussian for Hbeta, Ha
   and NII, or a kinematically-tied doublet blend for OII. This is the
   redshift fit and it is unchanged from the original version of the script,
   so the systemic redshifts and success flags are reproduced exactly.
3. Unlike [OIII] 4959/5007, the OII 3726/3729 flux ratio is NOT fixed by
   atomic physics (it is density-dependent, roughly 0.35-1.5), so only the
   kinematics are tied between the two OII components. Both amplitudes are
   left free.

Flux and width measurements (added October 2026)
-------------------------------------------------
Hbeta. The flux and its error are read from the same single-Gaussian fit
used for the redshift. No dust or slit-loss correction is applied here.

Halpha. After the redshift fit, a second fit is run on the same spectrum.
Halpha is fitted as a blend with [NII] 6548 and [NII] 6584:
    - both [NII] lines share the Halpha velocity and velocity width
      (kinematics tied to Halpha),
    - the [NII] 6584/6548 flux ratio is fixed at NII_FLUX_RATIO.
The Halpha flux and FWHM come from this blended fit, so [NII] is removed
from Halpha when the lines overlap. The redshift still comes from step 2.

FWHM columns, all in km/s:
    FWHM_obs   2 sqrt(2 ln 2) * sigma_vel from the fitted Gaussian
    FWHM_inst  c / (R * R_SCALE) at the observed Halpha wavelength, where R
               is the nominal NIRSpec resolution curve shipped with msaexp
               (jwst_nirspec_<grating>_disp.fits) and R_SCALE = 1.3 is the
               compact-source factor msaexp uses by default (scale_disp),
               after de Graaff et al. (2024)
    FWHM_int   sqrt(FWHM_obs^2 - FWHM_inst^2). NaN, with the unresolved flag
               set, when FWHM_obs <= FWHM_inst.

A NOTE ON LINE LABELS AND WAVELENGTHS
--------------------------------------
Labels follow the O3_5007A convention already used for [OIII]: element +
ionisation-stage digit, underscore, nearest integer vacuum wavelength, "A".
    Hbeta   -> H1_4861A   (vacuum 4862.683 AA)
    Ha      -> H1_6563A   (vacuum 6564.632 AA)
    NII     -> N2_6548A   (vacuum 6549.860 AA), N2_6584A (vacuum 6585.270 AA)
    OII     -> O2_3726A, O2_3729A, blend O2_3729A_b
LiMe only uses wavelengths from a user bands table if that table also has a
units_wave column. Otherwise it falls back to its own database (air
wavelengths) and then to parsing the label. N2_6584A is not in the LiMe
database, so for the Halpha blend every component is listed in the bands
table with an explicit vacuum wavelength and units_wave. This matters there
because the kinematic tie fixes the [NII] centres relative to Halpha using
these wavelengths. The redshift fits keep the original bands table, so
their behaviour is unchanged, and z is always computed from the fitted
observed centre and the vacuum rest wavelength.

Outputs, all written fresh (old files overwritten), one pair per line
-----------------------------------------------------------------------
  <name>_results_by_JELS_ID.csv   one row per source. Original columns
                                   (per-grating DJA z, z_<name>_<gr>, its
                                   error and S/N) first, then the new
                                   per-grating measurement columns.
  <name>_summary_by_JELS_ID.csv   one row per source. Original columns
                                   (gratings, per-grating success flags,
                                   <name>_success count) first, then the
                                   per-grating measurements and a best-value
                                   selection for Ha and Hbeta.
  Figures under JWST_SPECTRA_ROOT/<name>_fits/<ID>/, as before, plus
  <ID>_<grating>_Ha_blend_fit.png showing the data, the Halpha and [NII]
  components, the total model and the residuals.

New columns
-----------
Hbeta results and summary, per grating <gr>:
  Hbeta_<gr>_flux, Hbeta_<gr>_flux_err, Hbeta_<gr>_flux_snr
Hbeta summary, best value (highest flux S/N):
  Hbeta_flux_best, Hbeta_flux_best_err, Hbeta_flux_best_snr,
  Hbeta_flux_best_grating
Ha results and summary, per grating <gr>:
  Ha_<gr>_flux, Ha_<gr>_flux_err, Ha_<gr>_flux_snr,
  Ha_<gr>_fwhm_obs, Ha_<gr>_fwhm_obs_err, Ha_<gr>_fwhm_inst,
  Ha_<gr>_fwhm_int, Ha_<gr>_fwhm_int_err, Ha_<gr>_unresolved,
  Ha_<gr>_NII6584_flux, Ha_<gr>_NII6584_flux_err
Ha summary, best values:
  Ha_flux_best, Ha_flux_best_err, Ha_flux_best_snr, Ha_flux_best_grating
      highest Halpha flux S/N across gratings
  Ha_fwhm_obs_best, Ha_fwhm_obs_best_err, Ha_fwhm_inst_best,
  Ha_fwhm_int_best, Ha_fwhm_int_best_err, Ha_fwhm_unresolved_best,
  Ha_fwhm_best_grating
      among gratings with Halpha flux S/N >= FWHM_SNR_MIN, an H grating is
      preferred over an M grating, then the highest S/N wins
Fluxes are in erg/s/cm2, observed (no slit-loss or dust correction).

Success. A fit is successful when BOTH the detection S/N clears SNR_MIN and
the fitted centre error is finite and below CENTRE_ERR_MAX_AA. Both
defaults match the [OIII] script. This is the redshift success flag only.
Fluxes and widths are stored for every fitted spectrum, whatever the flag.

Requirements
------------
LiMe 2.4.3, and msaexp installed in the same environment for the
resolution curves (pip install msaexp). msaexp is located on disk only,
never imported, so its own dependencies are not needed here.

Usage
-----
python lime_lines_jointfit.py                 # walks the default grating dirs
python lime_lines_jointfit.py <parent_dir>    # override the spectra root

Filename convention expected: <ID>_<grating>_spectra_lime.fits
"""

import os
import sys
import re
import glob
import importlib.util

import numpy as np
import pandas as pd
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lime


# ----------------------------------------------------------------------
# Fixed configuration. Paths match ways-of-working.md / lime_OIII_jointfit.py.
# ----------------------------------------------------------------------
CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"

# The four gratings and their catalogue files.
GRATINGS = ["G235M_F170LP", "G235H_F170LP", "G395M_F290LP", "G395H_F290LP"]

# DJA catalogue column names.
DJA_ID_COL = "ID"
DJA_Z_COL = "z"

# --- Band geometry, rest-frame Angstrom. Same as lime_OIII_jointfit.py. ---
LINE_MARGIN = 15.0     # w3-w4 extends this far beyond each outer line
CONT_GAP = 5.0          # gap between line region and each continuum flank
CONT_WIDTH = 20.0       # width of each continuum flank
# ---------------------------------------------------------------------

# --- Success criteria. Both must pass. Same defaults as [OIII]. ---
SNR_MIN = 13.0
CENTRE_ERR_MAX_AA = 2.5
# -------------------------------------------------------------------

# --- Halpha + [NII] blend and FWHM settings. ---
C_KMS = 299792.458
GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # FWHM / sigma
NII_FLUX_RATIO = 3.05     # [NII] 6584 / 6548 flux ratio, fixed by atomic physics
R_SCALE = 1.3             # compact-source resolution factor, msaexp default
FWHM_SNR_MIN = 5.0        # min Halpha flux S/N for a grating to supply the best FWHM
# Local continuum for the flux fits only. LiMe's default ("central") draws a
# straight line through the first and last pixel of the line window, so two
# noisy pixels set the continuum. "adjacent" fits a line to both continuum
# flanks with their errors. In tests on synthetic G395M spectra this halved
# the Halpha flux scatter and gave errors that match it. The redshift fits
# keep LiMe's default so z is unchanged.
FLUX_CONT_SOURCE = "adjacent"
# ---------------------------------------------

# ----------------------------------------------------------------------
# Line definitions. Vacuum rest wavelengths, Angstrom.
# kind "single" fits one Gaussian. kind "doublet" fits a kinematically
# tied blend. amp_ratio_theory is only set where the ratio is fixed by
# atomic physics; leave None to fit both amplitudes free (OII).
# store_flux saves the flux of the redshift fit (Hbeta).
# ha_blend runs the extra Halpha + [NII] blend fit (Ha).
# ----------------------------------------------------------------------
LINES = [
    {
        "name": "Hbeta",
        "kind": "single",
        "label": "H1_4861A",
        "rest_vac": 4862.683,
        "store_flux": True,
    },
    {
        "name": "Ha",
        "kind": "single",
        "label": "H1_6563A",
        "rest_vac": 6564.632,
        "ha_blend": True,
    },
    {
        "name": "NII",
        "kind": "single",
        "label": "N2_6584A",
        "rest_vac": 6585.270,
    },
    {
        "name": "OII",
        "kind": "doublet",
        "label_blend": "O2_3729A_b",
        "label_ref": "O2_3729A",      # redder component, used for z
        "label_other": "O2_3726A",
        "rest_vac_ref": 3729.875,
        "rest_vac_other": 3727.092,
        "amp_ratio_theory": None,      # density-dependent, not fixed
    },
]

# Components of the Halpha + [NII] blend. Vacuum rest wavelengths, Angstrom.
HA_LABEL = "H1_6563A"
NII_RED_LABEL = "N2_6584A"
NII_BLUE_LABEL = "N2_6548A"
HA_BLEND_LABEL = "H1_6563A_b"
HA_REST_VAC = 6564.632
NII_RED_REST_VAC = 6585.270
NII_BLUE_REST_VAC = 6549.860
# ----------------------------------------------------------------------


def results_csv(line):
    return os.path.join(CATALOG_DIR, f"{line['name']}_results_by_JELS_ID.csv")


def summary_csv(line):
    return os.path.join(CATALOG_DIR, f"{line['name']}_summary_by_JELS_ID.csv")


def fig_root(line):
    return os.path.join(JWST_SPECTRA_ROOT, f"{line['name']}_fits")


def grating_short(grating):
    """G235M_F170LP -> G235M, for column and filename use."""
    return grating.split("_")[0]


def catalogue_path(grating):
    return os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{grating}_match_0p3as.fits")


def parse_id_grating(lime_path):
    """Pull the integer ID and grating string from the filename."""
    name = os.path.basename(lime_path)
    m = re.match(r"^(\d+)_([A-Z0-9]+_[A-Z0-9]+)_spectra_lime\.fits$", name)
    if m is None:
        raise ValueError(f"Cannot parse ID and grating from filename: {name}")
    src_id = int(m.group(1))
    grating = m.group(2)
    if grating not in GRATINGS:
        raise ValueError(f"Grating {grating} not one of {GRATINGS}")
    return src_id, grating


def get_dja_redshift(cat, src_id):
    """DJA redshift for this ID from an already-loaded catalogue table."""
    mask = cat[DJA_ID_COL] == src_id
    if mask.sum() == 0:
        return None
    return float(cat[DJA_Z_COL][mask][0])


def load_lime_spectrum(lime_path, redshift):
    """Load the _lime.fits SPECTRUM table into a lime.Spectrum.

    Also returns the cleaned wave, flux and err arrays (Angstrom, FLAM) for
    the blend plot.
    """
    from astropy.io import fits

    with fits.open(lime_path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave, flux, err = wave[good], flux[good], err[good]

    norm_flux = np.nanmedian(np.abs(flux[flux > 0])) if np.any(flux > 0) else 1e-20

    spec = lime.Spectrum(
        input_wave=wave,
        input_flux=flux,
        input_err=err,
        redshift=redshift,
        units_wave="AA",
        units_flux="FLAM",
        norm_flux=norm_flux,
    )
    return spec, wave, flux, err


def single_band_df(label, rest_vac, line_margin=LINE_MARGIN,
                    cont_gap=CONT_GAP, cont_width=CONT_WIDTH):
    w3 = rest_vac - line_margin
    w4 = rest_vac + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [rest_vac],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[label],
    )


def doublet_band_df(label, rest_vac_lo, rest_vac_hi, line_margin=LINE_MARGIN,
                     cont_gap=CONT_GAP, cont_width=CONT_WIDTH):
    """One-row bands frame whose line region spans both doublet components."""
    w3 = rest_vac_lo - line_margin
    w4 = rest_vac_hi + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [rest_vac_hi],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[label],
    )


def ha_blend_band_df():
    """Bands frame for the Halpha + [NII] blend.

    The blend row carries the band limits, spanning [NII] 6548 to
    [NII] 6584. One extra row per component gives LiMe the vacuum
    wavelength to use for that component. units_wave must be present or
    LiMe ignores these wavelengths (see module docstring).
    """
    band = doublet_band_df(HA_BLEND_LABEL, NII_BLUE_REST_VAC, NII_RED_REST_VAC)
    band.loc[HA_BLEND_LABEL, "wavelength"] = HA_REST_VAC
    for lab, wl in ((NII_BLUE_LABEL, NII_BLUE_REST_VAC),
                    (HA_LABEL, HA_REST_VAC),
                    (NII_RED_LABEL, NII_RED_REST_VAC)):
        band.loc[lab, "wavelength"] = wl
        for w in ("w1", "w2", "w3", "w4", "w5", "w6"):
            band.loc[lab, w] = band.loc[HA_BLEND_LABEL, w]
    band["units_wave"] = "Angstrom"
    return band


def ha_blend_fit_cfg():
    """fit_cfg for the Halpha + [NII] blend.

    Both [NII] lines take their velocity and velocity width from Halpha.
    With kinematics tied in velocity, sigma in Angstrom scales with
    wavelength, so the 6584/6548 amplitude ratio is the flux ratio
    times lambda_6548 / lambda_6584.
    """
    amp_ratio = NII_FLUX_RATIO * NII_BLUE_REST_VAC / NII_RED_REST_VAC
    return {
        HA_BLEND_LABEL: f"{NII_BLUE_LABEL}+{HA_LABEL}+{NII_RED_LABEL}",
        f"{NII_RED_LABEL}_kinem": HA_LABEL,
        f"{NII_BLUE_LABEL}_kinem": HA_LABEL,
        f"{NII_BLUE_LABEL}_amp": {"expr": f"{NII_RED_LABEL}_amp/{amp_ratio:.6f}"},
    }


def band_in_range(band_row, wave, zf):
    lo = float(band_row["w1"]) * zf
    hi = float(band_row["w6"]) * zf
    return (lo >= np.nanmin(wave)) and (hi <= np.nanmax(wave))


def centre_to_redshift(centre_obs_aa, rest_vac):
    return centre_obs_aa / rest_vac - 1.0


def centre_err_to_redshift_err(centre_err_aa, rest_vac):
    if centre_err_aa is None or not np.isfinite(centre_err_aa):
        return np.nan
    return centre_err_aa / rest_vac


def build_doublet_fit_cfg(line):
    """fit_cfg for the doublet blend. Kinematics tied, amplitude free
    unless amp_ratio_theory is set."""
    cfg = {
        line["label_blend"]: f"{line['label_other']}+{line['label_ref']}",
        f"{line['label_other']}_kinem": line["label_ref"],
    }
    if line.get("amp_ratio_theory") is not None:
        cfg[f"{line['label_other']}_amp"] = {
            "expr": f"{line['label_ref']}_amp/{line['amp_ratio_theory']}"
        }
    return cfg


def get_line_row(spec, label):
    """Return centre, centre_err, and detection S/N for one line label.

    S/N is LiMe's own snr_line (amplitude-based, Rola et al. definition).
    Falls back to profile_flux / its error only if snr_line is absent.
    """
    if label not in spec.frame.index:
        return np.nan, np.nan, np.nan
    row = spec.frame.loc[label]
    cols = spec.frame.columns

    centre = float(row["center"]) if "center" in cols else np.nan
    centre_err = (
        float(row["center_err"])
        if "center_err" in cols and np.isfinite(row["center_err"]) else np.nan
    )

    snr = np.nan
    if "snr_line" in cols and np.isfinite(row["snr_line"]):
        snr = float(row["snr_line"])
    elif "profile_flux" in cols and "profile_flux_err" in cols:
        f, fe = row["profile_flux"], row["profile_flux_err"]
        if np.isfinite(f) and np.isfinite(fe) and fe > 0:
            snr = float(f / fe)
    return centre, centre_err, snr


def frame_value(spec, label, col):
    """One float from the LiMe frame, NaN if missing or not finite."""
    if label not in spec.frame.index or col not in spec.frame.columns:
        return np.nan
    try:
        v = float(spec.frame.loc[label, col])
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def get_line_flux(spec, label):
    """Gaussian profile flux, its error and flux S/N (erg/s/cm2).

    LiMe de-normalises profile_flux by norm_flux when writing the frame,
    so this is in the input FLAM x Angstrom units.
    """
    f = frame_value(spec, label, "profile_flux")
    fe = frame_value(spec, label, "profile_flux_err")
    snr = f / fe if (np.isfinite(f) and np.isfinite(fe) and fe > 0) else np.nan
    return f, fe, snr


def is_success(snr, centre_err_aa):
    snr_ok = np.isfinite(snr) and snr >= SNR_MIN
    err_ok = np.isfinite(centre_err_aa) and centre_err_aa <= CENTRE_ERR_MAX_AA
    return bool(snr_ok and err_ok)


# ----------------------------------------------------------------------
# Instrumental resolution
# ----------------------------------------------------------------------
_R_CURVES = {}


def msaexp_data_dir():
    """Locate msaexp/data on disk without importing msaexp."""
    found = importlib.util.find_spec("msaexp")
    if found is None or not found.submodule_search_locations:
        raise ImportError(
            "msaexp not found. Install it in this environment with "
            "'pip install msaexp' (needed only for the NIRSpec resolution curves)."
        )
    return os.path.join(list(found.submodule_search_locations)[0], "data")


def resolution_curve(grating):
    """Wavelength [Angstrom] and nominal R for one grating, cached."""
    key = grating_short(grating).lower()
    if key not in _R_CURVES:
        path = os.path.join(msaexp_data_dir(), f"jwst_nirspec_{key}_disp.fits")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Resolution curve not found: {path}")
        tab = Table.read(path)
        wave_aa = np.asarray(tab["WAVELENGTH"], dtype=float) * 1e4   # micron -> AA
        _R_CURVES[key] = (wave_aa, np.asarray(tab["R"], dtype=float))
    return _R_CURVES[key]


def instrumental_fwhm_kms(grating, wave_obs_aa):
    """Instrumental FWHM in km/s at an observed wavelength."""
    if not np.isfinite(wave_obs_aa):
        return np.nan
    wave_aa, r_nom = resolution_curve(grating)
    r_eff = np.interp(wave_obs_aa, wave_aa, r_nom) * R_SCALE
    return C_KMS / r_eff


def intrinsic_fwhm(fwhm_obs, fwhm_obs_err, fwhm_inst):
    """Quadrature-subtracted FWHM, its error, and an unresolved flag."""
    if not (np.isfinite(fwhm_obs) and np.isfinite(fwhm_inst)):
        return np.nan, np.nan, False
    if fwhm_obs <= fwhm_inst:
        return np.nan, np.nan, True
    fwhm_int = np.sqrt(fwhm_obs ** 2 - fwhm_inst ** 2)
    err = fwhm_obs * fwhm_obs_err / fwhm_int if np.isfinite(fwhm_obs_err) else np.nan
    return fwhm_int, err, False


# ----------------------------------------------------------------------
# Halpha + [NII] blend
# ----------------------------------------------------------------------
def gaussian(x, flux, centre, sigma):
    return flux / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def plot_ha_blend(spec, wave, flux, err, zf, band, src_id, grating, png):
    """Data, Halpha and [NII] components, total model and residuals."""
    row = band.loc[HA_BLEND_LABEL]
    lo, hi = float(row["w1"]) * zf, float(row["w6"]) * zf
    sel = (wave >= lo) & (wave <= hi)
    x, y, e = wave[sel], flux[sel], err[sel]

    # Local linear continuum as fitted by LiMe (de-normalised in the frame).
    m = frame_value(spec, HA_LABEL, "m_cont")
    n = frame_value(spec, HA_LABEL, "n_cont")
    if np.isfinite(m) and np.isfinite(n):
        cont = m * x + n
    else:
        cont = np.full_like(x, frame_value(spec, HA_LABEL, "cont"))

    xf = np.linspace(x.min(), x.max(), 2000)
    cont_f = (m * xf + n) if (np.isfinite(m) and np.isfinite(n)) else np.full_like(xf, cont[0])

    comps = [
        (HA_LABEL, r"H$\alpha$", "tab:red"),
        (NII_BLUE_LABEL, r"[NII] 6548", "tab:blue"),
        (NII_RED_LABEL, r"[NII] 6584", "tab:blue"),
    ]
    total_f = cont_f.copy()
    total_d = cont.copy()
    comp_curves = []
    for lab, name, col in comps:
        f = frame_value(spec, lab, "profile_flux")
        c = frame_value(spec, lab, "center")
        s = frame_value(spec, lab, "sigma")
        if np.isfinite(f) and np.isfinite(c) and np.isfinite(s) and s > 0:
            g_f = gaussian(xf, f, c, s)
            total_f += g_f
            total_d += gaussian(x, f, c, s)
            comp_curves.append((name, col, g_f, lab))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.step(x, y, where="mid", color="k", lw=1, label="data", zorder=2)
    ax.fill_between(x, y - e, y + e, step="mid", color="0.8", lw=0, zorder=1)
    ax.plot(xf, total_f, color="tab:orange", lw=2.5, alpha=0.6,
            label="total model", zorder=3)
    ax.plot(xf, cont_f, color="0.4", ls=":", lw=1, label="continuum", zorder=4)
    for name, col, g_f, lab in comp_curves:
        ls = "-" if lab == HA_LABEL else "--"
        ax.plot(xf, cont_f + g_f, color=col, ls=ls, lw=1.2, label=name, zorder=5)
    for wl in (NII_BLUE_REST_VAC, HA_REST_VAC, NII_RED_REST_VAC):
        ax.axvline(wl * zf, color="0.7", lw=0.6, ls="-.", zorder=0)
    # continuum flanks used for the local continuum fit
    for a, b in (("w1", "w2"), ("w5", "w6")):
        ax.axvspan(float(row[a]) * zf, float(row[b]) * zf, color="0.5", alpha=0.12, lw=0)
        axr.axvspan(float(row[a]) * zf, float(row[b]) * zf, color="0.5", alpha=0.12, lw=0)

    fha, fha_e, snr_ha = get_line_flux(spec, HA_LABEL)
    fn2, fn2_e, snr_n2 = get_line_flux(spec, NII_RED_LABEL)
    ax.set_title(
        f"ID {src_id}  {grating}\n"
        f"Ha flux {fha:.3e} +/- {fha_e:.2e} (S/N {snr_ha:.1f})   "
        f"[NII]6584 {fn2:.3e} +/- {fn2_e:.2e} (S/N {snr_n2:.1f})",
        fontsize=9,
    )
    ax.set_ylabel(r"$f_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.legend(fontsize=8, loc="upper right")

    axr.step(x, (y - total_d) / e, where="mid", color="k", lw=1)
    axr.axhline(0, color="tab:orange", lw=1)
    axr.set_ylabel(r"resid / $\sigma$")
    axr.set_xlabel(r"observed wavelength [$\AA$]")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_ha_blend(spec, wave, flux, err, zf, grating, src_id, fdir):
    """Second fit: Halpha with [NII] 6548, 6584. Returns a dict of results.

    Runs on the Spectrum already used for the redshift fit, after that
    fit's results have been read, so it cannot change the redshift.
    """
    band = ha_blend_band_df()
    if not band_in_range(band.loc[HA_BLEND_LABEL], wave, zf):
        print("    Ha blend  out of range (NII flanks fall off the spectrum)")
        return {"blend_status": "out_of_range"}

    spec.fit.bands(HA_BLEND_LABEL, bands=band, fit_cfg=ha_blend_fit_cfg(),
                   cont_source=FLUX_CONT_SOURCE)

    gr = grating_short(grating)
    f_ha, f_ha_e, snr_ha = get_line_flux(spec, HA_LABEL)
    f_n2, f_n2_e, _ = get_line_flux(spec, NII_RED_LABEL)

    sig_v = frame_value(spec, HA_LABEL, "sigma_vel")
    sig_v_e = frame_value(spec, HA_LABEL, "sigma_vel_err")
    fwhm_obs = GAUSS_FWHM * sig_v
    fwhm_obs_e = GAUSS_FWHM * sig_v_e
    centre_obs = frame_value(spec, HA_LABEL, "center")
    fwhm_inst = instrumental_fwhm_kms(grating, centre_obs)
    fwhm_int, fwhm_int_e, unresolved = intrinsic_fwhm(fwhm_obs, fwhm_obs_e, fwhm_inst)

    print(f"    Ha blend  flux {f_ha:.3e} +/- {f_ha_e:.2e} (S/N {snr_ha:.1f})  "
          f"[NII]6584 {f_n2:.3e}  FWHM obs {fwhm_obs:.0f}  inst {fwhm_inst:.0f}  "
          f"int {fwhm_int:.0f} km/s  unresolved {unresolved}")

    png = os.path.join(fdir, f"{src_id}_{grating}_Ha_blend_fit.png")
    plot_ha_blend(spec, wave, flux, err, zf, band, src_id, grating, png)
    print(f"    figure    {png}")

    return {
        "blend_status": "fitted",
        f"Ha_{gr}_flux": f_ha,
        f"Ha_{gr}_flux_err": f_ha_e,
        f"Ha_{gr}_flux_snr": snr_ha,
        f"Ha_{gr}_fwhm_obs": fwhm_obs,
        f"Ha_{gr}_fwhm_obs_err": fwhm_obs_e,
        f"Ha_{gr}_fwhm_inst": fwhm_inst,
        f"Ha_{gr}_fwhm_int": fwhm_int,
        f"Ha_{gr}_fwhm_int_err": fwhm_int_e,
        f"Ha_{gr}_unresolved": unresolved,
        f"Ha_{gr}_NII6584_flux": f_n2,
        f"Ha_{gr}_NII6584_flux_err": f_n2_e,
    }


def fit_one_line(lime_path, z_dja, line):
    """Fit one line in one spectrum.

    Returns a dict with status 'fitted' or 'out_of_range', plus z, z_err,
    snr, success for 'fitted', and a 'meas' dict of any extra flux and
    width columns. Raises on genuine fit errors.
    """
    src_id, grating = parse_id_grating(lime_path)
    gr = grating_short(grating)
    name = line["name"]

    spec, wave, flux, err = load_lime_spectrum(lime_path, z_dja)
    zf = 1.0 + z_dja

    if line["kind"] == "single":
        band = single_band_df(line["label"], line["rest_vac"])
        ref_label = line["label"]
        rest_vac = line["rest_vac"]
    else:
        band = doublet_band_df(line["label_blend"], line["rest_vac_other"],
                                line["rest_vac_ref"])
        ref_label = line["label_ref"]
        rest_vac = line["rest_vac_ref"]

    if not band_in_range(band.iloc[0], wave, zf):
        print(f"    {name}  out of range")
        return {"status": "out_of_range"}

    fdir = os.path.join(fig_root(line), str(src_id))
    os.makedirs(fdir, exist_ok=True)
    contsub_png = os.path.join(fdir, f"{src_id}_{grating}_{name}_contsub.png")
    fit_png = os.path.join(fdir, f"{src_id}_{grating}_{name}_fit.png")

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=contsub_png)

    # ---- Redshift fit, unchanged ----
    if line["kind"] == "single":
        spec.fit.bands(line["label"], bands=band)
    else:
        fit_cfg = build_doublet_fit_cfg(line)
        spec.fit.bands(line["label_blend"], bands=band, fit_cfg=fit_cfg)

    centre, centre_err, snr = get_line_row(spec, ref_label)
    z_line = centre_to_redshift(centre, rest_vac)
    z_line_err = centre_err_to_redshift_err(centre_err, rest_vac)
    success = is_success(snr, centre_err)

    print(f"    {name}  z {z_line:.6f} +/- {z_line_err:.6f}  "
          f"S/N {snr:.2f}  centre_err {centre_err} AA  success {success}")

    spec.plot.bands(ref_label, fname=fit_png)

    # ---- Extra measurements ----
    meas = {}
    if line.get("store_flux"):
        # Refit with the flank continuum for the flux. z already read above.
        spec.fit.bands(line["label"], bands=band, cont_source=FLUX_CONT_SOURCE)
        f, fe, fsnr = get_line_flux(spec, ref_label)
        meas = {f"{name}_{gr}_flux": f,
                f"{name}_{gr}_flux_err": fe,
                f"{name}_{gr}_flux_snr": fsnr}
        print(f"    {name}  flux {f:.3e} +/- {fe:.2e} erg/s/cm2 (S/N {fsnr:.1f})")

    if line.get("ha_blend"):
        try:
            b = fit_ha_blend(spec, wave, flux, err, zf, grating, src_id, fdir)
        except Exception as e:
            print(f"    Ha blend  FAILED  reason: {e}")
            b = {"blend_status": "failed"}
        meas = {k: v for k, v in b.items() if k != "blend_status"}

    return {
        "status": "fitted",
        "z": z_line,
        "z_err": z_line_err,
        "snr": snr,
        "success": success,
        "meas": meas,
    }


# ----------------------------------------------------------------------
# Best-value selection across gratings
# ----------------------------------------------------------------------
def pick_best_flux(row, name, grs):
    """Highest flux S/N across gratings. Returns (grating, flux, err, snr)."""
    best = (None, np.nan, np.nan, np.nan)
    for gr in grs:
        snr = row.get(f"{name}_{gr}_flux_snr", np.nan)
        if snr is None or not np.isfinite(snr):
            continue
        if best[0] is None or snr > best[3]:
            best = (gr, row.get(f"{name}_{gr}_flux"),
                    row.get(f"{name}_{gr}_flux_err"), snr)
    return best


def pick_best_fwhm_grating(row, grs):
    """Among gratings with Ha flux S/N >= FWHM_SNR_MIN, prefer H over M,
    then the highest S/N."""
    cands = []
    for gr in grs:
        snr = row.get(f"Ha_{gr}_flux_snr", np.nan)
        fwhm = row.get(f"Ha_{gr}_fwhm_obs", np.nan)
        if snr is None or fwhm is None or not (np.isfinite(snr) and np.isfinite(fwhm)):
            continue
        if snr < FWHM_SNR_MIN:
            continue
        cands.append((gr.endswith("H"), snr, gr))
    if not cands:
        return None
    return max(cands)[2]


def add_best_columns(sum_df, line):
    """Append best-value columns to the summary frame for Ha and Hbeta."""
    name = line["name"]
    grs = [grating_short(g) for g in GRATINGS]
    out = {}
    for idx, row in sum_df.iterrows():
        row = row.to_dict()
        gr, f, fe, snr = pick_best_flux(row, name, grs)
        out.setdefault(f"{name}_flux_best", []).append(f)
        out.setdefault(f"{name}_flux_best_err", []).append(fe)
        out.setdefault(f"{name}_flux_best_snr", []).append(snr)
        out.setdefault(f"{name}_flux_best_grating", []).append(gr)

        if line.get("ha_blend"):
            g = pick_best_fwhm_grating(row, grs)
            for suff, outcol in (("fwhm_obs", "Ha_fwhm_obs_best"),
                                 ("fwhm_obs_err", "Ha_fwhm_obs_best_err"),
                                 ("fwhm_inst", "Ha_fwhm_inst_best"),
                                 ("fwhm_int", "Ha_fwhm_int_best"),
                                 ("fwhm_int_err", "Ha_fwhm_int_best_err")):
                out.setdefault(outcol, []).append(
                    row.get(f"Ha_{g}_{suff}", np.nan) if g else np.nan)
            out.setdefault("Ha_fwhm_unresolved_best", []).append(
                row.get(f"Ha_{g}_unresolved", np.nan) if g else np.nan)
            out.setdefault("Ha_fwhm_best_grating", []).append(g)

    for col, vals in out.items():
        sum_df[col] = vals
    return sum_df


def meas_suffixes(line):
    if line.get("ha_blend"):
        return ["flux", "flux_err", "flux_snr", "fwhm_obs", "fwhm_obs_err",
                "fwhm_inst", "fwhm_int", "fwhm_int_err", "unresolved",
                "NII6584_flux", "NII6584_flux_err"]
    if line.get("store_flux"):
        return ["flux", "flux_err", "flux_snr"]
    return []


def run_all(spectra_root, line):
    """Walk all four grating dirs, fit one line for every source, build
    that line's pair of CSVs."""
    name = line["name"]

    catalogues = {}
    for grating in GRATINGS:
        cp = catalogue_path(grating)
        if os.path.exists(cp):
            catalogues[grating] = Table.read(cp)
        else:
            print(f"catalogue missing, skipping grating: {cp}")

    results = {}   # src_id -> {col: value}
    summary = {}   # src_id -> {col: value}
    gratings_seen = {}

    for grating in GRATINGS:
        if grating not in catalogues:
            continue
        gr = grating_short(grating)
        cat = catalogues[grating]
        spec_dir = os.path.join(spectra_root, grating)
        files = sorted(glob.glob(os.path.join(spec_dir, "*_lime.fits")))
        print(f"\n{'#'*60}\n# {name}  {grating}   {len(files)} files\n{'#'*60}")

        for f in files:
            try:
                src_id, _ = parse_id_grating(f)
            except ValueError as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                continue

            z_dja = get_dja_redshift(cat, src_id)
            if z_dja is None:
                print(f"SKIPPED  {os.path.basename(f)}  no DJA z in catalogue")
                continue

            results.setdefault(src_id, {"ID": src_id})
            summary.setdefault(src_id, {"ID": src_id, f"{name}_success": 0})
            gratings_seen.setdefault(src_id, [])
            gratings_seen[src_id].append(gr)
            results[src_id][f"z_{gr}"] = z_dja

            print(f"\n--- ID {src_id}  {grating} ---")
            try:
                r = fit_one_line(f, z_dja, line)
            except Exception as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                summary[src_id][f"{name}_{gr}_success"] = False
                continue

            if r["status"] == "out_of_range":
                summary[src_id][f"{name}_{gr}_success"] = False
                continue

            results[src_id][f"z_{name}_{gr}"] = r["z"]
            results[src_id][f"z_{name}_{gr}_err"] = r["z_err"]
            results[src_id][f"z_{name}_{gr}_snr"] = r["snr"]

            summary[src_id][f"{name}_{gr}_success"] = r["success"]
            if r["success"]:
                summary[src_id][f"{name}_success"] += 1

            results[src_id].update(r["meas"])
            summary[src_id].update(r["meas"])

    for src_id, grs in gratings_seen.items():
        summary[src_id]["gratings"] = ", ".join(grs)

    grs_all = [grating_short(g) for g in GRATINGS]
    meas_cols = [f"{name}_{gr}_{s}" for gr in grs_all for s in meas_suffixes(line)]

    # ---- Results CSV: original columns first, new measurements after ----
    res_df = pd.DataFrame(list(results.values()))
    if len(res_df):
        res_df = res_df.sort_values("ID")
    res_cols = ["ID"]
    for gr in grs_all:
        if f"z_{gr}" in res_df.columns:
            res_cols.append(f"z_{gr}")
    for gr in grs_all:
        for suff in ("", "_err", "_snr"):
            c = f"z_{name}_{gr}{suff}"
            if c in res_df.columns:
                res_cols.append(c)
    res_cols += [c for c in meas_cols if c in res_df.columns]
    res_df = res_df.reindex(columns=res_cols)
    res_df.to_csv(results_csv(line), index=False)

    # ---- Summary CSV: original columns first, then measurements and best ----
    sum_df = pd.DataFrame(list(summary.values()))
    if len(sum_df):
        sum_df = sum_df.sort_values("ID")
    sum_cols = ["ID", "gratings"]
    for gr in grs_all:
        c = f"{name}_{gr}_success"
        if c in sum_df.columns:
            sum_cols.append(c)
    sum_cols.append(f"{name}_success")
    sum_cols += [c for c in meas_cols if c in sum_df.columns]
    sum_df = sum_df.reindex(columns=sum_cols)
    if meas_suffixes(line) and len(sum_df):
        sum_df = add_best_columns(sum_df, line)
    sum_df.to_csv(summary_csv(line), index=False)

    print("\n" + "=" * 60)
    print(f"DONE  {name}")
    print(f"  sources processed   {len(results)}")
    if len(sum_df):
        print(f"  with >=1 success    {int((sum_df[f'{name}_success'] > 0).sum())}")
        if f"{name}_flux_best" in sum_df.columns:
            print(f"  with a flux         {int(sum_df[f'{name}_flux_best'].notna().sum())}")
        if "Ha_fwhm_obs_best" in sum_df.columns:
            print(f"  with a best FWHM    {int(sum_df['Ha_fwhm_obs_best'].notna().sum())}")
    print(f"  results csv         {os.path.abspath(results_csv(line))}")
    print(f"  summary csv         {os.path.abspath(summary_csv(line))}")
    print(f"  figures root        {os.path.abspath(fig_root(line))}")
    print("=" * 60)


def main():
    spectra_root = sys.argv[1] if len(sys.argv) == 2 else JWST_SPECTRA_ROOT

    print("lime_lines_jointfit.py")
    print(f"  spectra root   {spectra_root}")
    print(f"  catalogue dir  {CATALOG_DIR}")
    print("  lines          " + ", ".join(l["name"] for l in LINES))
    if any(l.get("ha_blend") for l in LINES):
        print(f"  resolution     {msaexp_data_dir()}  (R x {R_SCALE})")
    print("  output files")
    for line in LINES:
        print(f"    {os.path.abspath(results_csv(line))}")
        print(f"    {os.path.abspath(summary_csv(line))}")
        print(f"    {os.path.abspath(fig_root(line))}/<ID>/")

    for line in LINES:
        os.makedirs(fig_root(line), exist_ok=True)
        run_all(spectra_root, line)


if __name__ == "__main__":
    main()