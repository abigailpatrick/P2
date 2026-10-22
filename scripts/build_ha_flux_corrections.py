#!/usr/bin/env python
"""
build_ha_flux_corrections.py

The total, dust-corrected Halpha flux of every source in the z_sys sample,
for the Lya escape fractions (measure_lya_fesc.py). Keyed on Isaac's IDs.

    F_Ha,corr = F_Ha,total x 10^(0.4 A_Ha)

Every source falls in one case, recorded in ha_case
---------------------------------------------------
  case 1  ha_source spec, dust_method balmer
          Halpha from a NIRSpec spectrum, made total with its slit-loss
          factor, and A_Ha from the Balmer decrement Halpha / Hbeta.
  case 2  ha_source spec, dust_method sed
          Halpha from a NIRSpec spectrum, made total with its slit-loss
          factor, and A_Ha from Ken's SED fit (sed_aha), because no usable
          Hbeta was measured.
  case 3  ha_source phot, dust_method sed
          No usable spectral Halpha, so Halpha from the excess flux in the
          photometric band containing it (Isaac's total photometry, so
          already a total flux) and A_Ha from the SED fit, as Paper 1.
  case 0  no corrected Halpha, ha_reason says why. Usually ha_source none (no
          Halpha), but ha_source spec or phot if Halpha was measured and only
          the A_Ha is missing (no SED fit).
Hbeta is only ever used in the decrement, never as a replacement for Halpha.

1. Spectral line fluxes
-----------------------
From lime_Ha_fits.csv and lime_Hbeta_fits.csv (lime_fit_lines.py), one row
per source and grating, erg/s/cm2. Halpha is deblended from [NII]. A line
counts if its flux S/N >= --ha-snr-min (3) or --hb-snr-min (3), and only if
the observed line lies inside the grating's nominal range (JDox: G235M
1.66-3.07, G235H 1.66-3.05, G395M 2.87-5.10, G395H 2.87-5.14 micron). The
DJA spectra run beyond these limits, but there the flux calibration is
poor and second-order light contaminates, which gave unphysical Balmer
decrements (--allow-extended keeps those lines).

2. Slit-loss factor S, per line and spectrum
--------------------------------------------
For every spectrum, S = f_nu(Isaac total photometry) / f_nu(spectrum through
the same filter) is computed in every NIRCam band (synthetic photometry,
photon-weighted as for AB fluxes, Fukugita et al. 1996, Bessell & Murphy 2012):
    f_nu(synthetic) = int f_nu T / lambda dlambda / int T / lambda dlambda
A band passes if
  a. its half-maximum range lies inside the grating's nominal range,
  b. the spectrum covers >= --min-coverage (0.9) of its photon-weighted
     response (small gaps are interpolated, the few uncovered per cent filled
     with the median f_nu in the band),
  c. both fluxes have S/N >= --slit-snr-min (3),
  d. S lies inside --slit-range (0.5 to 5). Below 0.5 the spectrum would hold
     twice the galaxy's total light (a wrong match, a neighbour in the shutter
     or a bad background), above 5 the shutter mostly missed the galaxy.
Both fluxes contain the lines and the continuum, so S is total light over
light in the shutter. Sources whose photometry Flag contains 1, 2 or 3 get
no factor (--allow-flagged-phot). Every passing band is written to
--bands-out. Then, for each line, by --slit-mode:
  local (default)
    1. line_band    a passing band containing the line (transmission at the
                    line >= half its peak), medium before wide, then S/N
    2. interp       otherwise S interpolated in log S against log lambda
                    between the nearest passing band blueward and redward of
                    the line, so S is taken at the line's own wavelength
    3. spec_median  otherwise (bands on one side only) the median of the
                    spectrum's passing bands
  median
    the median over all passing bands of the spectrum, as Shapley et al.
    (2023) scale each spectrum by the median ratio over all bands with good
    S/N (they require 3, --min-bands here, default 2). With fewer bands the
    local choice is used.
  band
    only the band containing the line (the previous behaviour)
A median's error is the larger of the scatter between bands / sqrt(N) and the
typical single-band error. An interpolated S carries both bands' errors.
Where none of these exist the spectrum has no factor of its own and
--fallback grating (default) takes, in order,
  other_grating   the median S of the same source's spectrum in another
                  grating (most passing bands first). The error adds the
                  measured scatter between two gratings of one source
                  (fractional NMAD of their ratio) in quadrature.
  grating_median  the median of the own factors of that line in that
                  grating, with its NMAD as the error, if the grating has
                  at least --min-grating of them
  sample_median   the median of the own factors of that line over the whole
                  sample, with the NMAD as its error
--fallback sample goes straight to the sample median (the previous
behaviour). ha_slit_note says why the spectrum had no factor. Slit losses
scatter by about 45 per cent from source to source but agree to about 10
per cent between two gratings of one source, so the same source is used
first. For each line, a spectrum with its
own factor is chosen over one without, then the highest line S/N. Every
spectral Halpha also gets the spectrum median and the number of passing bands
(ha_slit_spec_median, ha_slit_nbands), to check the factor used.
    F_Ha,total = S x F_Ha,obs

3. Balmer decrement (case 1)
----------------------------
balmer_spectra = same      Halpha and Hbeta both pass in one spectrum (the
                           one with the best S/N on the ratio). The observed
                           ratio is used, as the slit loss largely cancels
                           (--balmer-basis slitcorr corrects each line
                           first instead). Both versions are printed.
balmer_spectra = different no spectrum has both, so the best Halpha and the
                           best Hbeta spectra are each slit-corrected first,
                           R = (S_Ha F_Ha) / (S_Hb F_Hb).
    E(B-V)_gas = 2.5 / (k_Hb - k_Ha) log10(R / 2.86)   (0 if R < 2.86)
    A_Ha       = k_Ha E(B-V)_gas
Case B 2.86 (10^4 K, n_e = 100 cm^-3, Osterbrock & Ferland 2006), Calzetti
et al. (2000) curve with R_V = 4.05, so k_Ha = 3.33 and k_Hb = 4.60.

4. SED attenuation (cases 2 and 3)
----------------------------------
A_Ha = sed_aha from build_sed_catalog.py (Ken's BAGPIPES A_Ha posterior,
median with 16th and 84th percentiles). Every source also gets
ha_flux_corr_sed, its Halpha corrected with the SED A_Ha whatever its case,
so the escape fractions can be compared with one dust method throughout.

5. Photometric Halpha (case 3, and a check for every source)
-------------------------------------------------------------
In the band containing Halpha (medium preferred, then wide, transmission
>= half its peak, no other strong line in it):
    Delta f_nu = f_nu,band - f_nu,cont
    F_Ha = Delta f_nu c int(T / lambda) dlambda / sum_i r_i T(lambda_i) lambda_i
with the sum over Halpha (r = 1), [NII] 6584 (r = n), [NII] 6548 (n / 3.05)
and [SII] 6716 + 6731 (r = s, split evenly). n is the sample median
[NII]6584 / Halpha of the spectral fits (--nii-ha to fix it), s is
--sii-ha (default 0, so any [SII] in the band is counted as Halpha). The
continuum is a power law in f_nu through the nearest band on each side
that contains no strong line ([OII], Hbeta, [OIII], Halpha + [NII], [SII],
[SIII], HeI 10830, Pa gamma, Pa beta, Pa alpha) and is evaluated at the
line band's pivot. Both sides are needed unless --phot-one-side (then a
flat f_nu from one band). It is computed for every source with photometry,
so where both exist it is compared with the spectral total Halpha.

Errors
------
--n-mc (5000) draws per source of every input: line fluxes, slit factors,
photometric fluxes (Gaussian, drawn values below zero allowed for the
fluxes, factors kept > 0) and the SED A_Ha (split normal from its
percentiles, kept >= 0). The decrement is recomputed in each draw. Values
are the best estimates. Errors are value - p16 and p84 - value (clipped at
0). The draws of ha_flux_corr and ha_flux_corr_sed are saved for
measure_lya_fesc.py, so its errors carry the full distributions.

Columns of ha_flux_corrections.csv
----------------------------------
  ID, z_sys, ha_case (0 to 3), ha_source (spec, phot, none), ha_reason,
  dust_method (balmer, sed, none), balmer_spectra (same, different, none)
  ha_grating, ha_flux_obs, ha_flux_obs_err, ha_snr          spectral Halpha used
  ha_slit_factor, ha_slit_factor_err, ha_slit_band, ha_slit_source
                                (line_band, interp, spec_median, median, other_grating,
                                 grating_median, sample_median)
  ha_slit_note, ha_slit_spec_median, ha_slit_nbands
  hb_grating, hb_flux_obs, hb_flux_obs_err, hb_snr          Hbeta of the decrement
  hb_slit_factor, hb_slit_factor_err, hb_slit_band, hb_slit_source
  balmer_ha_grating             spectrum of the decrement's Halpha
  balmer_ratio (+ _err_lo, _err_hi)   the R used
  balmer_ratio_obs, balmer_ratio_slitcorr   both versions, where defined
  ebv_gas (+ errs), a_ha_balmer (+ errs)
  a_ha_sed (+ errs)                   from sed_properties.csv
  a_ha (+ _err_lo, _err_hi)           the A_Ha used (balmer or sed)
  ha_flux_total (+ errs)              S x F_obs, or the photometric flux
  ha_flux_corr (+ errs)               the final value, erg/s/cm2
  ha_flux_corr_sed (+ errs)           with the SED A_Ha whatever the case
  ha_phot_flux (+ _err), ha_phot_snr, ha_phot_band, ha_phot_cont_bands,
  ha_phot_cont_method, ha_phot_note   the photometric estimate for every source
  isaac_flag, phot_flagged

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, z_sys
  jwst_catalogs/lime_Ha_fits.csv, lime_Hbeta_fits.csv
  jwst_catalogs/cosmos_primer_minerva_production.fits   Isaac's photometry
  jwst_catalogs/sed_properties.csv                  sed_aha (build_sed_catalog.py)
  jwst_spectra_pm/<grating>_<filter>/<ID>_<grating>_<filter>_spectra_lime.fits
  jwst_catalogs/nircam_filters/JWST_NIRCam.<FILT>.dat   (--download-filters)

Outputs
-------
  jwst_catalogs/ha_flux_corrections.csv
  jwst_catalogs/ha_flux_corrections_mc.npz      MC draws for measure_lya_fesc.py
  jwst_catalogs/ha_slit_bands.csv               every passing band factor of every spectrum
  plots/ha_flux_corrections.png                 diagnostic figure

Usage
-----
python build_ha_flux_corrections.py --download-filters   # once, if the curves are missing
python build_ha_flux_corrections.py
python build_ha_flux_corrections.py --balmer-basis slitcorr
python build_ha_flux_corrections.py --slit-mode median            # Shapley-style median of all bands
python build_ha_flux_corrections.py --slit-mode band --min-coverage 0.7   # the previous behaviour
python build_ha_flux_corrections.py --sii-ha 0.1 --phot-one-side
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from astropy.io import fits

import p2_common as pc
from fit_muv_beta import FILTERS, load_isaac

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
SPEC_ROOT = f"{pc.P2_ROOT}/jwst_spectra_pm"
FILTER_DIR = f"{CAT_DIR}/nircam_filters"
OUT_CSV = f"{CAT_DIR}/ha_flux_corrections.csv"
OUT_NPZ = f"{CAT_DIR}/ha_flux_corrections_mc.npz"
OUT_PNG = f"{pc.P2_ROOT}/plots/ha_flux_corrections.png"
SED_CSV = f"{CAT_DIR}/sed_properties.csv"
ISAAC_FITS = f"{CAT_DIR}/cosmos_primer_minerva_production.fits"

GRATING_FILTER = {"G235M": "F170LP", "G235H": "F170LP", "G395M": "F290LP", "G395H": "F290LP"}
NOMINAL_UM = pc.NIRSPEC_NOMINAL_UM     # nominal grating ranges, micron, shared with z_sys
# NIRCam bands of Isaac's catalogue that G235 / G395 can cover, used for the
# slit factors and as the photometric Halpha band
SLIT_FILTERS = ["F182M", "F200W", "F210M", "F250M", "F277W", "F300M",
                "F356W", "F360M", "F410M", "F444W", "F460M"]
SVO_URL = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=JWST/NIRCam.{filt}"

C_ANG = 2.99792458e18          # AA / s
HA_REST = 6564.61              # vacuum AA
HB_REST = 4862.68
NII_A, NII_B = 6585.27, 6549.86
SII_A, SII_B = 6718.29, 6732.67
R_INT = 2.86
RV_CALZETTI = 4.05
COVER_T_FLOOR = 0.01
GAP_TOL_PIX = 5
# strong lines (rest vacuum AA) a continuum band must avoid
STRONG = [3727.1, 3729.9, 4862.7, 4960.3, 5008.2, 6549.9, 6564.6, 6585.3, 6718.3, 6732.7,
          9071.1, 9533.2, 10833.3, 10941.1, 12821.6, 18756.1]
HA_COMPLEX = [6549.9, 6564.6, 6585.3, 6718.3, 6732.7]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download-filters", action="store_true",
                   help="download any missing NIRCam filter curves from SVO, then exit")
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--ha-fits", default=f"{CAT_DIR}/lime_Ha_fits.csv")
    p.add_argument("--hb-fits", default=f"{CAT_DIR}/lime_Hbeta_fits.csv")
    p.add_argument("--isaac", default=ISAAC_FITS)
    p.add_argument("--sed-csv", default=SED_CSV)
    p.add_argument("--spec-root", default=SPEC_ROOT)
    p.add_argument("--filter-dir", default=FILTER_DIR)
    p.add_argument("--ha-snr-min", type=float, default=3.0,
                   help="flux S/N for a spectral or photometric Halpha to be used")
    p.add_argument("--hb-snr-min", type=float, default=3.0,
                   help="flux S/N for Hbeta to be used in the decrement")
    p.add_argument("--slit-mode", choices=["local", "median", "band"], default="local",
                   help="local: the band containing the line, else interpolate between the "
                        "nearest bands either side, else the spectrum median. median: the "
                        "median over all passing bands (Shapley et al. 2023), local if fewer "
                        "than --min-bands. band: only the band containing the line")
    p.add_argument("--slit-snr-min", type=float, default=3.0)
    p.add_argument("--min-coverage", type=float, default=0.90,
                   help="fraction of a band's photon-weighted response that the spectrum must cover")
    p.add_argument("--min-bands", type=int, default=2,
                   help="bands needed for --slit-mode median (Shapley et al. use 3)")
    p.add_argument("--slit-range", type=float, nargs=2, default=[0.5, 5.0],
                   help="band slit factors outside this range are not used")
    p.add_argument("--bands-out", default=f"{CAT_DIR}/ha_slit_bands.csv",
                   help="every band slit factor of every spectrum used")
    p.add_argument("--allow-extended", action="store_true",
                   help="also use lines beyond the grating's nominal wavelength range")
    p.add_argument("--min-for-median", type=int, default=3)
    p.add_argument("--fallback", choices=["grating", "sample"], default="grating",
                   help="for a spectrum with no factor of its own. grating: the same source's "
                        "other grating, else the median of its grating, else the sample median. "
                        "sample: the sample median only")
    p.add_argument("--min-grating", type=int, default=5,
                   help="own factors a grating needs before its median is used as a fallback")
    p.add_argument("--cross-grating-err", type=float, default=0.10,
                   help="fractional error added for an other_grating factor if fewer than 5 "
                        "sources have factors in two gratings to measure it from")
    p.add_argument("--allow-flagged-phot", action="store_true",
                   help="also use photometry whose Flag contains 1, 2 or 3")
    p.add_argument("--balmer-basis", choices=["auto", "slitcorr"], default="auto",
                   help="auto: observed ratio when both lines are in one spectrum, "
                        "slit-corrected otherwise. slitcorr: always slit-correct both")
    p.add_argument("--nii-ha", type=float, default=None,
                   help="[NII]6584/Halpha for the photometric Halpha (default the sample median)")
    p.add_argument("--sii-ha", type=float, default=0.0,
                   help="([SII]6716+6731)/Halpha for the photometric Halpha (default 0)")
    p.add_argument("--phot-one-side", action="store_true",
                   help="allow a photometric continuum from one side only (flat f_nu)")
    p.add_argument("--n-mc", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--out-npz", default=OUT_NPZ)
    p.add_argument("--plot", default=OUT_PNG)
    return p.parse_args()


# ============================================================================
# Filters and attenuation
# ============================================================================

def filter_path(filt, d):
    return os.path.join(d, f"JWST_NIRCam.{filt}.dat")


def download_filters(d):
    os.makedirs(d, exist_ok=True)
    for filt in SLIT_FILTERS:
        out = filter_path(filt, d)
        if os.path.exists(out):
            print(f"  have  {os.path.abspath(out)}")
            continue
        url = SVO_URL.format(filt=filt)
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
        rows = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
        if len(rows) < 10:
            raise RuntimeError(f"Download for {filt} looks empty, got {len(rows)} rows")
        with open(out, "w") as fh:
            fh.write(f"# SVO FPS JWST/NIRCam.{filt}\n# {url}\n# wavelength_AA transmission\n")
            fh.write("\n".join(rows) + "\n")
        print(f"  wrote {os.path.abspath(out)}  ({len(rows)} rows)")


def load_filters(d):
    missing = [f for f in SLIT_FILTERS if not os.path.exists(filter_path(f, d))]
    if missing:
        raise SystemExit(f"Filter curves missing for {missing} in {d}. "
                         "Run with --download-filters first.")
    curves = {}
    for filt in SLIT_FILTERS:
        x = np.loadtxt(filter_path(filt, d))
        o = np.argsort(x[:, 0])
        curves[filt] = (x[o, 0], x[o, 1])
    return curves


def filter_rank(filt):
    return {"N": 0, "M": 1, "W": 2}[filt[-1]]


def t_at(curve, lam):
    lam_f, t_f = curve
    return float(np.interp(lam, lam_f, t_f, left=0.0, right=0.0))


def k_calzetti(lam_aa):
    x = lam_aa / 1e4
    if 0.63 <= x <= 2.20:
        return 2.659 * (-1.857 + 1.040 / x) + RV_CALZETTI
    return 2.659 * (-2.156 + 1.509 / x - 0.198 / x ** 2 + 0.011 / x ** 3) + RV_CALZETTI


K_HA = k_calzetti(HA_REST)
K_HB = k_calzetti(HB_REST)


def a_ha_from_ratio(r):
    """A_Ha (mag) from Halpha/Hbeta, 0 where R <= 2.86, NaN where R <= 0."""
    r = np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = K_HA * 2.5 / (K_HB - K_HA) * np.log10(np.clip(r, R_INT, None) / R_INT)
    return np.where(r > 0, a, np.nan)


# ============================================================================
# Spectra and slit factors
# ============================================================================

def spectrum_path(root, sid, gr):
    full = f"{gr}_{GRATING_FILTER[gr]}"
    return os.path.join(root, full, f"{sid}_{full}_spectra_lime.fits")


def load_spectrum(root, sid, gr):
    """Wavelength (AA), f_lambda and error (erg/s/cm2/AA) of the LiMe file."""
    with fits.open(spectrum_path(root, sid, gr)) as h:
        t = h["SPECTRUM"].data
        w, f, e = (np.asarray(t[c], dtype=float) for c in ("WAVE", "FLUX", "ERR"))
    ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    o = np.argsort(w[ok])
    return w[ok][o], f[ok][o], e[ok][o]


def _trapz(y, x):
    return (getattr(np, "trapezoid", None) or np.trapz)(y, x)


def covered_mask(w, x):
    dw = np.median(np.diff(w))
    x = np.atleast_1d(x)
    i = np.searchsorted(w, x)
    inside = (i > 0) & (i < len(w))
    gap = np.full(x.size, np.inf)
    gap[inside] = w[i[inside]] - w[i[inside] - 1]
    return inside & (gap <= GAP_TOL_PIX * dw)


def coverage(w, lam_f, t_f):
    keep = t_f >= COVER_T_FLOOR * t_f.max()
    lf, tf = lam_f[keep], t_f[keep]
    wt = tf / lf
    return _trapz(wt * covered_mask(w, lf), lf) / _trapz(wt, lf)


def synthetic_fnu(w, f, e, lam_f, t_f):
    """Photon-weighted synthetic f_nu (uJy) and error from an f_lambda spectrum,
    int f_nu T/lambda dlambda / int T/lambda dlambda. Uncovered wings are filled
    with the median f_nu inside the band."""
    t = np.interp(w, lam_f, t_f, left=0.0, right=0.0)
    on = t >= COVER_T_FLOOR * t_f.max()
    fnu = f * w ** 2 / C_ANG          # erg/s/cm2/Hz
    enu = e * w ** 2 / C_ANG
    dl = np.gradient(w)
    den = np.sum(t / w * dl)
    f_cov = np.sum(fnu * t / w * dl) / den
    e_cov = np.sqrt(np.sum((enu * t / w * dl) ** 2)) / den
    cov = coverage(w, lam_f, t_f)
    if cov < 1.0 and on.any():
        fill = np.median(fnu[on])
        fill_err = 1.253 * np.median(enu[on]) / np.sqrt(on.sum())
    else:
        fill, fill_err = 0.0, 0.0
    val = cov * f_cov + (1 - cov) * fill
    err = np.hypot(cov * e_cov, (1 - cov) * fill_err)
    return val / 1e-29, err / 1e-29, cov


DIRECT_SOURCES = ("line_band", "interp", "spec_median", "median")
SLIT_MODES = ("local", "median", "band")


def band_nominal(curve, grating):
    """True if the band's half-maximum range lies inside the grating's nominal range."""
    lam, t = curve
    on = lam[t >= 0.5 * t.max()]
    lo, hi = NOMINAL_UM.get(grating, (np.nan, np.nan))
    return bool(on.size and on.min() / 1e4 >= lo and on.max() / 1e4 <= hi)


def spectrum_band_factors(spec, grating, phot_row, curves, a):
    """S = photometry / synthetic photometry in every NIRCam band of one
    spectrum. Returns (passing, rejected): passing is a list of dicts (band,
    pivot, factor, factor_err, snr, cov, synth, phot), sorted by pivot, of the
    bands inside the nominal range with coverage >= --min-coverage, S/N >=
    --slit-snr-min in both fluxes and S inside --slit-range. rejected lists
    'band reason' strings."""
    if spec is None:
        return [], ["no spectrum file"]
    w, f, e = spec
    good, bad = [], []
    for filt in SLIT_FILTERS:
        fp, ep = phot_row.get(filt.lower(), np.nan), phot_row.get(f"{filt.lower()}_err", np.nan)
        if not (np.isfinite(fp) and np.isfinite(ep) and ep > 0):
            continue
        lam_f, t_f = curves[filt]
        if not band_nominal(curves[filt], grating):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            fs, es, cov = synthetic_fnu(w, f, e, lam_f, t_f)
        if not (np.isfinite(fs) and np.isfinite(es) and es > 0) or cov <= 0:
            continue
        if cov < a.min_coverage:
            bad.append(f"{filt} coverage {cov:.2f}")
            continue
        if not (fs / es >= a.slit_snr_min and fp / ep >= a.slit_snr_min):
            bad.append(f"{filt} S/N synth {fs / es:.1f} phot {fp / ep:.1f}")
            continue
        fac = fp / fs
        fac_e = fac * np.hypot(ep / fp, es / fs)
        if not (a.slit_range[0] <= fac <= a.slit_range[1]):
            bad.append(f"{filt} S {fac:.2f} outside {a.slit_range[0]:g} to {a.slit_range[1]:g}")
            continue
        good.append(dict(band=filt, pivot=FILTERS[filt][1], factor=fac, factor_err=fac_e,
                         snr=fac / fac_e, cov=cov, synth=fs, phot=fp))
    return sorted(good, key=lambda d: d["pivot"]), bad


def spec_median(bands):
    """Median S of a spectrum's passing bands, with an error that is the
    larger of the scatter between bands / sqrt(N) and the typical single-band
    error. None if there are no bands."""
    if not bands:
        return None
    v = np.array([b["factor"] for b in bands])
    ev = np.array([b["factor_err"] for b in bands])
    med = float(np.median(v))
    scat = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    err = max(scat, float(np.median(ev)))
    return dict(band="+".join(b["band"] for b in bands), factor=med, factor_err=err,
                snr=med / err if err > 0 else np.nan, n=v.size)


def line_factor(bands, lam_line, curves, a):
    """The slit factor for one line from its spectrum's passing bands, by
    --slit-mode. Returns (dict with factor, factor_err, band, snr, source) or
    (None, reason)."""
    if not bands:
        return None, "no band passes"
    med = spec_median(bands)
    if a.slit_mode == "median":
        if med["n"] >= a.min_bands:
            return dict(med, source="median"), ""
        # too few bands for a median: fall through to the local choice
    containing = [b for b in bands if t_at(curves[b["band"]], lam_line) >= 0.5 * curves[b["band"]][1].max()]
    if containing:
        best = min(containing, key=lambda b: (filter_rank(b["band"]), -b["snr"]))
        return dict(best, source="line_band"), ""
    if a.slit_mode == "band":
        return None, "no passing band contains the line"
    blue = [b for b in bands if b["pivot"] < lam_line]
    red = [b for b in bands if b["pivot"] > lam_line]
    if blue and red:
        b1, b2 = blue[-1], red[0]
        wgt = np.log(lam_line / b1["pivot"]) / np.log(b2["pivot"] / b1["pivot"])
        lnS = (1 - wgt) * np.log(b1["factor"]) + wgt * np.log(b2["factor"])
        S = float(np.exp(lnS))
        Se = S * float(np.hypot((1 - wgt) * b1["factor_err"] / b1["factor"],
                                wgt * b2["factor_err"] / b2["factor"]))
        return dict(band=f"{b1['band']}+{b2['band']}", factor=S, factor_err=Se,
                    snr=S / Se if Se > 0 else np.nan, source="interp"), ""
    return dict(med, source="spec_median"), ""


# ============================================================================
# Photometric Halpha
# ============================================================================

def band_has_line(filt, z, lines):
    _, _, lo, hi = FILTERS[filt]
    return any(lo <= l * (1 + z) <= hi for l in lines)


def phot_halpha(prow, z, curves, nii, sii, a):
    """Photometric Halpha (erg/s/cm2) from the band excess. Returns a dict with
    flux, err, snr, band, cont bands, method, note, and a function to redraw
    it from drawn band fluxes for the MC."""
    out = dict(flux=np.nan, err=np.nan, snr=np.nan, band="", cont="", method="", note="")
    lam = HA_REST * (1 + z)
    other = [l for l in STRONG if l not in HA_COMPLEX]
    cands = []
    for filt in SLIT_FILTERS:
        fb, eb = prow.get(filt.lower(), np.nan), prow.get(f"{filt.lower()}_err", np.nan)
        if not (np.isfinite(fb) and np.isfinite(eb) and eb > 0):
            continue
        tmax = curves[filt][1].max()
        if t_at(curves[filt], lam) < 0.5 * tmax:
            continue
        if band_has_line(filt, z, other):
            continue
        lam_f, t_f = curves[filt]
        cands.append((filter_rank(filt), -t_at(curves[filt], lam) / _trapz(t_f / lam_f, lam_f), filt))
    if not cands:
        out["note"] = "no clean band contains Halpha"
        return out, None
    band = sorted(cands)[0][2]
    lam_f, t_f = curves[band]
    i_band = _trapz(t_f / lam_f, lam_f)
    lines = [(HA_REST, 1.0), (NII_A, nii), (NII_B, nii / 3.05), (SII_A, sii / 2), (SII_B, sii / 2)]
    denom = sum(r * t_at(curves[band], l * (1 + z)) * l * (1 + z) for l, r in lines)
    piv_band = FILTERS[band][1]

    clean = []
    for f in FILTERS:
        if f == band:
            continue
        fv, ev = prow.get(f.lower(), np.nan), prow.get(f"{f.lower()}_err", np.nan)
        if not (np.isfinite(fv) and np.isfinite(ev) and ev > 0):
            continue
        if band_has_line(f, z, STRONG):
            continue
        clean.append((FILTERS[f][1], f))
    blue = [c for c in clean if c[0] < lam]
    red = [c for c in clean if c[0] > lam]
    blue = max(blue)[1] if blue else None
    red = min(red)[1] if red else None
    if blue and red:
        method, cont = "power_law", [blue, red]
    elif (blue or red) and a.phot_one_side:
        method, cont = "one_side_flat", [blue or red]
    else:
        out.update(band=band, note="no clean continuum band on "
                   + ("both sides" if not (blue or red) else "one side"))
        return out, None

    def calc(fb, fc):
        """fb: band flux uJy, fc: list of continuum fluxes uJy (arrays ok)."""
        if method == "power_law":
            (p1, p2), (f1, f2) = (FILTERS[cont[0]][1], FILTERS[cont[1]][1]), fc
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.log(np.clip(f2, 1e-12, None) / np.clip(f1, 1e-12, None)) / np.log(p2 / p1)
                fcont = f1 * (piv_band / p1) ** slope
            fcont = np.where((f1 > 0) & (f2 > 0), fcont, 0.5 * (f1 + f2))
        else:
            fcont = fc[0]
        return (fb - fcont) * 1e-29 * C_ANG * i_band / denom

    fb = prow[band.lower()]
    fc = [prow[c.lower()] for c in cont]
    flux = float(calc(fb, fc))
    out.update(flux=flux, band=band, cont="+".join(cont), method=method)

    def draw(n, rng):
        b = rng.normal(fb, prow[f"{band.lower()}_err"], n)
        c = [rng.normal(prow[x.lower()], prow[f"{x.lower()}_err"], n) for x in cont]
        return calc(b, c)
    return out, draw


# ============================================================================
# Helpers
# ============================================================================

def split_normal(val, lo, hi, n, rng, floor=None):
    u = rng.standard_normal(n)
    lo = 0.0 if not np.isfinite(lo) else lo
    hi = 0.0 if not np.isfinite(hi) else hi
    x = val + np.where(u < 0, u * lo, u * hi)
    return np.clip(x, floor, None) if floor is not None else x


def pos_normal(val, err, n, rng):
    """Normal draws kept > 0 by redrawing (for slit factors)."""
    err = err if np.isfinite(err) and err > 0 else 0.0
    x = rng.normal(val, err, n)
    bad = x <= 0
    tries = 0
    while bad.any() and tries < 20:
        x[bad] = rng.normal(val, err, bad.sum())
        bad = x <= 0
        tries += 1
    x[bad] = val
    return x


def errs(val, draws):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size < 10 or not np.isfinite(val):
        return np.nan, np.nan
    p16, p84 = np.percentile(d, [16, 84])
    return max(val - p16, 0.0), max(p84 - val, 0.0)


def in_nominal(df, z_of, rest):
    """True for rows whose observed line is inside the grating's nominal range."""
    lam = df["ID"].map(z_of).add(1.0) * rest / 1e4
    lo = df["grating"].map(lambda g: NOMINAL_UM.get(g, (np.nan, np.nan))[0])
    hi = df["grating"].map(lambda g: NOMINAL_UM.get(g, (np.nan, np.nan))[1])
    return (lam >= lo) & (lam <= hi)


def line_rows(df, snr_min):
    """{ID: [(grating, flux, err, snr), ...]} of fitted rows with S/N >= snr_min."""
    df = df[df["status"] == "fitted"].copy()
    for c in ("flux", "flux_err", "flux_snr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["flux_snr"] >= snr_min) & np.isfinite(df["flux"]) & (df["flux_err"] > 0)]
    out = {}
    for _, r in df.iterrows():
        out.setdefault(int(r["ID"]), []).append((r["grating"], float(r["flux"]),
                                                 float(r["flux_err"]), float(r["flux_snr"])))
    return out


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ============================================================================
# Main
# ============================================================================

def main():
    a = parse_args()
    print("build_ha_flux_corrections.py")
    if a.download_filters:
        print(f"  filter curves in {os.path.abspath(a.filter_dir)}")
        download_filters(a.filter_dir)
        return
    for p_ in (a.zsys, a.ha_fits, a.hb_fits, a.isaac, a.sed_csv):
        print(f"  reading  {os.path.abspath(p_)}")
    print(f"  reading  {os.path.abspath(a.spec_root)}/<grating>_<filter>/<ID>_*_spectra_lime.fits")
    print(f"  reading  {os.path.abspath(a.filter_dir)}/JWST_NIRCam.<FILT>.dat")
    for p_ in (a.out, a.out_npz, a.bands_out, a.plot):
        print(f"  writing  {os.path.abspath(p_)}")
    print(f"  Halpha S/N >= {a.ha_snr_min:g}, Hbeta S/N >= {a.hb_snr_min:g}, "
          f"Balmer basis {a.balmer_basis}, k(Ha) {K_HA:.3f}, k(Hb) {K_HB:.3f} (Calzetti 2000), "
          f"{a.n_mc} MC draws")

    rng = np.random.default_rng(a.seed)
    curves = load_filters(a.filter_dir)
    zs = pc.load_sample(a.zsys, need=("ID", "z_sys"))
    ha_df, hb_df = pd.read_csv(a.ha_fits), pd.read_csv(a.hb_fits)
    for d in (ha_df, hb_df):
        d["ID"] = d["ID"].astype(int)
    z_of = dict(zip(zs["ID"].astype(int), zs["z_sys"].astype(float)))
    for name, df, rest in (("Halpha", ha_df, HA_REST), ("Hbeta", hb_df, HB_REST)):
        ok = in_nominal(df, z_of, rest)
        bad = df[(df["status"] == "fitted") & ~ok & df["ID"].isin(z_of)]

        if len(bad):
            print(f"  {name} fits beyond the nominal grating range ({len(bad)})"
                  + (", kept (--allow-extended)" if a.allow_extended else ", not used") + ": "
                  + ", ".join(f"{int(i)} {g} ({(z_of[int(i)] + 1) * rest / 1e4:.2f} um)"
                              for i, g in zip(bad["ID"], bad["grating"])))
        if not a.allow_extended:
            df.loc[(df["status"] == "fitted") & ~ok, "status"] = "extended_range"
    ha_rows, hb_rows = line_rows(ha_df, a.ha_snr_min), line_rows(hb_df, a.hb_snr_min)
    isaac, absent = load_isaac(a.isaac, zs["ID"].tolist(), "total")
    if absent:
        print(f"  WARNING not in Isaac's catalogue: {absent}")
    sed = None
    if os.path.exists(a.sed_csv):
        sed = pd.read_csv(a.sed_csv)
        sed["ID"] = sed["ID"].astype(int)
        sed = sed.set_index("ID")
    else:
        print("  WARNING no sed_properties.csv, cases 2 and 3 cannot be dust corrected")

    # [NII]/Halpha for the photometric Halpha
    if a.nii_ha is None:
        h = ha_df[ha_df["status"] == "fitted"].copy()
        for c in ("flux", "flux_snr", "NII6584_flux"):
            h[c] = pd.to_numeric(h[c], errors="coerce") if c in h.columns else np.nan
        h = h[(h["flux_snr"] >= 5) & (h["flux"] > 0) & np.isfinite(h["NII6584_flux"])]
        ratios = (h["NII6584_flux"] / h["flux"]).clip(lower=0)
        nii = float(ratios.median()) if len(ratios) >= 3 else 0.1
        print(f"  [NII]6584/Ha for the photometric Halpha {nii:.3f} "
              f"(median of {len(ratios)} spectral fits with Halpha S/N >= 5)")
    else:
        nii = a.nii_ha
        print(f"  [NII]6584/Ha for the photometric Halpha {nii:.3f} (fixed)")
    print(f"  [SII]/Ha for the photometric Halpha {a.sii_ha:.3f}"
          + ("  (any [SII] in the band is counted as Halpha)" if a.sii_ha == 0 else ""))

    # ---------------------------------------------------------------- pass 1: slit factors
    spec_cache, slit, band_tab, spec_bands = {}, {}, [], {}

    def spec(sid, gr):
        k = (sid, gr)
        if k not in spec_cache:
            try:
                spec_cache[k] = load_spectrum(a.spec_root, sid, gr)
            except (FileNotFoundError, OSError, KeyError):
                spec_cache[k] = None
        return spec_cache[k]

    flagged = {}
    for _, s in zs.iterrows():
        sid, z = int(s["ID"]), float(s["z_sys"])
        prow = isaac.loc[sid] if sid in isaac.index else None
        fl = int(prow["Flag"]) if prow is not None and np.isfinite(prow["Flag"]) else -1
        flagged[sid] = any(c in "123" for c in str(fl)) if fl >= 0 else False
        for line, rows, rest in (("ha", ha_rows, HA_REST), ("hb", hb_rows, HB_REST)):
            for gr, *_ in rows.get(sid, []):
                if prow is None:
                    slit[(sid, line, gr)] = (None, "not in Isaac's catalogue")
                    continue
                if flagged[sid] and not a.allow_flagged_phot:
                    slit[(sid, line, gr)] = (None, f"photometry Flag {fl}")
                    continue
                if (sid, gr) not in spec_bands:
                    good, bad = spectrum_band_factors(spec(sid, gr), gr, prow, curves, a)
                    spec_bands[(sid, gr)] = (good, bad)
                    for b in good:
                        band_tab.append(dict(ID=sid, grating=gr, band=b["band"], pivot_um=b["pivot"] / 1e4,
                                             S=b["factor"], S_err=b["factor_err"], coverage=b["cov"]))
                good, bad = spec_bands[(sid, gr)]
                d_, why_ = line_factor(good, rest * (1 + z), curves, a)
                if d_ is None and bad:
                    why_ = why_ + " (" + "; ".join(bad) + ")"
                slit[(sid, line, gr)] = (d_, why_)
    print(f"  slit mode {a.slit_mode}, band coverage >= {a.min_coverage:g}, S/N >= {a.slit_snr_min:g}, "
          f"S in {a.slit_range[0]:g} to {a.slit_range[1]:g}"
          + (f", median needs >= {a.min_bands} bands" if a.slit_mode == "median" else ""))
    nb = np.array([len(g) for g, _ in spec_bands.values()])
    if nb.size:
        print(f"  spectra with passing bands: {nb.size} spectra, "
              + ", ".join(f"{k} band{'s' if k != 1 else ''}: {int((nb == k).sum())}" for k in range(3))
              + f", 3 or more: {int((nb >= 3).sum())}")
    med = {}
    for line in ("ha", "hb"):
        best = {}
        for (sid, ln, gr), (d, _) in slit.items():
            if ln == line and d is not None:
                best[sid] = max(best.get(sid, d), d, key=lambda x: x["snr"])
        vals = np.array([d["factor"] for d in best.values()])
        if vals.size >= a.min_for_median:
            med[line] = (float(np.median(vals)), nmad(vals))
        else:
            med[line] = (np.nan, np.nan)
        kinds = pd.Series([d["source"] for d in best.values()]).value_counts().to_dict()
        print(f"  {line} slit factors: {vals.size} sources with their own factor {kinds}"
              + (f", median {med[line][0]:.2f}, NMAD {med[line][1]:.2f}, 16-84th "
                 f"{np.percentile(vals, 16):.2f} to {np.percentile(vals, 84):.2f}, "
                 f"range {vals.min():.2f} to {vals.max():.2f}" if vals.size else ""))

    # medians of the own factors of each line in each grating (fallback 2)
    gmed = {}
    for line in ("ha", "hb"):
        for gr in GRATING_FILTER:
            v = np.array([d["factor"] for (sid, ln, g), (d, _) in slit.items()
                          if ln == line and g == gr and d is not None])
            gmed[(line, gr)] = ((float(np.median(v)), nmad(v), v.size) if v.size >= a.min_grating
                                else (np.nan, np.nan, v.size))
    if a.fallback == "grating":
        print(f"  fallback medians by grating (need >= {a.min_grating} own factors):")
        for line in ("ha", "hb"):
            print(f"    {line}: " + ", ".join(
                f"{gr} {m:.2f} +/- {e:.2f} ({k})" if np.isfinite(m) else f"{gr} not used ({k})"
                for gr, (m, e, k) in ((g, gmed[(line, g)]) for g in GRATING_FILTER)))

    # spectrum medians of every grating of a source (fallback 1), filled on demand
    def other_grating_bands(sid, gr):
        prow = isaac.loc[sid] if sid in isaac.index else None
        if prow is None or (flagged.get(sid, False) and not a.allow_flagged_phot):
            return []
        out_ = []
        for g in GRATING_FILTER:
            if g == gr:
                continue
            if (sid, g) not in spec_bands:
                spec_bands[(sid, g)] = spectrum_band_factors(spec(sid, g), g, prow, curves, a)
            good = spec_bands[(sid, g)][0]
            if good:
                out_.append((g, good))
        return out_

    # scatter between two gratings of one source, from sources with factors in both
    rat = []
    for sid in {k[0] for k in spec_bands}:
        meds = [spec_median(spec_bands[(sid, g)][0]) for g in GRATING_FILTER if (sid, g) in spec_bands]
        meds = [m_["factor"] for m_ in meds if m_]
        if len(meds) >= 2:
            rat.append(np.log(meds[1] / meds[0]))
    cross_err = (float(nmad(np.array(rat))) if len(rat) >= 5 else a.cross_grating_err)
    if a.fallback == "grating":
        print(f"  scatter between two gratings of one source: {100 * cross_err:.0f} per cent "
              + (f"(NMAD of ln ratio, {len(rat)} sources)" if len(rat) >= 5 else "(--cross-grating-err)"))

    def factor(sid, line, gr):
        d, why = slit.get((sid, line, gr), (None, "no line"))
        if d is not None:
            return d["factor"], d["factor_err"], d["band"], d["source"], ""
        if a.fallback == "grating":
            og = other_grating_bands(sid, gr)
            if og:
                g, good = max(og, key=lambda t: len(t[1]))
                m_ = spec_median(good)
                e_ = float(np.hypot(m_["factor_err"], cross_err * m_["factor"]))
                return m_["factor"], e_, f"{g}:{m_['band']}", "other_grating", why
            m, e, _ = gmed.get((line, gr), (np.nan, np.nan, 0))
            if np.isfinite(m):
                return m, e, "", "grating_median", why
        m, e = med[line]
        if np.isfinite(m):
            return m, e, "", "sample_median", why
        return np.nan, np.nan, "", "none", why

    def spec_med(sid, gr):
        m = spec_median(spec_bands.get((sid, gr), ([], []))[0])
        return (m["factor"], m["n"]) if m else (np.nan, 0)

    def pick(rows, sid, line):
        """Best spectrum for a line: a direct slit factor first, then S/N."""
        if not rows:
            return None
        return max(rows, key=lambda r: (slit.get((sid, line, r[0]), (None,))[0] is not None, r[3]))

    # ---------------------------------------------------------------- pass 2: per source
    recs, draws_corr, draws_sed = [], [], []
    n = a.n_mc
    for _, s in zs.iterrows():
        sid, z = int(s["ID"]), float(s["z_sys"])
        prow = isaac.loc[sid] if sid in isaac.index else None
        r = dict(ID=sid, z_sys=z, ha_case=0, ha_source="none", ha_reason="", dust_method="none",
                 balmer_spectra="none", isaac_flag=prow["Flag"] if prow is not None else np.nan,
                 phot_flagged=int(flagged.get(sid, False)))
        d_corr = np.full(n, np.nan)
        d_sed = np.full(n, np.nan)
        pdraws, d_f, d_S = None, None, None

        # SED A_Ha
        a_sed, a_sed_lo, a_sed_hi = np.nan, np.nan, np.nan
        if sed is not None and sid in sed.index and int(sed.loc[sid, "sed_ok"]) == 1:
            a_sed = float(sed.loc[sid, "sed_aha"])
            a_sed_lo = float(sed.loc[sid, "sed_aha_err_lo"])
            a_sed_hi = float(sed.loc[sid, "sed_aha_err_hi"])
        r.update(a_ha_sed=a_sed, a_ha_sed_err_lo=a_sed_lo, a_ha_sed_err_hi=a_sed_hi)
        a_sed_draw = (split_normal(a_sed, a_sed_lo, a_sed_hi, n, rng, floor=0.0)
                      if np.isfinite(a_sed) else np.full(n, np.nan))

        # photometric Halpha, for every source (case 3 and a check)
        ph, ph_draw = ({}, None)
        if prow is None:
            r["ha_phot_note"] = "not in Isaac's catalogue"
        elif flagged[sid] and not a.allow_flagged_phot:
            r["ha_phot_note"] = f"photometry Flag {int(prow['Flag'])}"
        else:
            ph, ph_draw = phot_halpha(prow, z, curves, nii, a.sii_ha, a)
            pdraws = ph_draw(n, rng) if ph_draw is not None else None
            if pdraws is not None:
                ph["err"] = float(np.nanstd(pdraws))
                ph["snr"] = ph["flux"] / ph["err"] if ph["err"] > 0 else np.nan
            r.update(ha_phot_flux=ph["flux"], ha_phot_flux_err=ph["err"], ha_phot_snr=ph["snr"],
                     ha_phot_band=ph["band"], ha_phot_cont_bands=ph["cont"],
                     ha_phot_cont_method=ph["method"], ha_phot_note=ph["note"])

        # spectral Halpha
        best_ha = pick(ha_rows.get(sid, []), sid, "ha")
        f_tot = np.nan
        d_tot = None
        if best_ha is not None:
            gr, f, e, snr = best_ha
            S, Se, band, src, why = factor(sid, "ha", gr)
            smed, nband = spec_med(sid, gr)
            r.update(ha_grating=gr, ha_flux_obs=f, ha_flux_obs_err=e, ha_snr=snr,
                     ha_slit_factor=S, ha_slit_factor_err=Se, ha_slit_band=band,
                     ha_slit_source=src, ha_slit_note=why, ha_slit_spec_median=smed,
                     ha_slit_nbands=nband)
            if np.isfinite(S):
                d_f = rng.normal(f, e, n)
                d_S = pos_normal(S, Se, n, rng)
                f_tot, d_tot = f * S, d_f * d_S
                r["ha_source"] = "spec"
            else:
                r["ha_reason"] = f"spectral Halpha but no slit factor ({why})"
        if r["ha_source"] == "none":
            # does the spectrum have Halpha at all?
            fitted = ha_df[(ha_df["ID"] == sid) & (ha_df["status"] == "fitted")]
            low = pd.to_numeric(fitted.get("flux_snr", pd.Series(dtype=float)), errors="coerce")
            ext = ha_df[(ha_df["ID"] == sid) & (ha_df["status"] == "extended_range")]
            why_spec = ("Halpha S/N < %g in every spectrum" % a.ha_snr_min if len(fitted)
                        else "Halpha only beyond the nominal range of " + ",".join(ext["grating"])
                        if len(ext) else "no spectrum covers Halpha")
            if np.isfinite(ph.get("snr", np.nan)) and ph["snr"] >= a.ha_snr_min:
                f_tot, d_tot = ph["flux"], pdraws
                r["ha_source"] = "phot"
                r["ha_reason"] = why_spec
            else:
                r["ha_reason"] = why_spec + (
                    f"; photometric Halpha S/N {ph['snr']:.1f}" if np.isfinite(ph.get("snr", np.nan))
                    else f"; no photometric Halpha ({r.get('ha_phot_note', '') or ph.get('note', '')})")
            if len(fitted) and low.notna().any():
                r["ha_snr_best_below_cut"] = float(low.max())
        r["ha_flux_total"] = f_tot

        # Balmer decrement
        a_bal, d_abal = np.nan, None
        if r["ha_source"] == "spec":
            pairs = []
            hb_by_g = {g: (fl, er, sn) for g, fl, er, sn in hb_rows.get(sid, [])}
            for g, fl, er, sn in ha_rows.get(sid, []):
                if g in hb_by_g:
                    rs = 1 / np.sqrt(1 / sn ** 2 + 1 / hb_by_g[g][2] ** 2)
                    pairs.append((rs, g))
            hb_pick = None
            if pairs:
                g = max(pairs)[1]
                ha_f = next(x for x in ha_rows[sid] if x[0] == g)
                hb_f = (g, *hb_by_g[g])
                r["balmer_spectra"] = "same"
                hb_pick = hb_f
            else:
                cand = pick(hb_rows.get(sid, []), sid, "hb")
                if cand is not None:
                    ha_f, hb_f = best_ha, cand
                    r["balmer_spectra"] = "different"
                    hb_pick = hb_f
            if hb_pick is not None:
                Sa, Sae, _, sa_src, _ = factor(sid, "ha", ha_f[0])
                Sb, Sbe, bandb, sb_src, whyb = factor(sid, "hb", hb_f[0])
                r.update(balmer_ha_grating=ha_f[0], hb_grating=hb_f[0], hb_flux_obs=hb_f[1],
                         hb_flux_obs_err=hb_f[2], hb_snr=hb_f[3], hb_slit_factor=Sb,
                         hb_slit_factor_err=Sbe, hb_slit_band=bandb, hb_slit_source=sb_src)
                R_obs = ha_f[1] / hb_f[1]
                R_cor = (ha_f[1] * Sa) / (hb_f[1] * Sb) if np.isfinite(Sa) and np.isfinite(Sb) else np.nan
                if r["balmer_spectra"] == "same":
                    r["balmer_ratio_obs"] = R_obs
                r["balmer_ratio_slitcorr"] = R_cor
                use_cor = (a.balmer_basis == "slitcorr") or r["balmer_spectra"] == "different"
                R = R_cor if use_cor else R_obs
                if np.isfinite(R):
                    # draws, sharing the Halpha flux draw when it is the same measurement
                    same_meas = ha_f[0] == best_ha[0]
                    dha = d_f if same_meas else rng.normal(ha_f[1], ha_f[2], n)
                    dhb = rng.normal(hb_f[1], hb_f[2], n)
                    if use_cor:
                        dsa = d_S if same_meas else pos_normal(Sa, Sae, n, rng)
                        dR = (dha * dsa) / (dhb * pos_normal(Sb, Sbe, n, rng))
                    else:
                        dR = dha / dhb
                    a_bal, d_abal = float(a_ha_from_ratio(R)), a_ha_from_ratio(dR)
                    ebv = a_bal / K_HA
                    r.update(balmer_ratio=R, ebv_gas=ebv, a_ha_balmer=a_bal)
                    lo, hi = errs(R, dR)
                    r.update(balmer_ratio_err_lo=lo, balmer_ratio_err_hi=hi)
                    lo, hi = errs(a_bal, d_abal)
                    r.update(a_ha_balmer_err_lo=lo, a_ha_balmer_err_hi=hi,
                             ebv_gas_err_lo=lo / K_HA, ebv_gas_err_hi=hi / K_HA)
                else:
                    r["balmer_spectra"] = "none"
                    r["ha_reason"] = f"Hbeta found but no slit factor for it ({whyb})"

        # the case, the A_Ha used and the corrected flux
        if r["ha_source"] == "spec" and np.isfinite(a_bal):
            r.update(ha_case=1, dust_method="balmer", a_ha=a_bal,
                     a_ha_err_lo=r["a_ha_balmer_err_lo"], a_ha_err_hi=r["a_ha_balmer_err_hi"])
            d_a = d_abal
        elif r["ha_source"] in ("spec", "phot") and np.isfinite(a_sed):
            r.update(ha_case=2 if r["ha_source"] == "spec" else 3, dust_method="sed", a_ha=a_sed,
                     a_ha_err_lo=a_sed_lo, a_ha_err_hi=a_sed_hi)
            d_a = a_sed_draw
        else:
            d_a = None
            if r["ha_source"] != "none":
                r["ha_reason"] = (r["ha_reason"] + "; " if r["ha_reason"] else "") + "no A_Ha"
                r["ha_case"] = 0
        if d_a is not None:
            val = f_tot * 10 ** (0.4 * r["a_ha"])
            d_corr = d_tot * 10 ** (0.4 * d_a)
            lo, hi = errs(val, d_corr)
            r.update(ha_flux_corr=val, ha_flux_corr_err_lo=lo, ha_flux_corr_err_hi=hi)
        if d_tot is not None:
            lo, hi = errs(f_tot, d_tot)
            r.update(ha_flux_total_err_lo=lo, ha_flux_total_err_hi=hi)
            if np.isfinite(a_sed):
                val = f_tot * 10 ** (0.4 * a_sed)
                d_sed = d_tot * 10 ** (0.4 * a_sed_draw)
                lo, hi = errs(val, d_sed)
                r.update(ha_flux_corr_sed=val, ha_flux_corr_sed_err_lo=lo,
                         ha_flux_corr_sed_err_hi=hi)
        recs.append(r)
        draws_corr.append(d_corr)
        draws_sed.append(d_sed)

    out = pd.DataFrame(recs)
    cols = ["ID", "z_sys", "ha_case", "ha_source", "dust_method", "balmer_spectra", "ha_reason",
            "ha_grating", "ha_flux_obs", "ha_flux_obs_err", "ha_snr", "ha_slit_factor",
            "ha_slit_factor_err", "ha_slit_band", "ha_slit_source", "ha_slit_note",
            "ha_slit_spec_median", "ha_slit_nbands",
            "hb_grating", "hb_flux_obs", "hb_flux_obs_err", "hb_snr", "hb_slit_factor",
            "hb_slit_factor_err", "hb_slit_band", "hb_slit_source", "balmer_ha_grating",
            "balmer_ratio", "balmer_ratio_err_lo", "balmer_ratio_err_hi", "balmer_ratio_obs",
            "balmer_ratio_slitcorr", "ebv_gas", "ebv_gas_err_lo", "ebv_gas_err_hi",
            "a_ha_balmer", "a_ha_balmer_err_lo", "a_ha_balmer_err_hi",
            "a_ha_sed", "a_ha_sed_err_lo", "a_ha_sed_err_hi", "a_ha", "a_ha_err_lo", "a_ha_err_hi",
            "ha_flux_total", "ha_flux_total_err_lo", "ha_flux_total_err_hi",
            "ha_flux_corr", "ha_flux_corr_err_lo", "ha_flux_corr_err_hi",
            "ha_flux_corr_sed", "ha_flux_corr_sed_err_lo", "ha_flux_corr_sed_err_hi",
            "ha_phot_flux", "ha_phot_flux_err", "ha_phot_snr", "ha_phot_band",
            "ha_phot_cont_bands", "ha_phot_cont_method", "ha_phot_note", "ha_snr_best_below_cut",
            "isaac_flag", "phot_flagged"]
    out = out.reindex(columns=cols)

    report(out, a, nii)
    for p_ in (a.out, a.out_npz, a.bands_out, a.plot):
        os.makedirs(os.path.dirname(os.path.abspath(p_)), exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.DataFrame(band_tab, columns=["ID", "grating", "band", "pivot_um", "S", "S_err", "coverage"]
                 ).to_csv(a.bands_out, index=False)
    np.savez(a.out_npz, ID=out["ID"].to_numpy(int), ha_flux_corr=np.array(draws_corr),
             ha_flux_corr_sed=np.array(draws_sed))
    diagnostic_plot(out, a.plot)
    print(f"\nWrote  {os.path.abspath(a.out)}")
    print(f"Wrote  {os.path.abspath(a.out_npz)}")
    print(f"Wrote  {os.path.abspath(a.bands_out)}")
    print(f"Wrote  {os.path.abspath(a.plot)}")


# ============================================================================
# Report and figure
# ============================================================================

def report(out, a, nii):
    cat = None
    if os.path.exists(pc.LYA_CATALOG):
        cat = pd.read_csv(pc.LYA_CATALOG)
        cat["ID"] = cat["ID"].astype(int)
        cat = cat.set_index("ID")
    o = out.set_index("ID")
    ana = det = pd.Series(True, index=o.index)
    if cat is not None and "in_analysis" in cat.columns:
        ana = cat["in_analysis"].reindex(o.index) == 1
        det = ana & (cat["lya_det"].reindex(o.index) == 1)
    print("\n  cases                                   all   in analysis   Lya det")
    for c, lab in ((1, "1  spec Halpha, Balmer decrement"), (2, "2  spec Halpha, SED A_Ha"),
                   (3, "3  phot Halpha, SED A_Ha"), (0, "0  no Halpha")):
        m = o["ha_case"] == c
        print(f"    {lab:<36s} {int(m.sum()):>4d} {int((m & ana).sum()):>12d} {int((m & det).sum()):>9d}")
    m = o["ha_case"] == 1
    for b in ("same", "different"):
        mm = m & (o["balmer_spectra"] == b)
        print(f"      case 1, Balmer from {b:<10s}          {int(mm.sum()):>4d} "
              f"{int((mm & ana).sum()):>12d} {int((mm & det).sum()):>9d}")
    labels = {"line_band": "band containing the line", "interp": "interpolated, bands either side",
              "spec_median": "spectrum median", "median": "median of all bands",
              "other_grating": "same source, other grating",
              "grating_median": "median of its grating", "sample_median": "sample median"}
    for src, lab in labels.items():
        mm = (o["ha_source"] == "spec") & (o["ha_slit_source"] == src)
        if mm.any():
            print(f"      spec Halpha, S from {lab:<32s} {int(mm.sum()):>3d} "
                  f"({int((mm & det).sum())} Lya det)")
    own = o[o["ha_slit_source"].isin(DIRECT_SOURCES) & (o["ha_slit_nbands"] >= 2)]
    if len(own):
        q = own["ha_slit_factor"] / own["ha_slit_spec_median"]
        print(f"      S used / spectrum median, {len(own)} spectra with >= 2 bands: median {q.median():.2f}, "
              f"16-84th {q.quantile(.16):.2f} to {q.quantile(.84):.2f}")
        off = own[(q > 1.3) | (q < 1 / 1.3)]
        for i, r in off.iterrows():
            print(f"        {i:>6d}  S {r['ha_slit_factor']:.2f} ({r['ha_slit_source']}, {r['ha_slit_band']}) "
                  f"but spectrum median {r['ha_slit_spec_median']:.2f} from {int(r['ha_slit_nbands'])} bands")

    none_det = o[(o["ha_case"] == 0) & det]
    if len(none_det):
        print(f"\n  Lya detections with no Halpha ({len(none_det)})")
        for i, r in none_det.iterrows():
            print(f"    {i:>6d}  z {r['z_sys']:.3f}  {r['ha_reason']}")

    big = o[o["ha_slit_source"].isin(DIRECT_SOURCES) & ((o["ha_slit_factor"] > 3) | (o["ha_slit_factor"] < 0.7))]
    if len(big):
        print(f"\n  Halpha slit factors above 3 or below 0.7 ({len(big)}), worth a look")
        for i, r in big.sort_values("ha_slit_factor").iterrows():
            print(f"    {i:>6d}  {r['ha_grating']}  S {r['ha_slit_factor']:.2f} +/- "
                  f"{r['ha_slit_factor_err']:.2f} in {r['ha_slit_band']}")
    own_ha = o[(o["ha_source"] == "spec") & o["ha_slit_source"].isin(DIRECT_SOURCES)]
    for sub, lab in ((own_ha, "all"), (own_ha[ana.reindex(own_ha.index).fillna(False)], "in analysis")):
        if len(sub) < 3:
            continue
        S = sub["ha_slit_factor"].to_numpy(float)
        print(f"\n  Halpha slit factors by grating, own factors, {lab}: whole sample median {np.median(S):.2f}, "
              f"NMAD {nmad(S):.2f} ({100 * nmad(S) / np.median(S):.0f} per cent)")
        for g, grp in sub.groupby("ha_grating"):
            s = grp["ha_slit_factor"].to_numpy(float)
            print(f"    {g}: median {np.median(s):.2f}, NMAD {nmad(s):.2f} "
                  f"({100 * nmad(s) / np.median(s):.0f} per cent), N {s.size}")
        rel = sub["ha_slit_factor"] / sub["ha_grating"].map(sub.groupby("ha_grating")["ha_slit_factor"].median())
        print(f"    scatter about each grating's own median: {100 * nmad(rel.to_numpy(float)):.0f} per cent "
              f"(compare the whole-sample figure above)")

    b = o[(o["balmer_spectra"] == "same") & o["balmer_ratio_obs"].notna()
          & o["balmer_ratio_slitcorr"].notna()]
    if len(b):
        dA = a_ha_from_ratio(b["balmer_ratio_slitcorr"]) - a_ha_from_ratio(b["balmer_ratio_obs"])
        print(f"\n  Balmer, same spectrum, slit-corrected minus observed ratio ({len(b)} sources): "
              f"ratio median {np.median(b['balmer_ratio_slitcorr'] / b['balmer_ratio_obs']):.3f}x, "
              f"A_Ha median {np.median(dA):+.2f} mag, NMAD {nmad(dA):.2f} mag "
              f"(used {'slit-corrected' if a.balmer_basis == 'slitcorr' else 'observed'})")
    c1 = o[(o["ha_case"] == 1) & o["a_ha_sed"].notna()]
    if len(c1):
        d = c1["a_ha_balmer"] - c1["a_ha_sed"]
        print(f"  case 1, A_Ha Balmer minus SED ({len(c1)}): median {d.median():+.2f}, NMAD "
              f"{nmad(d):.2f} mag. Ha_corr ratio Balmer/SED median "
              f"{np.median(c1['ha_flux_corr'] / c1['ha_flux_corr_sed']):.2f}")
        print(f"    below 2.86 (A_Ha = 0): {int((c1['balmer_ratio'] < R_INT).sum())}, "
              f"A_Ha > 1.5 mag: {c1.index[c1['a_ha_balmer'] > 1.5].tolist()}")
    both = o[(o["ha_source"] == "spec") & (o["ha_phot_snr"] >= a.ha_snr_min)
             & o["ha_flux_total"].notna()]
    if len(both):
        ratio = both["ha_phot_flux"] / both["ha_flux_total"]
        print(f"\n  photometric / spectral total Halpha, where both exist ({len(both)}): median "
              f"{ratio.median():.2f}, NMAD {nmad(ratio):.2f}, 16-84th {ratio.quantile(.16):.2f} "
              f"to {ratio.quantile(.84):.2f}  ([NII]/Ha {nii:.3f}, [SII]/Ha {a.sii_ha:g})")
        for band, grp in both.groupby("ha_phot_band"):
            rr = grp["ha_phot_flux"] / grp["ha_flux_total"]
            print(f"    in {band}: {len(grp)} sources, median {rr.median():.2f}")
    c3 = o[o["ha_case"] == 3]
    if len(c3):
        print(f"\n  case 3 sources ({len(c3)})")
        for i, r in c3.iterrows():
            print(f"    {i:>6d}  z {r['z_sys']:.3f}  {r['ha_phot_band']} with continuum "
                  f"{r['ha_phot_cont_bands']}  S/N {r['ha_phot_snr']:.1f}"
                  f"{'  LYA DET' if det.get(i, False) else ''}")


def _e2(d, col):
    return np.nan_to_num(np.vstack([d[f"{col}_err_lo"].to_numpy(float),
                                    d[f"{col}_err_hi"].to_numpy(float)]))


def diagnostic_plot(out, path):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(2, 2, figsize=(7.09, 6.0))
    d = out[out["ha_slit_source"].isin(DIRECT_SOURCES)]
    v = d["ha_slit_factor"].to_numpy(float)
    if v.size:
        ax[0, 0].hist(v, bins=np.linspace(0, max(4, np.nanmax(v) * 1.05), 25), color="#8c4a8c",
                      edgecolor="k", lw=0.5)
        ax[0, 0].axvline(np.median(v), color="k", ls="--", lw=0.8)
    ax[0, 0].set_xlabel(r"H$\alpha$ slit factor $S$")
    ax[0, 0].set_ylabel("N")
    ax[0, 0].set_title(f"own factors (not the sample median), N = {v.size}", fontsize=8)

    c = out[out["a_ha_balmer"].notna() & out["a_ha_sed"].notna()]
    a_ = ax[0, 1]
    for b, mk in (("same", "o"), ("different", "s")):
        cc = c[c["balmer_spectra"] == b]
        a_.errorbar(cc["a_ha_sed"].to_numpy(float), cc["a_ha_balmer"].to_numpy(float),
                    xerr=_e2(cc, "a_ha_sed"), yerr=_e2(cc, "a_ha_balmer"), fmt=mk, ms=4,
                    mfc="#2a9d8f" if b == "same" else "white", mec="k", ecolor="0.6", lw=0.6,
                    label=f"{b} spectrum ({len(cc)})")
    lim = max(1.0, float(np.nanmax(np.concatenate([[1], c["a_ha_balmer"], c["a_ha_sed"]]))) * 1.1)
    a_.plot([0, lim], [0, lim], "k--", lw=0.8)
    a_.set_xlim(-0.05, lim)
    a_.set_ylim(-0.05, lim)
    a_.set_xlabel(r"$A_{\rm H\alpha}$ SED [mag]")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ Balmer decrement [mag]")
    a_.legend(frameon=False, fontsize=6)

    b = out[(out["ha_source"] == "spec") & out["ha_phot_flux"].notna()
            & out["ha_flux_total"].notna()]
    a_ = ax[1, 0]
    if len(b):
        x = b["ha_flux_total"].to_numpy(float)
        rat = b["ha_phot_flux"].to_numpy(float) / x
        rerr = np.abs(rat) * np.hypot(np.nan_to_num(b["ha_phot_flux_err"].to_numpy(float)) /
                                      np.abs(b["ha_phot_flux"].to_numpy(float)),
                                      0.5 * (b["ha_flux_total_err_lo"] + b["ha_flux_total_err_hi"]
                                             ).to_numpy(float) / x)
        a_.errorbar(x, rat, yerr=np.nan_to_num(rerr), fmt="o", ms=3, mfc="#e39b3b", mec="k",
                    ecolor="0.6", lw=0.6)
        a_.axhline(1, color="k", ls="--", lw=0.8)
        a_.set_xscale("log")
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.set_ylim(min(0, np.nanmin(rat) - 0.2), max(2, np.nanmax(rat) + 0.2))
    a_.set_xlabel(r"H$\alpha$ spectral, total [erg s$^{-1}$ cm$^{-2}$]")
    a_.set_ylabel(r"H$\alpha$ photometric / spectral")
    a_.set_title(f"N = {len(b)}", fontsize=8)

    a_ = ax[1, 1]
    for case, col in ((1, "#2a9d8f"), (2, "#3b6fb6"), (3, "#e39b3b")):
        cc = out[out["ha_case"] == case]
        a_.scatter(cc["z_sys"], cc["a_ha"], s=14, color=col, edgecolor="k", lw=0.4,
                   label=f"case {case} ({len(cc)})")
    a_.set_xlabel(r"$z_{\rm sys}$")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ used [mag]")
    a_.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()