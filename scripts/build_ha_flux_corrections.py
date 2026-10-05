#!/usr/bin/env python
"""Slit-loss and Balmer-decrement dust corrections for the Halpha fluxes.

Builds one row per source with the systemic redshift, the observed Halpha and
Hbeta fluxes from the LiMe fits, slit-loss corrected fluxes for both lines,
and the fully corrected (slit loss + dust) Halpha flux for sources with a
usable Hbeta. Sources without Hbeta keep the slit-corrected Halpha only. The
SED-based dust fallback for them will be added later.

Sample
------
From grating_sources_with_zsys.csv: rows with in_muse true and a finite
z_sys. Duplicates are NOT collapsed. The duplicate column is carried through
(0 = no pair, both members of a pair share the same nonzero number) so they
can be collapsed later.

Uncorrected fluxes
------------------
From Ha_summary_by_JELS_ID.csv and Hbeta_summary_by_JELS_ID.csv, written by
lime_lines_jointfit.py. Halpha is the [NII]-deblended flux. Units erg/s/cm2.
Grating choice per source:
  1. If one or more gratings have both Halpha and Hbeta, use the one with the
     highest S/N on the ratio, 1/sqrt(1/SN_Ha^2 + 1/SN_Hb^2). The flux
     calibration of one spectrum then largely cancels in the decrement.
  2. If that same-grating Hbeta is below HB_SNR_MIN but another grating has
     Hbeta at or above it, take each line from its own best-S/N grating.
  3. Otherwise each line comes from its own best-S/N grating.
grating_ha, grating_hb and same_grating record what was used.

Slit-loss correction
--------------------
Two modes, chosen with --slit-mode.
  dja: no extra correction. The msaexp path-loss
      correction already in the DJA spectra is taken as sufficient, so the
      slit factor is 1 with zero error and slit_source = "dja". Agreed with
      supervisor while the meaning of the _corr photometry is confirmed.
  phot (default): an extra correction from photometry, described below.
For each line in phot mode, in the spectrum that line's flux came from:
  1. Candidate filters are the NIRCam filters with photometry for the source
     whose transmission at the observed line wavelength is at least half the
     filter peak (the half-power definition JDox uses for lambda-/lambda+).
  2. The line itself must fall on the spectrum, and at least MIN_COVERAGE of
     the filter (weighted by T c/lambda, as in the synthetic flux, above
     COVER_T_FLOOR of peak) must lie on the spectrum. Gaps up to GAP_TOL_PIX
     median pixels (masked pixels) are interpolated over. Filter wings often
     run past the grating edge or into the H-grating detector gap, so full
     coverage is rare.
  3. Synthetic photometry, photon-counting convention:
        <f_nu> = int f_lambda T lambda dlambda / int T (c/lambda) dlambda
     computed on the spectrum's own pixels where covered, with errors from
     the pixel errors assumed independent. Any uncovered wing is filled with
     the median f_nu of the spectrum pixels inside the filter, which a
     narrow emission line barely moves. On synthetic tests with a strong line
     this recovers the full-filter flux to ~1% at 75% coverage.
  4. factor = f_nu,phot / f_nu,synth. Both must have S/N >= SLIT_SNR_MIN.
  5. Among passing filters a narrow band is preferred, then a medium band,
     then a wide band, then the highest factor S/N.
  6. If no filter passes, the factor is the median of the direct factors for
     that line across the sample, with the NMAD scatter of those factors as
     its error (needs at least MIN_FOR_MEDIAN direct factors).
  flux_slitcorr = flux_uncorr x factor, errors combined in quadrature.

Balmer decrement dust correction (Monte Carlo)
----------------------------------------------
Only for sources with Halpha and Hbeta flux S/N >= HB_SNR_MIN (both lines).
Ratio basis (balmer_basis column, --balmer-basis):
  observed_same_grating  both lines from one spectrum. The observed ratio is
      used and the slit factors cancel. The per-filter slit factors carry
      photometric noise and systematics (e.g. duplicate IDs of one galaxy
      gave Hbeta factors of 0.32 and 0.92 from the same spectrum), which
      scrambled the ratio when applied to each line separately. The msaexp
      path-loss correction handles the wavelength dependence of the point-
      source loss between the two lines.
  slitcorr  lines from different gratings, so each is slit-corrected first.
Each draw samples the observed fluxes and the slit factors from Gaussians,
then
  R        = Halpha / Hbeta (observed or slit-corrected, as above)
  E(B-V)   = 2.5 / (k(Hb) - k(Ha)) log10(R / R_INT), set to 0 when R < R_INT
  A_Ha     = k(Ha) E(B-V)
  F_full   = F_Ha,obs x slit_factor_Ha x 10^(0.4 A_Ha)
Draws with a non-positive flux, factor or ratio are discarded (the fraction
is printed). The whole decrement is also always run with slit factor 1 for both lines,
giving the *_dja columns (trust the msaexp path-loss correction as it is),
so the photometric slit-loss correction can be compared directly.
Reported values are the 50th percentile, with err_lo and err_hi
being the 50-16 and 84-50 percentile differences.
k(lambda) is computed from the Calzetti et al. (2000) formula, R_V = 4.05,
the same curve Paper 1 used. R_INT = 2.86, Case B at T = 1e4 K,
n_e = 100 cm^-3 (Osterbrock & Ferland 2006).

Photometry (the flux the spectrum is corrected to)
--------------------------------------------------
The slit factor corrects the spectrum to whatever the comparison photometry
measures, so that choice decides how close to total the result is. All of it
is set on the command line (--phot-cats, --phot-base, the suffix options and
--flux-unit-jy), so a total-flux catalogue can be swapped in later with no
code changes. The column stem used is written to slit_reference.
Default: NIRCam_<FILT>_APER_600_mas_flux_corr and _fluxerr_corr from the four
JELS_F356W_DJA_*_match_0p3as.fits catalogues (Corey's photometry), as in
fit_muv_beta.py. Per Ken (Oct 2026), _corr is most likely a Milky Way
extinction correction, NOT an aperture-to-total correction. These fluxes
therefore correct the spectrum to 0.6 arcsec aperture flux, not total. A
larger aperture from the same catalogue gets closer to total as a stand-in
until total-corrected fluxes are available. The flux unit is inferred from
the magnitude-column zero point unless --flux-unit-jy is given. The first
catalogue containing the ID is used.

Filter curves
-------------
SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012; Rodrigo & Solano
2020), ASCII format, wavelength in Angstrom and transmission, e.g.
  http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=JWST/NIRCam.F444W
Download once with --download-filters, which writes
<FILTER_DIR>/JWST_NIRCam.<FILT>.dat and then exits.

Usage
-----
python build_ha_flux_corrections.py --download-filters   # once
python build_ha_flux_corrections.py                       # 0.6 arcsec _corr photometry
python build_ha_flux_corrections.py --phot-base 'NIRCam_{filt}_APER_2_as' \
    --out /ceph/cephfs/apatrick/P2/jwst_catalogs/ha_hb_flux_corrections_2as.csv  # 2 arcsec
python build_ha_flux_corrections.py --slit-mode dja       # DJA path loss only
"""

import os
import sys
import argparse
import urllib.request

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"
SPEC_ROOT = f"{P2}/jwst_spectra"
ZSYS_CSV = f"{CAT_DIR}/grating_sources_with_zsys.csv"
HA_SUMMARY = f"{CAT_DIR}/Ha_summary_by_JELS_ID.csv"
HB_SUMMARY = f"{CAT_DIR}/Hbeta_summary_by_JELS_ID.csv"
FILTER_DIR = f"{CAT_DIR}/nircam_filters"
OUT_CSV = f"{CAT_DIR}/ha_hb_flux_corrections.csv"

PHOT_CATALOGUES = [
    f"{CAT_DIR}/JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
]
# Comparison photometry. Set from the command line in main(), so a different
# catalogue (e.g. total fluxes) can be swapped in without editing the code.
PHOT = {
    "cats": PHOT_CATALOGUES,
    "id_col": "ID",
    "base": "NIRCam_{filt}_APER_600_mas",
    "flux": "_flux_corr",
    "err": "_fluxerr_corr",
    "mag": "_mag_corr",
    "unit_jy": None,       # None = infer from the matching magnitude column
}

GRATINGS_FULL = {"G235M": "G235M_F170LP", "G235H": "G235H_F170LP",
                 "G395M": "G395M_F290LP", "G395H": "G395H_F290LP"}

# NIRCam filters in the photometric catalogues that the G235/G395 gratings
# can cover. The bluer NIRCam filters lie below the G235 range.
SLIT_FILTERS = ["F200W", "F277W", "F356W", "F410M", "F444W", "F466N", "F470N"]
SVO_URL = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=JWST/NIRCam.{filt}"

# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
C_ANG = 2.99792458e18          # speed of light, Angstrom / s
HA_REST = 6564.632             # vacuum, Angstrom, as lime_lines_jointfit.py
HB_REST = 4862.683
R_INT = 2.86                   # Case B Ha/Hb, T = 1e4 K, n_e = 100 cm^-3
RV_CALZETTI = 4.05             # Calzetti et al. (2000)

# ----------------------------------------------------------------------------
# Choices (all adjustable)
# ----------------------------------------------------------------------------
HB_SNR_MIN = 3.0          # flux S/N cut on Hbeta (and Halpha) for the decrement
SLIT_SNR_MIN = 3.0        # S/N cut on synthetic and photometric fluxes
MIN_COVERAGE = 0.70       # min fraction of the filter (photon weighted) on the spectrum
COVER_T_FLOOR = 0.01      # filter curve below this fraction of peak is ignored
GAP_TOL_PIX = 5           # gaps up to this many median pixels are interpolated over
BALMER_BASIS = "auto"     # auto: same grating -> observed ratio, else slit-corrected
MIN_FOR_MEDIAN = 3        # min direct factors needed for the sample-median fallback
N_MC = 10000
MC_SEED = 42

ZP_TO_JY = {23.9: 1e-6, 31.4: 1e-9, 8.9: 1.0, 16.4: 1e-3}


# ----------------------------------------------------------------------------
# Filter curves
# ----------------------------------------------------------------------------
def filter_path(filt):
    return os.path.join(FILTER_DIR, f"JWST_NIRCam.{filt}.dat")


def download_filters():
    os.makedirs(FILTER_DIR, exist_ok=True)
    for filt in SLIT_FILTERS:
        url = SVO_URL.format(filt=filt)
        out = filter_path(filt)
        print(f"  {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
        rows = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
        if len(rows) < 10:
            raise RuntimeError(f"Download for {filt} looks empty, got {len(rows)} rows")
        with open(out, "w") as fh:
            fh.write(f"# SVO FPS JWST/NIRCam.{filt}\n# {url}\n# wavelength_AA transmission\n")
            fh.write("\n".join(rows) + "\n")
        print(f"    -> {os.path.abspath(out)}  ({len(rows)} rows)")


def load_filters():
    curves = {}
    for filt in SLIT_FILTERS:
        p = filter_path(filt)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing. Run with --download-filters first.")
        d = np.loadtxt(p)
        order = np.argsort(d[:, 0])
        curves[filt] = (d[order, 0], d[order, 1])
    return curves


def filter_class_rank(filt):
    """Narrow (N) before medium (M) before wide (W)."""
    return {"N": 0, "M": 1, "W": 2}[filt[-1]]


# ----------------------------------------------------------------------------
# Attenuation
# ----------------------------------------------------------------------------
def k_calzetti(lam_aa):
    """Calzetti et al. (2000) k(lambda), lambda in Angstrom, valid 0.12-2.2 um."""
    x = lam_aa / 1e4
    if 0.63 <= x <= 2.20:
        return 2.659 * (-1.857 + 1.040 / x) + RV_CALZETTI
    if 0.12 <= x < 0.63:
        return 2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + RV_CALZETTI
    raise ValueError(f"Calzetti curve undefined at {lam_aa} AA")


K_HA = k_calzetti(HA_REST)
K_HB = k_calzetti(HB_REST)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def truthy(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def load_sample():
    df = pd.read_csv(ZSYS_CSV)
    n0 = len(df)
    df = df[truthy(df["in_muse"])]
    n1 = len(df)
    df = df[np.isfinite(pd.to_numeric(df["z_sys"], errors="coerce"))]
    print(f"  {n0} rows, {n1} in_muse, {len(df)} with z_sys")
    n_dup = int((pd.to_numeric(df["duplicate"], errors="coerce").fillna(0) > 0).sum())
    print(f"  of which {n_dup} are members of duplicate pairs (kept, not collapsed)")
    keep = ["ID", "grating", "z_dja", "z_sys", "z_sys_err", "duplicate"]
    out = df[keep].copy()
    out["ID"] = out["ID"].astype(int)
    return out.reset_index(drop=True)


def load_summary(path):
    s = pd.read_csv(path)
    s["ID"] = s["ID"].astype(int)
    return s.set_index("ID")


def line_fluxes(summ, src_id, name):
    """{grating: (flux, err, snr)} for gratings with a finite flux."""
    out = {}
    if src_id not in summ.index:
        return out
    row = summ.loc[src_id]
    for gr in GRATINGS_FULL:
        f = row.get(f"{name}_{gr}_flux", np.nan)
        e = row.get(f"{name}_{gr}_flux_err", np.nan)
        if pd.notna(f) and pd.notna(e) and np.isfinite(f) and np.isfinite(e) and e > 0:
            out[gr] = (float(f), float(e), float(f) / float(e))
    return out


def choose_gratings(ha, hb):
    """Return (grating_ha, grating_hb). Either may be None."""
    best = lambda d: max(d, key=lambda g: d[g][2]) if d else None
    g_ha_best, g_hb_best = best(ha), best(hb)
    common = [g for g in ha if g in hb]
    if common:
        ratio_snr = {g: 1.0 / np.sqrt(1.0 / ha[g][2] ** 2 + 1.0 / hb[g][2] ** 2)
                     if ha[g][2] > 0 and hb[g][2] > 0 else -np.inf for g in common}
        g = max(ratio_snr, key=ratio_snr.get)
        if hb[g][2] >= HB_SNR_MIN or hb[g_hb_best][2] < HB_SNR_MIN:
            return g, g
    return g_ha_best, g_hb_best


def flux_unit_to_jy(tab, fcol, mcol):
    f = np.asarray(tab[fcol], dtype=float)
    m = np.asarray(tab[mcol], dtype=float)
    ok = np.isfinite(f) & np.isfinite(m) & (f > 0) & (m > 0) & (m < 50)
    if ok.sum() == 0:
        return None
    zp = np.median(m[ok] + 2.5 * np.log10(f[ok]))
    for ref, scale in ZP_TO_JY.items():
        if abs(zp - ref) < 0.3:
            return scale
    raise ValueError(f"Could not identify flux unit of {fcol}: zero point {zp:.3f}")


def phot_cols(filt):
    base = PHOT["base"].format(filt=filt)
    return base + PHOT["flux"], base + PHOT["err"], base + PHOT["mag"]


def load_photometry():
    cats = []
    for path in PHOT["cats"]:
        tab = Table.read(path).to_pandas()
        tab[PHOT["id_col"]] = tab[PHOT["id_col"]].astype(int)
        tab = tab.drop_duplicates(subset=PHOT["id_col"]).set_index(PHOT["id_col"])
        scales = {}
        for filt in SLIT_FILTERS:
            fcol, ecol, mcol = phot_cols(filt)
            if fcol not in tab.columns or ecol not in tab.columns:
                continue
            if PHOT["unit_jy"] is not None:
                scales[filt] = PHOT["unit_jy"]
            elif mcol in tab.columns:
                scales[filt] = flux_unit_to_jy(tab, fcol, mcol)
            else:
                scales[filt] = None
        print(f"  read {os.path.abspath(path)}  ({len(tab)} IDs, "
              f"filters {', '.join(f for f, s in scales.items() if s) or 'NONE'})")
        cats.append((tab, scales))
    for _, scales in cats:   # fill units not inferable in one catalogue
        for filt, s in scales.items():
            if s is None:
                others = [c[1].get(filt) for c in cats if c[1].get(filt)]
                scales[filt] = others[0] if others else None
    if not any(s for _, sc in cats for s in sc.values()):
        raise ValueError(f"No photometry columns matching {phot_cols('<FILT>')} found. "
                         "Check --phot-base and the suffix options.")
    return cats


def source_photometry(cats, src_id):
    """{filt: (f_nu Jy, err Jy)} from the first catalogue holding the ID."""
    for tab, scales in cats:
        if src_id not in tab.index:
            continue
        row = tab.loc[src_id]
        out = {}
        for filt, s in scales.items():
            if not s:
                continue
            fcol, ecol, _ = phot_cols(filt)
            f, e = float(row[fcol]), float(row[ecol])
            if np.isfinite(f) and np.isfinite(e) and e > 0:
                out[filt] = (f * s, e * s)
        return out
    return {}


def spectrum_path(src_id, gr):
    full = GRATINGS_FULL[gr]
    return os.path.join(SPEC_ROOT, full, f"{src_id}_{full}_spectra_lime.fits")


def load_spectrum(src_id, gr):
    with fits.open(spectrum_path(src_id, gr)) as h:
        t = h["SPECTRUM"].data
        w, f, e = (np.asarray(t[c], dtype=float) for c in ("WAVE", "FLUX", "ERR"))
    ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    order = np.argsort(w[ok])
    return w[ok][order], f[ok][order], e[ok][order]


# ----------------------------------------------------------------------------
# Slit loss
# ----------------------------------------------------------------------------
def _trapz(y, x):
    return (getattr(np, "trapezoid", None) or np.trapz)(y, x)


def covered_mask(w, x):
    """True where wavelength x falls on the spectrum, allowing small gaps."""
    dw = np.median(np.diff(w))
    i = np.searchsorted(w, x)
    inside = (i > 0) & (i < len(w))
    gap = np.full(np.size(x), np.inf)
    gap[inside] = w[i[inside]] - w[i[inside] - 1]
    return inside & (gap <= GAP_TOL_PIX * dw)


def coverage(w, lam_f, t_f):
    """Fraction of the filter's photon-counting weight, T c/lambda, on the spectrum."""
    keep = t_f >= COVER_T_FLOOR * t_f.max()
    lf, tf = lam_f[keep], t_f[keep]
    wt = tf * C_ANG / lf
    return _trapz(wt * covered_mask(w, lf), lf) / _trapz(wt, lf)


def synthetic_fnu(w, f, e, lam_f, t_f):
    """Photon-counting synthetic f_nu (Jy) and error from an f_lambda spectrum.

    The part of the filter the spectrum covers is integrated directly. Any
    uncovered filter wing is filled with the median f_nu of the spectrum
    pixels inside the filter (a continuum level that a narrow emission line
    barely moves), weighted by the uncovered fraction of the filter.
    """
    t = np.interp(w, lam_f, t_f, left=0.0, right=0.0)
    on = t >= COVER_T_FLOOR * t_f.max()
    dl = np.gradient(w)
    den = np.sum(t * C_ANG / w * dl)
    fnu_cov = np.sum(f * t * w * dl) / den
    err_cov = np.sqrt(np.sum((e * t * w * dl) ** 2)) / den
    cov = coverage(w, lam_f, t_f)
    if cov < 1.0:
        fnu_pix = f[on] * w[on] ** 2 / C_ANG
        enu_pix = e[on] * w[on] ** 2 / C_ANG
        fill = np.median(fnu_pix)
        fill_err = 1.253 * np.median(enu_pix) / np.sqrt(on.sum())
    else:
        fill, fill_err = 0.0, 0.0
    fnu = cov * fnu_cov + (1 - cov) * fill
    err = np.hypot(cov * err_cov, (1 - cov) * fill_err)
    return fnu / 1e-23, err / 1e-23


def direct_slit_factor(src_id, gr, lam_line, phot, curves):
    """Best direct factor for one line. Returns dict, or None if no filter passes."""
    try:
        w, f, e = load_spectrum(src_id, gr)
    except (FileNotFoundError, OSError, KeyError):
        return None, "no spectrum"
    cands, reasons = [], []
    for filt in SLIT_FILTERS:
        if filt not in phot:
            continue
        lam_f, t_f = curves[filt]
        if np.interp(lam_line, lam_f, t_f, left=0, right=0) < 0.5 * t_f.max():
            continue
        if not covered_mask(w, np.array([lam_line]))[0]:
            reasons.append(f"{filt} line in spectrum gap")
            continue
        cov = coverage(w, lam_f, t_f)
        if cov < MIN_COVERAGE:
            reasons.append(f"{filt} cov {cov:.2f}")
            continue
        fs, es = synthetic_fnu(w, f, e, lam_f, t_f)
        fp, ep = phot[filt]
        if not (fs / es >= SLIT_SNR_MIN and fp / ep >= SLIT_SNR_MIN):
            reasons.append(f"{filt} S/N synth {fs/es:.1f} phot {fp/ep:.1f}")
            continue
        fac = fp / fs
        fac_e = fac * np.sqrt((ep / fp) ** 2 + (es / fs) ** 2)
        cands.append(dict(filt=filt, factor=fac, factor_err=fac_e,
                          snr=fac / fac_e, synth=fs, phot=fp, cov=cov))
    if not cands:
        return None, "; ".join(reasons) if reasons else "no filter contains line"
    best = min(cands, key=lambda c: (filter_class_rank(c["filt"]), -c["snr"]))
    return best, ""


# ----------------------------------------------------------------------------
# Balmer MC
# ----------------------------------------------------------------------------
def balmer_mc(fha, eha, fhb, ehb, sha, esha, shb, eshb, use_slit_in_ratio, rng):
    """Monte Carlo Balmer decrement.

    The fully corrected Halpha is always slit-loss corrected. The ratio uses
    slit-corrected fluxes only when use_slit_in_ratio is True (lines from
    different gratings). Otherwise the observed ratio from one spectrum is
    used and the slit factors cancel.
    """
    ha_obs = rng.normal(fha, eha, N_MC)
    hb_obs = rng.normal(fhb, ehb, N_MC)
    s_ha = rng.normal(sha, esha, N_MC)
    if use_slit_in_ratio:
        ratio = (ha_obs * s_ha) / (hb_obs * rng.normal(shb, eshb, N_MC))
    else:
        ratio = ha_obs / hb_obs
    good = (ha_obs > 0) & (hb_obs > 0) & (s_ha > 0) & (ratio > 0)
    ratio, ha_corr = ratio[good], (ha_obs * s_ha)[good]
    ebv = np.clip(2.5 / (K_HB - K_HA) * np.log10(ratio / R_INT), 0.0, None)
    a_ha = K_HA * ebv
    full = ha_corr * 10 ** (0.4 * a_ha)

    def pct(x):
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        return p50, p50 - p16, p84 - p50

    return dict(ratio=pct(ratio), ebv=pct(ebv), a_ha=pct(a_ha),
                full=pct(full), frac_dropped=1 - good.mean())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global OUT_CSV
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download-filters", action="store_true",
                   help="Download the NIRCam filter curves from SVO and exit.")
    p.add_argument("--slit-mode", choices=["dja", "phot"], default="phot",
                   help="phot (default): correct the spectrum to the comparison photometry "
                        "by filter convolution. dja: trust the msaexp path-loss correction "
                        "already in the DJA spectra, slit factor 1.")
    p.add_argument("--phot-cats", nargs="+", default=PHOT_CATALOGUES,
                   help="Photometric catalogue(s) to compare against. The first one "
                        "containing the ID is used. Default: the four JELS_F356W_DJA "
                        "catalogues.")
    p.add_argument("--phot-id-col", default=PHOT["id_col"],
                   help="ID column in the photometric catalogues. Default ID.")
    p.add_argument("--phot-base", default=PHOT["base"],
                   help="Column name stem with {filt} for the filter, e.g. "
                        "'NIRCam_{filt}_APER_600_mas' (default) or a larger aperture.")
    p.add_argument("--flux-suffix", default=PHOT["flux"])
    p.add_argument("--err-suffix", default=PHOT["err"])
    p.add_argument("--mag-suffix", default=PHOT["mag"],
                   help="Used only to infer the flux unit.")
    p.add_argument("--flux-unit-jy", type=float, default=None,
                   help="Flux unit in Jy (e.g. 1e-9 for nJy). Set this if the catalogue "
                        "has no magnitude columns. Default: infer from magnitudes.")
    p.add_argument("--balmer-basis", choices=["auto", "slitcorr"], default=BALMER_BASIS,
                   help="auto (default): Balmer ratio from the observed fluxes when both "
                        "lines come from one grating, slit-corrected otherwise. slitcorr: "
                        "always slit-corrected.")
    p.add_argument("--out", default=OUT_CSV,
                   help=f"Output CSV. Default {OUT_CSV}")
    args = p.parse_args()
    OUT_CSV = args.out
    PHOT.update(cats=args.phot_cats, id_col=args.phot_id_col, base=args.phot_base,
                flux=args.flux_suffix, err=args.err_suffix, mag=args.mag_suffix,
                unit_jy=args.flux_unit_jy)

    print("build_ha_flux_corrections.py")
    if args.download_filters:
        print(f"Downloading NIRCam filter curves to {os.path.abspath(FILTER_DIR)}")
        download_filters()
        return

    print(f"  sample        {os.path.abspath(ZSYS_CSV)}")
    print(f"  Ha fluxes     {os.path.abspath(HA_SUMMARY)}")
    print(f"  Hb fluxes     {os.path.abspath(HB_SUMMARY)}")
    print(f"  slit mode     {args.slit_mode}")
    if args.slit_mode == "phot":
        print(f"  spectra       {os.path.abspath(SPEC_ROOT)}/<grating>/")
        print(f"  filters       {os.path.abspath(FILTER_DIR)}")
        print(f"  phot columns  {phot_cols('<FILT>')[0]}, {phot_cols('<FILT>')[1]}")
    print(f"  output        {os.path.abspath(OUT_CSV)}")
    print(f"  k(Ha) {K_HA:.3f}  k(Hb) {K_HB:.3f}  (Calzetti 2000, R_V {RV_CALZETTI})")

    sample = load_sample()
    ha_s, hb_s = load_summary(HA_SUMMARY), load_summary(HB_SUMMARY)
    curves, cats = None, None
    if args.slit_mode == "phot":
        curves = load_filters()
        cats = load_photometry()

    rows = []
    for _, src in sample.iterrows():
        sid, z = int(src["ID"]), float(src["z_sys"])
        r = src.to_dict()
        ha, hb = line_fluxes(ha_s, sid, "Ha"), line_fluxes(hb_s, sid, "Hbeta")
        g_ha, g_hb = choose_gratings(ha, hb)
        r["grating_ha"], r["grating_hb"] = g_ha, g_hb
        r["same_grating"] = bool(g_ha and g_hb and g_ha == g_hb)
        for pre, d, g in (("ha", ha, g_ha), ("hb", hb, g_hb)):
            f, e, s = d[g] if g else (np.nan, np.nan, np.nan)
            r[f"{pre}_flux_uncorr"], r[f"{pre}_flux_uncorr_err"], r[f"{pre}_flux_uncorr_snr"] = f, e, s

        phot = source_photometry(cats, sid) if args.slit_mode == "phot" else {}
        for pre, g, rest in (("ha", g_ha, HA_REST), ("hb", g_hb, HB_REST)):
            r[f"{pre}_slit_filter"] = None
            r[f"{pre}_slit_factor"] = np.nan
            r[f"{pre}_slit_factor_err"] = np.nan
            r[f"{pre}_slit_source"] = None
            r[f"{pre}_slit_note"] = ""
            r[f"{pre}_slit_coverage"] = np.nan
            if g is None:
                continue
            if args.slit_mode == "dja":
                # Trust the msaexp path-loss correction already in the DJA spectra.
                r[f"{pre}_slit_factor"] = 1.0
                r[f"{pre}_slit_factor_err"] = 0.0
                r[f"{pre}_slit_source"] = "dja"
                continue
            best, why = direct_slit_factor(sid, g, rest * (1 + z), phot, curves)
            if best is not None:
                r[f"{pre}_slit_filter"] = best["filt"]
                r[f"{pre}_slit_factor"] = best["factor"]
                r[f"{pre}_slit_factor_err"] = best["factor_err"]
                r[f"{pre}_slit_source"] = "direct"
                r[f"{pre}_slit_coverage"] = best["cov"]
            else:
                r[f"{pre}_slit_note"] = why
        rows.append(r)

    out = pd.DataFrame(rows)
    out["slit_reference"] = (f"{phot_cols('<FILT>')[0]}" if args.slit_mode == "phot"
                             else "dja_pathloss_only")

    # ---- sample-median fallback ----
    for pre in ("ha", "hb"):
        direct = out.loc[out[f"{pre}_slit_source"] == "direct", f"{pre}_slit_factor"].to_numpy()
        need = out[f"{pre}_flux_uncorr"].notna() & out[f"{pre}_slit_source"].isna()
        if args.slit_mode == "dja":
            pass
        elif len(direct) >= MIN_FOR_MEDIAN:
            med = np.median(direct)
            nmad = 1.4826 * np.median(np.abs(direct - med))
            out.loc[need, f"{pre}_slit_factor"] = med
            out.loc[need, f"{pre}_slit_factor_err"] = nmad
            out.loc[need, f"{pre}_slit_source"] = "sample_median"
            print(f"  {pre} slit factor: {len(direct)} direct, median {med:.3f}, "
                  f"NMAD {nmad:.3f}, applied to {int(need.sum())} sources")
        else:
            print(f"  {pre} slit factor: only {len(direct)} direct, no median fallback")

        f, fe = out[f"{pre}_flux_uncorr"], out[f"{pre}_flux_uncorr_err"]
        s, se = out[f"{pre}_slit_factor"], out[f"{pre}_slit_factor_err"]
        out[f"{pre}_flux_slitcorr"] = f * s
        out[f"{pre}_flux_slitcorr_err"] = np.abs(f * s) * np.sqrt((fe / f) ** 2 + (se / s) ** 2)

    # ---- Balmer decrement ----
    # Two versions are always computed.
    #   main: Halpha slit-loss corrected to the comparison photometry.
    #   _dja: slit factor 1 for both lines, i.e. trust the msaexp path-loss
    #         correction in the DJA spectra as it stands.
    rng = np.random.default_rng(MC_SEED)
    cols = ["balmer_ratio", "ebv_neb", "A_ha", "ha_flux_fullcorr"]
    keys = ("ratio", "ebv", "a_ha", "full")
    for c in cols:
        for v in ("", "_dja"):
            for suff in ("", "_err_lo", "_err_hi"):
                out[c + v + suff] = np.nan
    out["dust_method"] = None
    out["balmer_basis"] = None
    for i, r in out.iterrows():
        obs = [r["ha_flux_uncorr"], r["ha_flux_uncorr_err"], r["hb_flux_uncorr"], r["hb_flux_uncorr_err"]]
        if not all(np.isfinite(v) for v in obs):
            continue
        if r["ha_flux_uncorr_snr"] < HB_SNR_MIN or r["hb_flux_uncorr_snr"] < HB_SNR_MIN:
            continue
        out.loc[i, "dust_method"] = "balmer"

        # DJA-only version: no extra slit factor on either line
        mc = balmer_mc(*obs, 1.0, 0.0, 1.0, 0.0, False, rng)
        for c, key in zip(cols, keys):
            out.loc[i, c + "_dja"], out.loc[i, c + "_dja_err_lo"], out.loc[i, c + "_dja_err_hi"] = mc[key]

        # Photometric slit-loss version
        slit = [r["ha_slit_factor"], r["ha_slit_factor_err"], r["hb_slit_factor"], r["hb_slit_factor_err"]]
        if not all(np.isfinite(v) for v in slit):
            continue
        slit_in_ratio = (args.balmer_basis == "slitcorr") or not bool(r["same_grating"])
        mc = balmer_mc(*obs, *slit, slit_in_ratio, rng)
        out.loc[i, "balmer_basis"] = "slitcorr" if slit_in_ratio else "observed_same_grating"
        for c, key in zip(cols, keys):
            out.loc[i, c], out.loc[i, c + "_err_lo"], out.loc[i, c + "_err_hi"] = mc[key]
        if mc["frac_dropped"] > 0.01:
            print(f"  ID {r['ID']}: {100*mc['frac_dropped']:.1f}% of MC draws had a non-positive flux")

    order = (["ID", "grating", "z_dja", "z_sys", "z_sys_err", "duplicate",
              "grating_ha", "grating_hb", "same_grating", "slit_reference",
              "ha_flux_uncorr", "ha_flux_uncorr_err", "ha_flux_uncorr_snr",
              "hb_flux_uncorr", "hb_flux_uncorr_err", "hb_flux_uncorr_snr"]
             + [f"{p}_{c}" for p in ("ha", "hb") for c in
                ("slit_filter", "slit_coverage", "slit_factor", "slit_factor_err", "slit_source", "slit_note",
                 "flux_slitcorr", "flux_slitcorr_err")]
             + [c + s for c in cols for s in ("", "_err_lo", "_err_hi")] + ["balmer_basis"]
             + [c + "_dja" + s for c in cols for s in ("", "_err_lo", "_err_hi")] + ["dust_method"])
    out = out[order].sort_values("ID")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ---- compact values-only table ----
    compact_cols = ["ID", "z_sys",
                    "ha_flux_uncorr", "ha_flux_uncorr_err",
                    "ha_flux_slitcorr", "ha_flux_slitcorr_err",
                    "ha_flux_fullcorr", "ha_flux_fullcorr_err_lo", "ha_flux_fullcorr_err_hi",
                    "ha_flux_fullcorr_dja", "ha_flux_fullcorr_dja_err_lo", "ha_flux_fullcorr_dja_err_hi",
                    "hb_flux_uncorr", "hb_flux_uncorr_err",
                    "hb_flux_slitcorr", "hb_flux_slitcorr_err"]
    compact_csv = os.path.splitext(OUT_CSV)[0] + "_values.csv"
    out[compact_cols].to_csv(compact_csv, index=False)

    print("\n" + "=" * 60)
    print(f"  sources                     {len(out)}")
    print(f"  with Ha flux                {int(out['ha_flux_uncorr'].notna().sum())}")
    print(f"  with Hb flux                {int(out['hb_flux_uncorr'].notna().sum())}")
    print(f"  Ha slit direct / median     {int((out['ha_slit_source']=='direct').sum())} / "
          f"{int((out['ha_slit_source']=='sample_median').sum())}")
    print(f"  Hb slit direct / median     {int((out['hb_slit_source']=='direct').sum())} / "
          f"{int((out['hb_slit_source']=='sample_median').sum())}")
    print(f"  Ha and Hb same grating      {int(out['same_grating'].sum())}")
    print(f"  Balmer corrected            {int((out['dust_method']=='balmer').sum())}")
    print(f"  full output                 {os.path.abspath(OUT_CSV)}")
    print(f"  values-only output          {os.path.abspath(compact_csv)}")
    print("=" * 60)


if __name__ == "__main__":
    main()