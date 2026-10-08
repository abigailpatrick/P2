#!/usr/bin/env python
"""Slit-loss and Balmer-decrement dust corrections for the Halpha and Hbeta
fluxes, using the PRIMER + MINERVA total photometry.

One row per source with the observed Halpha and Hbeta fluxes, the slit-loss
factor for each line, the slit-corrected fluxes, and for sources where both
lines are measured the Balmer decrement, E(B-V), A_Halpha and the fully
corrected (slit loss + dust) Halpha flux.

How it works
------------
1. Sample. Every source in grating_sources_with_zsys.csv with a z_sys
   (build_zsys_catalog.py, so the same z_sys rule as everywhere else).
   --in-muse-only keeps just the MUSE sources. Duplicate pairs are kept,
   the master catalogue's is_primary column picks one later.

2. Line fluxes. From Ha_summary_by_JELS_ID.csv and Hbeta_summary_by_JELS_ID.csv
   (lime_lines_jointfit.py), per grating, in erg/s/cm2. Halpha is the
   [NII]-deblended flux. Which grating each line is taken from:
     a. if one or more gratings have both lines, the one with the highest
        S/N on the ratio, 1/sqrt(1/SN_Ha^2 + 1/SN_Hb^2), so the flux
        calibration of a single spectrum cancels in the decrement;
     b. unless that grating's Hbeta is below --line-snr-min while another
        grating's Hbeta is above it, in which case
     c. each line comes from its own highest-S/N grating.
   grating_ha, grating_hb and same_grating record the choice.

3. Slit-loss factor, per line, in the spectrum the line came from.
   The NIRSpec spectrum is compared with the PRIMER+MINERVA photometry
   (match_primer_minerva.py output, matched to JELS by position) in a filter
   that contains the line:
     a. candidate filters have transmission at the observed line wavelength
        of at least half their peak (the half-power definition),
     b. the line must fall on the spectrum, and at least --min-coverage of
        the filter (photon weighted, T c/lambda) must lie on it. Small
        gaps are interpolated over, any uncovered wing is filled with the
        median f_nu of the spectrum inside the filter,
     c. synthetic f_nu = int f_lambda T lambda dlambda / int T c/lambda dlambda,
     d. factor = f_nu(photometry) / f_nu(synthetic), both at S/N >= 3,
     e. among passing filters a medium band is preferred (MINERVA), then a
        wide band, then the highest factor S/N.
   The photometry is TOTAL by default (aperture flux x total_correction, as
   Ken advised), so the corrected line fluxes are total fluxes.
   --phot aperture uses the 0.3/0.5 arcsec aperture fluxes instead.
   Sources whose photometry Flag contains 1, 2 or 3 (flags combine as
   digits, e.g. 561 = Flags 5, 6 and 1), and sources marked pm_blend
   by match_primer_minerva.py (a comparably bright neighbour within
   0.3 arcsec, --allow-blends to keep them) (near stars or bright
   neighbours, or a capped total correction) get no direct factor.
   Where no filter passes, the factor is the median of the direct factors
   for that line across the sample, with the NMAD scatter as its error
   (needs at least --min-for-median direct factors). ha_slit_source says
   which: direct or sample_median.
   --slit-mode dja skips all of this and uses factor 1 (trust the msaexp
   path-loss correction in the DJA spectra).

4. Balmer decrement, Monte Carlo (--n-mc draws). Only where Halpha and
   Hbeta both have S/N >= --line-snr-min.
     ratio basis   observed_same_grating: both lines from one spectrum,
                   observed ratio, slit factors cancel.
                   slitcorr: different gratings, each line slit-corrected
                   first. --balmer-basis slitcorr forces this always.
     E(B-V)        2.5 / (k(Hb) - k(Ha)) log10(R / 2.86), 0 if R < 2.86
                   (Case B, 1e4 K, n_e = 100 cm^-3, Osterbrock & Ferland 2006)
     A_Ha          k(Ha) E(B-V), Calzetti et al. (2000), R_V = 4.05
     F_full        F_Ha,obs x slit_factor_Ha x 10^(0.4 A_Ha)
   Values are the 50th percentile, errors 50-16 and 84-50. The same
   decrement with slit factor 1 gives the *_dja columns for comparison.
   Sources without a usable Hbeta have no dust correction yet. The
   SED-based fallback (A_V from Ken's fits scaled to Halpha) comes once the
   SED block exists. dust_method says which applied.

Outputs
-------
  ha_hb_flux_corrections.csv   full table (read by build_master_catalog.py)
  ha_hb_flux_corrections_values.csv   just the main values
Column names are unchanged from the previous version, so the master
catalogue needs no edits. New columns: z_sys_line, phot_mode, pm_sep_arcsec,
pm_flag, pm_total_correction, and per line <ha|hb>_slit_phot_ujy and
<ha|hb>_slit_synth_ujy (the two fluxes the factor is the ratio of).

Usage
-----
python build_ha_flux_corrections.py --download-filters   # once
python build_ha_flux_corrections.py                      # total photometry
python build_ha_flux_corrections.py --phot aperture --out .../ha_hb_flux_corrections_aper.csv
python build_ha_flux_corrections.py --slit-mode dja      # no photometric slit correction
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as pc  # noqa: E402

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
SPEC_ROOT = f"{pc.P2_ROOT}/jwst_spectra"
FILTER_DIR = f"{CAT_DIR}/nircam_filters"

GRATINGS_FULL = {"G235M": "G235M_F170LP", "G235H": "G235H_F170LP",
                 "G395M": "G395M_F290LP", "G395H": "G395H_F290LP"}

# Bands in the PRIMER+MINERVA catalogue that G235 / G395 can cover
# (G235 starts at 1.66 um, so nothing bluer than F182M is useful).
SLIT_FILTERS = ["F182M", "F200W", "F210M", "F250M", "F277W", "F300M",
                "F356W", "F360M", "F410M", "F444W", "F460M"]
SVO_URL = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=JWST/NIRCam.{filt}"

C_ANG = 2.99792458e18          # speed of light, Angstrom / s
HA_REST = 6564.632             # vacuum, Angstrom, as lime_lines_jointfit.py
HB_REST = 4862.683
R_INT = 2.86                   # Case B Ha/Hb
RV_CALZETTI = 4.05
COVER_T_FLOOR = 0.01           # filter curve below this fraction of peak ignored
GAP_TOL_PIX = 5                # spectrum gaps up to this many pixels interpolated


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download-filters", action="store_true",
                   help="Download any missing NIRCam filter curves from SVO, then exit.")
    p.add_argument("--zsys-csv", default=f"{CAT_DIR}/grating_sources_with_zsys.csv")
    p.add_argument("--ha-summary", default=f"{CAT_DIR}/Ha_summary_by_JELS_ID.csv")
    p.add_argument("--hb-summary", default=f"{CAT_DIR}/Hbeta_summary_by_JELS_ID.csv")
    p.add_argument("--phot-csv", default=f"{CAT_DIR}/primer_minerva_by_JELS_ID.csv",
                   help="match_primer_minerva.py output.")
    p.add_argument("--phot", choices=["total", "aperture"], default="total")
    p.add_argument("--slit-mode", choices=["phot", "dja"], default="phot")
    p.add_argument("--allow-flagged-phot", action="store_true",
                   help="Also use photometry with Flag 1, 2 or 3.")
    p.add_argument("--allow-blends", action="store_true",
                   help="Also use photometry of sources with pm_blend True.")
    p.add_argument("--in-muse-only", action="store_true")
    p.add_argument("--line-snr-min", type=float, default=3.0,
                   help="S/N needed on both lines for the Balmer decrement.")
    p.add_argument("--slit-snr-min", type=float, default=3.0)
    p.add_argument("--min-coverage", type=float, default=0.70)
    p.add_argument("--min-for-median", type=int, default=3)
    p.add_argument("--balmer-basis", choices=["auto", "slitcorr"], default="auto")
    p.add_argument("--n-mc", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=f"{CAT_DIR}/ha_hb_flux_corrections.csv")
    return p.parse_args()


# ============================================================================
# Filters
# ============================================================================

def filter_path(filt):
    return os.path.join(FILTER_DIR, f"JWST_NIRCam.{filt}.dat")


def download_filters():
    os.makedirs(FILTER_DIR, exist_ok=True)
    for filt in SLIT_FILTERS:
        out = filter_path(filt)
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


def load_filters():
    curves = {}
    missing = [f for f in SLIT_FILTERS if not os.path.exists(filter_path(f))]
    if missing:
        raise FileNotFoundError(f"Filter curves missing for {missing} in {FILTER_DIR}. "
                                "Run with --download-filters first.")
    for filt in SLIT_FILTERS:
        d = np.loadtxt(filter_path(filt))
        o = np.argsort(d[:, 0])
        curves[filt] = (d[o, 0], d[o, 1])
    return curves


def filter_rank(filt):
    """Medium before wide."""
    return {"N": 0, "M": 1, "W": 2}[filt[-1]]


# ============================================================================
# Attenuation
# ============================================================================

def k_calzetti(lam_aa):
    """Calzetti et al. (2000) k(lambda), lambda in Angstrom, 0.12-2.2 um."""
    x = lam_aa / 1e4
    if 0.63 <= x <= 2.20:
        return 2.659 * (-1.857 + 1.040 / x) + RV_CALZETTI
    if 0.12 <= x < 0.63:
        return 2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + RV_CALZETTI
    raise ValueError(f"Calzetti curve undefined at {lam_aa} AA")


K_HA = k_calzetti(HA_REST)
K_HB = k_calzetti(HB_REST)


# ============================================================================
# Step 1-2: sample and line fluxes
# ============================================================================

def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def load_sample(path, in_muse_only):
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(int)
    n0 = len(df)
    df = df[np.isfinite(pd.to_numeric(df["z_sys"], errors="coerce"))]
    if in_muse_only:
        df = df[truthy(df["in_muse"])]
    print(f"  sample: {len(df)} of {n0} sources with a z_sys"
          f"{' and in MUSE' if in_muse_only else ''}")
    keep = [c for c in ("ID", "z_sys", "z_sys_err", "z_sys_line", "in_muse", "duplicate")
            if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_summary(path):
    s = pd.read_csv(path)
    s["ID"] = s["ID"].astype(int)
    return s.set_index("ID")


def line_fluxes(summ, sid, name):
    """{grating: (flux, err, snr)} for gratings with a finite flux."""
    if sid not in summ.index:
        return {}
    row, out = summ.loc[sid], {}
    for gr in GRATINGS_FULL:
        f = row.get(f"{name}_{gr}_flux", np.nan)
        e = row.get(f"{name}_{gr}_flux_err", np.nan)
        if pd.notna(f) and pd.notna(e) and np.isfinite(f) and np.isfinite(e) and e > 0:
            out[gr] = (float(f), float(e), float(f) / float(e))
    return out


def choose_gratings(ha, hb, snr_min):
    """(grating_ha, grating_hb), either may be None. See step 2 in the docstring."""
    def best(d):
        return max(d, key=lambda g: d[g][2]) if d else None
    g_ha, g_hb = best(ha), best(hb)
    common = [g for g in ha if g in hb]
    if common:
        rsnr = {g: 1.0 / np.sqrt(1.0 / ha[g][2] ** 2 + 1.0 / hb[g][2] ** 2)
                if ha[g][2] > 0 and hb[g][2] > 0 else -np.inf for g in common}
        g = max(rsnr, key=rsnr.get)
        if hb[g][2] >= snr_min or hb[g_hb][2] < snr_min:
            return g, g
    return g_ha, g_hb


# ============================================================================
# Step 3: slit loss
# ============================================================================

def load_photometry(path, mode):
    """{ID: {"bands": {FILT: (f_nu Jy, err Jy)}, "flag": int, "sep": .., "tc": ..}}"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run match_primer_minerva.py first.")
    t = pd.read_csv(path)
    t["ID"] = t["ID"].astype(int)
    suf = "_tot" if mode == "total" else "_ap"
    out = {}
    for _, r in t.iterrows():
        bands = {}
        for filt in SLIT_FILTERS:
            f, e = r.get(filt.lower() + suf, np.nan), r.get(filt.lower() + suf + "_err", np.nan)
            if np.isfinite(f) and np.isfinite(e) and e > 0:
                bands[filt] = (f * 1e-6, e * 1e-6)          # uJy -> Jy
        out[int(r["ID"])] = dict(bands=bands, flag=r.get("pm_flag", np.nan),
                                 blend=str(r.get("pm_blend", "")).lower() in ("true", "1", "1.0"),
                                 sep=r.get("pm_sep_arcsec", np.nan),
                                 tc=r.get("pm_total_correction", np.nan))
    return out


def spectrum_path(sid, gr):
    full = GRATINGS_FULL[gr]
    return os.path.join(SPEC_ROOT, full, f"{sid}_{full}_spectra_lime.fits")


def load_spectrum(sid, gr):
    with fits.open(spectrum_path(sid, gr)) as h:
        t = h["SPECTRUM"].data
        w, f, e = (np.asarray(t[c], dtype=float) for c in ("WAVE", "FLUX", "ERR"))
    ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    o = np.argsort(w[ok])
    return w[ok][o], f[ok][o], e[ok][o]


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
    """Fraction of the filter's photon-counting weight on the spectrum."""
    keep = t_f >= COVER_T_FLOOR * t_f.max()
    lf, tf = lam_f[keep], t_f[keep]
    wt = tf * C_ANG / lf
    return _trapz(wt * covered_mask(w, lf), lf) / _trapz(wt, lf)


def synthetic_fnu(w, f, e, lam_f, t_f):
    """Photon-counting synthetic f_nu (Jy) and error from an f_lambda spectrum.
    Uncovered filter wings are filled with the median f_nu inside the filter."""
    t = np.interp(w, lam_f, t_f, left=0.0, right=0.0)
    on = t >= COVER_T_FLOOR * t_f.max()
    dl = np.gradient(w)
    den = np.sum(t * C_ANG / w * dl)
    fnu_cov = np.sum(f * t * w * dl) / den
    err_cov = np.sqrt(np.sum((e * t * w * dl) ** 2)) / den
    cov = coverage(w, lam_f, t_f)
    if cov < 1.0:
        fill = np.median(f[on] * w[on] ** 2 / C_ANG)
        fill_err = 1.253 * np.median(e[on] * w[on] ** 2 / C_ANG) / np.sqrt(on.sum())
    else:
        fill, fill_err = 0.0, 0.0
    fnu = cov * fnu_cov + (1 - cov) * fill
    err = np.hypot(cov * err_cov, (1 - cov) * fill_err)
    return fnu / 1e-23, err / 1e-23


def direct_slit_factor(sid, gr, lam_line, bands, curves, a):
    """Best direct factor for one line. Returns (dict or None, reason)."""
    try:
        w, f, e = load_spectrum(sid, gr)
    except (FileNotFoundError, OSError, KeyError):
        return None, "no spectrum"
    cands, reasons = [], []
    for filt in SLIT_FILTERS:
        if filt not in bands:
            continue
        lam_f, t_f = curves[filt]
        if np.interp(lam_line, lam_f, t_f, left=0, right=0) < 0.5 * t_f.max():
            continue
        if not covered_mask(w, np.array([lam_line]))[0]:
            reasons.append(f"{filt} line in spectrum gap")
            continue
        cov = coverage(w, lam_f, t_f)
        if cov < a.min_coverage:
            reasons.append(f"{filt} cov {cov:.2f}")
            continue
        fs, es = synthetic_fnu(w, f, e, lam_f, t_f)
        fp, ep = bands[filt]
        if not (fs / es >= a.slit_snr_min and fp / ep >= a.slit_snr_min):
            reasons.append(f"{filt} S/N synth {fs/es:.1f} phot {fp/ep:.1f}")
            continue
        fac = fp / fs
        fac_e = fac * np.sqrt((ep / fp) ** 2 + (es / fs) ** 2)
        cands.append(dict(filt=filt, factor=fac, factor_err=fac_e, snr=fac / fac_e,
                          synth=fs, phot=fp, cov=cov))
    if not cands:
        return None, "; ".join(reasons) if reasons else "no filter contains line"
    return min(cands, key=lambda c: (filter_rank(c["filt"]), -c["snr"])), ""


def slit_factors(out, phot, curves, a):
    """Fill the <ha|hb>_slit_* columns in place."""
    for pre in ("ha", "hb"):
        for c in ("slit_filter", "slit_source", "slit_note"):
            out[f"{pre}_{c}"] = None
        for c in ("slit_factor", "slit_factor_err", "slit_coverage",
                  "slit_phot_ujy", "slit_synth_ujy"):
            out[f"{pre}_{c}"] = np.nan

    for i, r in out.iterrows():
        sid, z = int(r["ID"]), float(r["z_sys"])
        p = phot.get(sid)
        for pre, gcol, rest in (("ha", "grating_ha", HA_REST), ("hb", "grating_hb", HB_REST)):
            g = r[gcol]
            if not isinstance(g, str):
                continue
            if a.slit_mode == "dja":
                out.loc[i, [f"{pre}_slit_factor", f"{pre}_slit_factor_err"]] = 1.0, 0.0
                out.loc[i, f"{pre}_slit_source"] = "dja"
                continue
            if p is None or not p["bands"]:
                out.loc[i, f"{pre}_slit_note"] = "no PRIMER+MINERVA match"
                continue
            digits = {int(c) for c in str(int(p["flag"]))} if np.isfinite(p["flag"]) else set()
            if (not a.allow_flagged_phot) and digits & {1, 2, 3}:
                out.loc[i, f"{pre}_slit_note"] = f"photometry Flag {int(p['flag'])}"
                continue
            if (not a.allow_blends) and p.get("blend"):
                out.loc[i, f"{pre}_slit_note"] = "blended in PRIMER+MINERVA"
                continue
            best, why = direct_slit_factor(sid, g, rest * (1 + z), p["bands"], curves, a)
            if best is None:
                out.loc[i, f"{pre}_slit_note"] = why
                continue
            out.loc[i, f"{pre}_slit_filter"] = best["filt"]
            out.loc[i, f"{pre}_slit_factor"] = best["factor"]
            out.loc[i, f"{pre}_slit_factor_err"] = best["factor_err"]
            out.loc[i, f"{pre}_slit_coverage"] = best["cov"]
            out.loc[i, f"{pre}_slit_phot_ujy"] = best["phot"] * 1e6
            out.loc[i, f"{pre}_slit_synth_ujy"] = best["synth"] * 1e6
            out.loc[i, f"{pre}_slit_source"] = "direct"

    # sample-median fallback
    for pre in ("ha", "hb"):
        direct = out.loc[out[f"{pre}_slit_source"] == "direct", f"{pre}_slit_factor"].to_numpy()
        need = out[f"{pre}_flux_uncorr"].notna() & out[f"{pre}_slit_source"].isna()
        if a.slit_mode == "phot":
            if len(direct) >= a.min_for_median:
                med = float(np.median(direct))
                nmad = 1.4826 * float(np.median(np.abs(direct - med)))
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


# ============================================================================
# Step 4: Balmer decrement
# ============================================================================

def balmer_mc(fha, eha, fhb, ehb, sha, esha, shb, eshb, slit_in_ratio, n, rng):
    ha = rng.normal(fha, eha, n)
    hb = rng.normal(fhb, ehb, n)
    s_ha = rng.normal(sha, esha, n)
    ratio = (ha * s_ha) / (hb * rng.normal(shb, eshb, n)) if slit_in_ratio else ha / hb
    good = (ha > 0) & (hb > 0) & (s_ha > 0) & (ratio > 0)
    ratio, ha_corr = ratio[good], (ha * s_ha)[good]
    ebv = np.clip(2.5 / (K_HB - K_HA) * np.log10(ratio / R_INT), 0.0, None)
    a_ha = K_HA * ebv
    full = ha_corr * 10 ** (0.4 * a_ha)

    def pct(x):
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        return p50, p50 - p16, p84 - p50
    return dict(ratio=pct(ratio), ebv=pct(ebv), a_ha=pct(a_ha), full=pct(full),
                frac_dropped=1 - good.mean())


def dust_correct(out, a):
    rng = np.random.default_rng(a.seed)
    cols = ["balmer_ratio", "ebv_neb", "A_ha", "ha_flux_fullcorr"]
    keys = ("ratio", "ebv", "a_ha", "full")
    for c in cols:
        for v in ("", "_dja"):
            for s in ("", "_err_lo", "_err_hi"):
                out[c + v + s] = np.nan
    out["dust_method"] = None
    out["balmer_basis"] = None
    for i, r in out.iterrows():
        obs = [r["ha_flux_uncorr"], r["ha_flux_uncorr_err"],
               r["hb_flux_uncorr"], r["hb_flux_uncorr_err"]]
        if not all(np.isfinite(v) for v in obs):
            continue
        if r["ha_flux_uncorr_snr"] < a.line_snr_min or r["hb_flux_uncorr_snr"] < a.line_snr_min:
            continue
        out.loc[i, "dust_method"] = "balmer"
        mc = balmer_mc(*obs, 1.0, 0.0, 1.0, 0.0, False, a.n_mc, rng)
        for c, k in zip(cols, keys):
            out.loc[i, [c + "_dja", c + "_dja_err_lo", c + "_dja_err_hi"]] = mc[k]
        slit = [r["ha_slit_factor"], r["ha_slit_factor_err"],
                r["hb_slit_factor"], r["hb_slit_factor_err"]]
        if not all(np.isfinite(v) for v in slit):
            continue
        in_ratio = (a.balmer_basis == "slitcorr") or not bool(r["same_grating"])
        mc = balmer_mc(*obs, *slit, in_ratio, a.n_mc, rng)
        out.loc[i, "balmer_basis"] = "slitcorr" if in_ratio else "observed_same_grating"
        for c, k in zip(cols, keys):
            out.loc[i, [c, c + "_err_lo", c + "_err_hi"]] = mc[k]
        if mc["frac_dropped"] > 0.01:
            print(f"  ID {int(r['ID'])}: {100*mc['frac_dropped']:.1f}% of MC draws non-positive")
    return cols


# ============================================================================
# Main
# ============================================================================

def main():
    a = parse_args()
    print("build_ha_flux_corrections.py")
    if a.download_filters:
        print(f"Filter curves in {os.path.abspath(FILTER_DIR)}")
        download_filters()
        return

    print("[CONFIG]")
    print(f"  sample        {os.path.abspath(a.zsys_csv)}")
    print(f"  Ha fluxes     {os.path.abspath(a.ha_summary)}")
    print(f"  Hb fluxes     {os.path.abspath(a.hb_summary)}")
    print(f"  slit mode     {a.slit_mode}")
    if a.slit_mode == "phot":
        print(f"  photometry    {os.path.abspath(a.phot_csv)}  ({a.phot} fluxes)")
        print(f"  spectra       {os.path.abspath(SPEC_ROOT)}/<grating>/")
        print(f"  filters       {os.path.abspath(FILTER_DIR)}")
    print(f"  output        {os.path.abspath(a.out)}")
    print(f"  k(Ha) {K_HA:.3f}  k(Hb) {K_HB:.3f}  (Calzetti 2000)\n")

    out = load_sample(a.zsys_csv, a.in_muse_only)
    ha_s, hb_s = load_summary(a.ha_summary), load_summary(a.hb_summary)
    phot = load_photometry(a.phot_csv, a.phot) if a.slit_mode == "phot" or os.path.exists(a.phot_csv) else {}
    curves = load_filters() if a.slit_mode == "phot" else None

    # Step 2: line fluxes
    rows = []
    for _, r in out.iterrows():
        sid = int(r["ID"])
        ha, hb = line_fluxes(ha_s, sid, "Ha"), line_fluxes(hb_s, sid, "Hbeta")
        g_ha, g_hb = choose_gratings(ha, hb, a.line_snr_min)
        d = dict(grating_ha=g_ha, grating_hb=g_hb, same_grating=bool(g_ha and g_hb and g_ha == g_hb))
        for pre, lines, g in (("ha", ha, g_ha), ("hb", hb, g_hb)):
            fl, er, sn = lines[g] if g else (np.nan, np.nan, np.nan)
            d[f"{pre}_flux_uncorr"], d[f"{pre}_flux_uncorr_err"], d[f"{pre}_flux_uncorr_snr"] = fl, er, sn
        p = phot.get(sid, {})
        d["pm_sep_arcsec"] = p.get("sep", np.nan)
        d["pm_flag"] = p.get("flag", np.nan)
        d["pm_total_correction"] = p.get("tc", np.nan)
        rows.append(d)
    out = pd.concat([out, pd.DataFrame(rows)], axis=1)
    out["phot_mode"] = a.phot if a.slit_mode == "phot" else "none"
    out["slit_reference"] = (f"PRIMER+MINERVA {a.phot}" if a.slit_mode == "phot"
                             else "dja_pathloss_only")

    # Step 3: slit loss
    slit_factors(out, phot, curves, a)

    # Step 4: dust
    cols = dust_correct(out, a)

    # Write
    order = (["ID", "z_sys", "z_sys_err", "z_sys_line", "in_muse", "duplicate",
              "pm_sep_arcsec", "pm_flag", "pm_total_correction", "phot_mode",
              "grating_ha", "grating_hb", "same_grating", "slit_reference",
              "ha_flux_uncorr", "ha_flux_uncorr_err", "ha_flux_uncorr_snr",
              "hb_flux_uncorr", "hb_flux_uncorr_err", "hb_flux_uncorr_snr"]
             + [f"{p}_{c}" for p in ("ha", "hb") for c in
                ("slit_filter", "slit_coverage", "slit_phot_ujy", "slit_synth_ujy",
                 "slit_factor", "slit_factor_err", "slit_source", "slit_note",
                 "flux_slitcorr", "flux_slitcorr_err")]
             + [c + s for c in cols for s in ("", "_err_lo", "_err_hi")] + ["balmer_basis"]
             + [c + "_dja" + s for c in cols for s in ("", "_err_lo", "_err_hi")]
             + ["dust_method"])
    out = out[[c for c in order if c in out.columns]].sort_values("ID")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    compact = os.path.splitext(a.out)[0] + "_values.csv"
    out[["ID", "z_sys", "ha_flux_uncorr", "ha_flux_uncorr_err",
         "ha_flux_slitcorr", "ha_flux_slitcorr_err",
         "ha_flux_fullcorr", "ha_flux_fullcorr_err_lo", "ha_flux_fullcorr_err_hi",
         "ha_flux_fullcorr_dja", "ha_flux_fullcorr_dja_err_lo", "ha_flux_fullcorr_dja_err_hi",
         "hb_flux_uncorr", "hb_flux_uncorr_err", "hb_flux_slitcorr",
         "hb_flux_slitcorr_err"]].to_csv(compact, index=False)

    print("\n" + "=" * 60)
    print(f"  sources                     {len(out)}")
    print(f"  matched to PRIMER+MINERVA   {int(out['pm_sep_arcsec'].notna().sum())}")
    print(f"  with Ha / Hb flux           {int(out['ha_flux_uncorr'].notna().sum())} / "
          f"{int(out['hb_flux_uncorr'].notna().sum())}")
    for pre in ("ha", "hb"):
        src = out[f"{pre}_slit_source"]
        print(f"  {pre} slit direct / median     {int((src == 'direct').sum())} / "
              f"{int((src == 'sample_median').sum())}")
        filt = out.loc[src == "direct", f"{pre}_slit_filter"].value_counts()
        if len(filt):
            print(f"     filters used             " + ", ".join(f"{k} {v}" for k, v in filt.items()))
    print(f"  Ha and Hb same grating      {int(out['same_grating'].sum())}")
    print(f"  Balmer corrected            {int((out['dust_method'] == 'balmer').sum())}")
    for c in ("ha_slit_factor", "balmer_ratio"):
        v = out[c].dropna()
        if len(v):
            print(f"  {c:26s}  median {v.median():.2f}  (16-84: "
                  f"{v.quantile(.16):.2f}-{v.quantile(.84):.2f})")
    tot = (out["ha_flux_fullcorr"] / out["ha_flux_uncorr"]).dropna()
    if len(tot):
        print(f"  total Ha correction factor  median {tot.median():.2f}, max {tot.max():.1f}")
    print(f"  full output                 {os.path.abspath(a.out)}")
    print(f"  values-only output          {os.path.abspath(compact)}")
    print("=" * 60)


if __name__ == "__main__":
    main()