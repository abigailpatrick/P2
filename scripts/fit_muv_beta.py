#!/usr/bin/env python3
"""
Measure the UV slope beta and absolute UV magnitude M_UV for each P2 source by
fitting a power law to the JELS/PRIMER rest-frame UV photometry.

Method (follows Paper 1, Section 2.4, after Pirie et al. 2025)
----------------------------------------------------------------
For each source the filters probing the rest-frame UV continuum are fitted with

    f_lambda = f0 * (lambda_rest / 1500 A)^beta

by weighted non-linear least squares (scipy curve_fit, absolute_sigma=True).
f0 is the flux density at rest-frame 1500 A. That is converted to f_nu at the
observed wavelength 1500(1+z), then to an apparent AB magnitude, then to

    M_UV = m_1500 - 5 log10(D_L / 10 pc) + 2.5 log10(1 + z)

No dust correction is applied, matching Paper 1.

Filter selection, per source
----------------------------
A filter is used only if
  1. its blue half-power edge lies redward of LYA_CUT (default 1250 A) in the
     rest frame. The whole bandpass then sits clear of the Lya line and the
     IGM break, including the damping wing.
  2. its pivot wavelength lies at or below UV_MAX (default 3000 A) in the rest
     frame, so it still samples the UV continuum and not the Balmer break.
  3. the flux and error are finite and the error is positive (i.e. the source
     has coverage in that band).
Negative or low S/N fluxes are kept. The fit is done in linear flux space with
the measured errors, so non-detections still pull the fit in the right way.
At least MIN_FILTERS (default 2) filters are needed to fit.

Photometry
----------
Aperture-corrected 0.6 arcsec photometry, <INST>_<FILT>_APER_600_mas_flux_corr
and _fluxerr_corr, as in Paper 1. The flux unit is checked per column from the
matching _mag_corr column (zero point 23.9 = uJy, 31.4 = nJy) rather than
trusted from the header.

Redshift
--------
z_sys from systemic_redshifts_by_JELS_ID.csv with the shared rule in
p2_common.pick_zsys ([OIII] if S/N > 13, then Halpha if S/N > 13). If neither
passes, z_av from grating_sources_by_JELS_ID.csv.

Catalogue lookup
----------------
The four per-grating FITS catalogues are searched in order. Once a source is
found in one, its photometry is taken from that catalogue and the rest are not
checked. Matching is on the JELS ID column (not srcid).

Where the numbers come from
---------------------------
  filter pivots and half-power edges   JDox (NIRCam), SVO FPS (HST), see FILTERS
  power law, lambda0 = 1500 A, 3000 A  Paper 1 Sec. 2.4 / Pirie et al. (2025)
  f_lambda propto lambda^beta          Calzetti et al. (1994); Meurer et al. (1999)
  AB mags, 3631 Jy                     Oke & Gunn (1983)
  ZP 23.9 (uJy), 31.4 (nJy)            follow from the AB definition
  f_lambda = f_nu c / lambda^2         standard, c exact SI value
  M_UV distance modulus + 2.5log(1+z)  Hogg et al. (2002)
  H0 = 70, Om = 0.3                    Paper 1
  z_sys rule (OIII, Ha, S/N > 13)      p2_common.py
  1250 A Lya cut, >= 2 filters         adopted here, not from a reference

Outputs
-------
  CSV   one row per source, written to --out-csv
  PNG   one fit plot per source, written to --fig-root/<ID>/<ID>_muv_beta_fit.png

Usage
-----
python fit_muv_beta.py --all
python fit_muv_beta.py --id 12345
python fit_muv_beta.py --all --dry-run     # print filter selection, write nothing
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
from scipy.optimize import curve_fit

P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"

FITS_CATALOGUES = [
    "JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    "JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    "JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    "JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
]

# ---------------------------------------------------------------------------
# Filter wavelengths. Every number below is copied from the page cited for it,
# in the units used there. Nothing is estimated.
#
# NIRCam: JWST User Documentation (JDox), "NIRCam Filters", filter table
#   https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
#   Columns used: pivot wavelength, and the blue and red half-power
#   wavelengths (lambda-, lambda+), all in micron. Throughputs include the OTE,
#   NIRCam optics and detector QE, from commissioning flight data.
#
# HST: SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012; Rodrigo &
#   Solano 2020), VOTable for each filter ID, e.g.
#   http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=HST/ACS_WFC.F814W
#   PARAMs used: WavelengthPivot, WavelengthCen, FWHM, in Angstrom. SVO
#   computes these from the STScI stsynphot throughputs. WavelengthCen is the
#   midpoint of the two half-maximum points, so the half-power edges are
#   WavelengthCen -/+ FWHM/2, the same definition as the JDox lambda-/lambda+.
#   WFC3/UVIS F275W uses the UVIS2 entry.
# ---------------------------------------------------------------------------

# JDox: filter -> (pivot, lambda-, lambda+) in micron
NIRCAM_JDOX = {
    "F090W": (0.901, 0.795, 1.005),
    "F115W": (1.154, 1.013, 1.282),
    "F150W": (1.501, 1.331, 1.668),
    "F200W": (1.990, 1.755, 2.227),
    "F277W": (2.786, 2.423, 3.132),
    "F356W": (3.563, 3.135, 3.981),
    "F410M": (4.092, 3.866, 4.302),
    "F444W": (4.421, 3.881, 4.982),
    "F466N": (4.654, 4.629, 4.681),
    "F470N": (4.707, 4.683, 4.733),
}

# SVO: filter -> (column prefix, SVO filter ID, WavelengthPivot, WavelengthCen, FWHM) in Angstrom
HST_SVO = {
    "F275W": ("WFC3UV", "HST/WFC3_UVIS2.F275W", 2702.9201281817, 2725.9113782663, 471.41940838989),
    "F435W": ("ACSWFC", "HST/ACS_WFC.F435W", 4329.8468812521, 4369.6175314903, 900.04349333548),
    "F606W": ("ACSWFC", "HST/ACS_WFC.F606W", 5921.8841189137, 5962.1998266378, 2253.4024368618),
    "F814W": ("ACSWFC", "HST/ACS_WFC.F814W", 8045.5275046827, 8102.9069995926, 2098.146800223),
    "F125W": ("WFC3IR", "HST/WFC3_IR.F125W", 12486.069387882, 12503.907883119, 2993.7862698758),
    "F140W": ("WFC3IR", "HST/WFC3_IR.F140W", 13923.20638559, 13983.359286096, 3933.3200695301),
    "F160W": ("WFC3IR", "HST/WFC3_IR.F160W", 15370.33814998, 15437.705833583, 2876.730874382),
}


def build_filter_table():
    """filter -> (column prefix, pivot, blue half-power edge, red half-power edge), Angstrom,
    sorted by pivot."""
    tab = {}
    for f, (piv, lo, hi) in NIRCAM_JDOX.items():
        tab[f] = ("NIRCam", piv * 1e4, lo * 1e4, hi * 1e4)
    for f, (inst, _, piv, cen, fwhm) in HST_SVO.items():
        tab[f] = (inst, piv, cen - fwhm / 2, cen + fwhm / 2)
    return dict(sorted(tab.items(), key=lambda kv: kv[1][1]))


FILTERS = build_filter_table()

import p2_common as pc  # noqa: E402  shared z_sys rule

LINE_PRIORITY = pc.LINE_PRIORITY
LAM0 = 1500.0                  # rest-frame anchor, Angstrom
C_ANG = 2.99792458e18          # speed of light, Angstrom / s
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)   # as Paper 1

# zero point of m = -2.5 log10(f) + ZP  ->  flux unit in Jy
ZP_TO_JY = {23.9: 1e-6, 31.4: 1e-9, 8.9: 1.0, 16.4: 1e-3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Fit every source.")
    g.add_argument("--id", type=int, help="Fit one JELS ID.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the filter selection and fits, write nothing.")
    p.add_argument("--sources-csv", default=f"{CAT_DIR}/grating_sources_by_JELS_ID.csv")
    p.add_argument("--systemic-csv", default=f"{CAT_DIR}/systemic_redshifts_by_JELS_ID.csv")
    p.add_argument("--cat-dir", default=CAT_DIR)
    p.add_argument("--out-csv", default=f"{CAT_DIR}/muv_beta_by_JELS_ID.csv")
    p.add_argument("--fig-root", default=f"{P2}/jwst_spectra/MUV_fits",
                   help="Plots go to <fig-root>/<ID>/<ID>_muv_beta_fit.png")
    p.add_argument("--aper", default="600", choices=["300", "600", "900", "2"],
                   help="Aperture in mas ('2' means 2 arcsec). Default 600.")
    p.add_argument("--lya-cut", type=float, default=1250.0,
                   help="Rest-frame blue edge limit, Angstrom. Default 1250.")
    p.add_argument("--uv-max", type=float, default=3000.0,
                   help="Rest-frame pivot limit, Angstrom. Default 3000.")
    p.add_argument("--min-filters", type=int, default=2)
    p.add_argument("--oiii-snr-min", type=float, default=pc.SNR_MIN["OIII"])
    p.add_argument("--ha-snr-min", type=float, default=pc.SNR_MIN["Ha"])
    p.add_argument("--max-muv-err", type=float, default=0.5,
                   help="good_muv requires M_UV_err below this. Default 0.5.")
    p.add_argument("--max-beta-err", type=float, default=0.5,
                   help="good_beta requires beta_err below this. Default 0.5, "
                        "the Paper 1 threshold.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def col_names(filt, aper):
    inst = FILTERS[filt][0]
    ap = "2_as" if aper == "2" else f"{aper}_mas"
    base = f"{inst}_{filt}_APER_{ap}"
    return f"{base}_flux_corr", f"{base}_fluxerr_corr", f"{base}_mag_corr"


def flux_unit_to_jy(tab, fcol, mcol):
    """Infer the flux unit of a column from its matching magnitude column."""
    f = np.asarray(tab[fcol], dtype=float)
    m = np.asarray(tab[mcol], dtype=float)
    ok = np.isfinite(f) & np.isfinite(m) & (f > 0) & (m > 0) & (m < 50)
    if ok.sum() == 0:
        return None
    zp = np.median(m[ok] + 2.5 * np.log10(f[ok]))
    for ref, scale in ZP_TO_JY.items():
        if abs(zp - ref) < 0.3:   # units differ by >= 2.5 mag, so this is safe
            return scale
    raise ValueError(f"Could not identify flux unit of {fcol}: zero point {zp:.3f}")


def load_catalogues(cat_dir, aper):
    """Read the FITS catalogues once. Returns list of (name, table, {filt: Jy scale})."""
    cats = []
    for fname in FITS_CATALOGUES:
        path = os.path.join(cat_dir, fname)
        tab = Table.read(path).to_pandas()
        tab["ID"] = tab["ID"].astype(int)
        tab = tab.drop_duplicates(subset="ID").set_index("ID")
        scales = {}
        for filt in FILTERS:
            fcol, ecol, mcol = col_names(filt, aper)
            if fcol in tab.columns and ecol in tab.columns:
                s = flux_unit_to_jy(tab, fcol, mcol) if mcol in tab.columns else None
                scales[filt] = s
        print(f"  read {os.path.abspath(path)}  ({len(tab)} unique IDs)")
        cats.append((fname, tab, scales))
    # fill in any unit we could not infer in one catalogue from another
    for _, _, scales in cats:
        for filt, s in scales.items():
            if s is None:
                others = [c[2].get(filt) for c in cats if c[2].get(filt) is not None]
                scales[filt] = others[0] if others else None
    return cats


def pick_zsys(row, cuts):
    """(z, z_type) from the shared rule in p2_common.pick_zsys."""
    z, _, _, line, _ = pc.pick_zsys(row, cuts)
    return (z, f"z_sys_{line}") if line else (np.nan, None)


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------

def select_filters(z, scales, row, aper, lya_cut, uv_max):
    """Return (used, unused) lists of dicts for one source."""
    used, unused = [], []
    for filt, (inst, piv, blue, red) in FILTERS.items():
        fcol, ecol, _ = col_names(filt, aper)
        if fcol not in row.index or scales.get(filt) is None:
            continue
        f, e = float(row[fcol]), float(row[ecol])
        if not (np.isfinite(f) and np.isfinite(e) and e > 0):
            continue
        d = dict(filt=filt, fcol=fcol, ecol=ecol, raw_f=f, raw_e=e,
                 lam_obs=piv, lam_rest=piv / (1 + z),
                 fnu_jy=f * scales[filt], efnu_jy=e * scales[filt])
        ok = (blue / (1 + z) >= lya_cut) and (piv / (1 + z) <= uv_max)
        (used if ok else unused).append(d)
    return used, unused


def powerlaw(lam_rest, f0, beta):
    return f0 * (lam_rest / LAM0) ** beta


def fit_source(used, z):
    """Weighted power-law fit in f_lambda. Returns a dict of results."""
    lam_obs = np.array([d["lam_obs"] for d in used])
    lam_rest = lam_obs / (1 + z)
    fnu = np.array([d["fnu_jy"] for d in used])
    efnu = np.array([d["efnu_jy"] for d in used])

    # f_nu (Jy) -> f_lambda (erg s-1 cm-2 A-1), observed frame
    flam = fnu * 1e-23 * C_ANG / lam_obs**2
    eflam = efnu * 1e-23 * C_ANG / lam_obs**2
    scale = np.nanmax(np.abs(flam))
    y, ey = flam / scale, eflam / scale

    p0 = [max(np.median(y), 0.1), -2.0]
    popt, pcov = curve_fit(powerlaw, lam_rest, y, p0=p0, sigma=ey,
                           absolute_sigma=True,
                           bounds=([1e-6, -6.0], [np.inf, 4.0]), maxfev=20000)
    f0, beta = popt
    # If the points scatter more than their errors allow (chi2_nu > 1), the
    # covariance from absolute_sigma=True underestimates the parameter errors.
    # Inflate them by sqrt(chi2_nu), never shrink them. This scale factor is
    # the Birge ratio (Birge 1932, Phys. Rev. 40, 207), a common convention.
    # err_scale records it, so 1.0 means the errors were left as they were.
    dof = len(used) - 2
    chi2 = np.sum(((y - powerlaw(lam_rest, *popt)) / ey) ** 2)
    err_scale = np.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0
    f0_err, beta_err = np.sqrt(np.diag(pcov)) * err_scale

    # f0 is f_lambda,obs at lambda_obs = 1500(1+z)
    lam1500_obs = LAM0 * (1 + z)
    f1500_lam = f0 * scale
    f1500_nu_jy = f1500_lam * lam1500_obs**2 / C_ANG / 1e-23
    m1500 = -2.5 * np.log10(f1500_nu_jy / 3631.0)
    dl_pc = COSMO.luminosity_distance(z).to(u.pc).value
    muv = m1500 - 5 * np.log10(dl_pc / 10.0) + 2.5 * np.log10(1 + z)
    muv_err = 2.5 / np.log(10) * f0_err / f0

    return dict(beta=beta, beta_err=beta_err, f1500_nJy=f1500_nu_jy * 1e9,
                m_UV=m1500, M_UV=muv, M_UV_err=muv_err,
                chi2=chi2, dof=dof, err_scale=err_scale, popt=popt, scale=scale)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def plot_fit(src_id, z, z_type, used, unused, res, lya_cut, uv_max, out_png):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    def to_njy(d):
        return d["fnu_jy"] * 1e9, d["efnu_jy"] * 1e9

    if unused:
        x = [d["lam_rest"] for d in unused]
        yy, ee = zip(*[to_njy(d) for d in unused])
        ax.errorbar(x, yy, yerr=ee, fmt="o", mfc="none", color="0.6",
                    ms=6, capsize=2, label="not used")
    x = [d["lam_rest"] for d in used]
    yy, ee = zip(*[to_njy(d) for d in used])
    ax.errorbar(x, yy, yerr=ee, fmt="o", color="C0", ms=7, capsize=2,
                label="used in fit", zorder=3)
    for d, yv in zip(used, yy):
        ax.annotate(d["filt"], (d["lam_rest"], yv), textcoords="offset points",
                    xytext=(4, 6), fontsize=7, color="C0")

    if res is not None:
        lam = np.linspace(lya_cut * 0.9, max(uv_max, max(x)) * 1.05, 400)
        flam = powerlaw(lam, *res["popt"]) * res["scale"]
        fnu_njy = flam * (lam * (1 + z))**2 / C_ANG / 1e-23 * 1e9
        ax.plot(lam, fnu_njy, color="C3", lw=1.5, label="power-law fit")
        ax.axhline(res["f1500_nJy"], ls="--", color="k", lw=1,
                   label=r"$f_\nu$(1500 $\rm \AA$) $\rightarrow M_{\rm UV}$")
        ax.plot(LAM0, res["f1500_nJy"], "s", color="k", ms=6, zorder=4)
        txt = (rf"$M_{{\rm UV}} = {res['M_UV']:.2f} \pm {res['M_UV_err']:.2f}$" "\n"
               rf"$\beta = {res['beta']:.2f} \pm {res['beta_err']:.2f}$" "\n"
               rf"$N_{{\rm filt}} = {len(used)}$")
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(fc="white", ec="0.7"))

    ax.axvspan(0, lya_cut, color="0.9", zorder=0)
    ax.axvline(uv_max, color="0.6", ls=":", lw=1)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(900, 6000)
    ax.set_xticks([1000, 1500, 2000, 3000, 5000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Rest-frame wavelength [$\rm \AA$]")
    ax.set_ylabel(r"$f_\nu$ [nJy]")
    ax.set_title(f"ID {src_id}   z = {z:.4f} ({z_type})", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    sec = ax.secondary_xaxis("top", functions=(lambda l: l * (1 + z) / 1e4,
                                                lambda l: l * 1e4 / (1 + z)))
    sec.set_xlabel(r"Observed wavelength [$\mu$m]")
    lo, hi = 900 * (1 + z) / 1e4, 6000 * (1 + z) / 1e4
    sec.set_xticks([t for t in (0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5) if lo <= t <= hi])
    sec.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    sec.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    a = parse_args()
    print("Inputs")
    print(f"  sources csv   {os.path.abspath(a.sources_csv)}")
    print(f"  systemic csv  {os.path.abspath(a.systemic_csv)}")
    print(f"Outputs{' (dry run, nothing written)' if a.dry_run else ''}")
    print(f"  csv           {os.path.abspath(a.out_csv)}")
    print(f"  figures       {os.path.abspath(a.fig_root)}/<ID>/<ID>_muv_beta_fit.png")
    print(f"Aperture {a.aper} {'as' if a.aper == '2' else 'mas'}, "
          f"rest-frame window: blue edge >= {a.lya_cut:.0f} A, pivot <= {a.uv_max:.0f} A\n")

    sources = pd.read_csv(a.sources_csv)
    sources["ID"] = sources["ID"].astype(int)
    systemic = pd.read_csv(a.systemic_csv)
    systemic["ID"] = systemic["ID"].astype(int)
    systemic = systemic.set_index("ID")

    ids = sources["ID"].tolist() if a.all else [a.id]
    print("Reading photometric catalogues")
    cats = load_catalogues(a.cat_dir, a.aper)
    print()

    rows, used_cols = [], []
    for src_id in ids:
        srow = sources.loc[sources["ID"] == src_id]
        if srow.empty:
            print(f"[{src_id}] not in sources csv, skipped")
            continue
        z, z_type = (pick_zsys(systemic.loc[src_id], {"OIII": a.oiii_snr_min, "Ha": a.ha_snr_min})
                     if src_id in systemic.index else (np.nan, None))
        if not np.isfinite(z):
            z, z_type = float(srow["z_av"].iloc[0]), "z_av"

        out = dict(ID=src_id, z=z, z_type=z_type, phot_catalogue=None,
                   n_filters=0, filters_used="", beta=np.nan, beta_err=np.nan,
                   f1500_nJy=np.nan, m_UV=np.nan, M_UV=np.nan, M_UV_err=np.nan,
                   chi2=np.nan, dof=np.nan, fit_flag="")

        hit = next(((n, t.loc[src_id], s) for n, t, s in cats if src_id in t.index), None)
        if hit is None:
            out["fit_flag"] = "no_photometry"
            print(f"[{src_id}] z={z:.4f} ({z_type})  not found in any FITS catalogue")
            rows.append(out)
            continue
        cname, prow, scales = hit
        out["phot_catalogue"] = cname

        used, unused = select_filters(z, scales, prow, a.aper, a.lya_cut, a.uv_max)
        out["n_filters"] = len(used)
        out["filters_used"] = ";".join(d["filt"] for d in used)
        for d in used:
            out[d["fcol"]] = d["raw_f"]
            out[d["ecol"]] = d["raw_e"]
            for c in (d["fcol"], d["ecol"]):
                if c not in used_cols:
                    used_cols.append(c)

        res = None
        if len(used) < a.min_filters:
            out["fit_flag"] = "too_few_filters"
        else:
            try:
                res = fit_source(used, z)
                for k in ("beta", "beta_err", "f1500_nJy", "m_UV", "M_UV",
                          "M_UV_err", "chi2", "dof", "err_scale"):
                    out[k] = res[k]
                out["fit_flag"] = "ok"
            except (RuntimeError, ValueError) as err:
                out["fit_flag"] = "fit_failed"
                print(f"  fit failed: {err}")

        msg = (f"[{src_id}] z={z:.4f} ({z_type})  cat={cname.split('_')[3]}  "
               f"used={out['filters_used'] or '-'}")
        if res is not None:
            msg += f"  beta={res['beta']:.2f}+/-{res['beta_err']:.2f}  " \
                   f"M_UV={res['M_UV']:.2f}+/-{res['M_UV_err']:.2f}"
        else:
            msg += f"  {out['fit_flag']}"
        print(msg)

        if not a.dry_run and used:
            png = os.path.join(a.fig_root, str(src_id), f"{src_id}_muv_beta_fit.png")
            plot_fit(src_id, z, z_type, used, unused, res, a.lya_cut, a.uv_max, png)
            print(f"  wrote {os.path.abspath(png)}")
        rows.append(out)

    # column order: ID, z, then the photometry used (in wavelength order), then results
    order = {f: i for i, f in enumerate(FILTERS)}
    used_cols.sort(key=lambda c: (order[c.split("_")[1]], c.endswith("fluxerr_corr")))
    front = ["ID", "z", "z_type", "phot_catalogue"]
    back = ["n_filters", "filters_used", "beta", "beta_err", "f1500_nJy", "m_UV",
            "M_UV", "M_UV_err", "chi2", "dof", "err_scale", "fit_flag"]
    df = pd.DataFrame(rows).reindex(columns=front + used_cols + back)

    ok = df["fit_flag"] == "ok"
    df["good_muv"] = ok & (df["M_UV_err"] < a.max_muv_err)
    df["good_beta"] = ok & (df["beta_err"] < a.max_beta_err)

    print(f"\n{ok.sum()} of {len(df)} sources fitted")
    print(df["fit_flag"].value_counts().to_string())
    print(f"\n{(df['err_scale'] > 1).sum()} fits had chi2_nu > 1, "
          f"errors inflated by sqrt(chi2_nu)")
    print(f"\nGood M_UV  (M_UV_err < {a.max_muv_err})  {df['good_muv'].sum()} of {len(df)}")
    print(f"Good beta  (beta_err < {a.max_beta_err})  {df['good_beta'].sum()} of {len(df)}")
    for col, err in (("good_muv", "M_UV_err"), ("good_beta", "beta_err")):
        bad = df[ok & ~df[col]].sort_values(err, ascending=False)
        print(f"  not {col}: " + ", ".join(f"{i} ({e:.2f})" for i, e in
                                           zip(bad["ID"], bad[err])))
    if df["good_beta"].any():
        gb = df.loc[df["good_beta"], "beta"]
        print(f"  good beta median {gb.median():.2f}, 16-84th {gb.quantile(0.16):.2f} "
              f"to {gb.quantile(0.84):.2f}")
    if not a.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
        df.to_csv(a.out_csv, index=False)
        print(f"\nwrote {os.path.abspath(a.out_csv)}")
        print(f"figures in {os.path.abspath(a.fig_root)}")


if __name__ == "__main__":
    main()