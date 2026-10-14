#!/usr/bin/env python3
"""
fit_muv_beta.py

Measure the UV slope beta and absolute UV magnitude M_UV of every source in
the z_sys sample by fitting a power law to Isaac's PRIMER + MINERVA rest-frame
UV photometry (Holst et al. in prep.). These are the photometric fits
(muv_fit, beta_fit). The SED-based M_UV from Ken's fits comes later.

Method (Paper 1, Sect. 2.4, after Pirie et al. 2025)
----------------------------------------------------
For each source the filters probing the rest-frame UV continuum are fitted with

    f_lambda = f0 (lambda_rest / 1500 AA)^beta

by weighted non-linear least squares (scipy curve_fit, absolute_sigma=True),
in linear flux with Isaac's errors as they are. If the points scatter more
than their errors allow (chi2_nu > 1) the parameter errors are inflated by
sqrt(chi2_nu), never shrunk (err_scale records the factor). With two bands
there are no degrees of freedom and err_scale is 1.

f0 is f_lambda at rest-frame 1500 AA. It is converted to f_nu at the observed
1500(1+z), then to an apparent AB magnitude, then

    M_UV = m_1500 - 5 log10(D_L / 10 pc) + 2.5 log10(1 + z)

No dust correction is applied, as in Paper 1.

Filter selection, per source
----------------------------
A band is used only if
  1. its blue half-power edge is redward of --lya-cut (1250 AA) in the rest
     frame, so the whole band is clear of Lya and the IGM break,
  2. its pivot is at or below --uv-max (3000 AA) in the rest frame, so it
     samples the UV continuum and not the Balmer break,
  3. the flux and error are finite and the error is positive. -99 in Isaac's
     catalogue (band not imaged) is treated as missing.
Faint and negative fluxes are kept. At least --min-bands (2) are needed.

Photometry
----------
Read straight from cosmos_primer_minerva_production.fits by Number (= ID),
in uJy, MW-extinction corrected, PSF-matched. --phot total (default) uses
aperture flux x total_correction, so M_UV is a total magnitude. beta is a
colour and is the same either way. --phot aperture uses the 0.3 / 0.5 arcsec
aperture fluxes. Isaac's Flag is carried through. Digits 1 to 3 mean the
photometry may be affected (phot_flag = 1), the fit is still made.

Redshift
--------
z_sys from primer_minerva_in_muse_zsys.csv.

Columns of the output CSV
-------------------------
  ID, z_sys, isaac_flag, phot_flag, total_correction, n_bands, bands_used,
  beta_fit, beta_fit_err          UV slope
  muv_fit, muv_fit_err            absolute UV magnitude (total if --phot total)
  m_uv_fit                        apparent AB magnitude at rest 1500 AA
  f1500_fit, f1500_fit_err        f_nu at rest 1500 AA (observed frame), nJy
  chi2, dof, err_scale
  fit_flag                        ok, too_few_bands, no_photometry, fit_failed
  beta_fit_good                   1 if fit_flag ok and beta_fit_err < 0.5 (Paper 1)
  muv_fit_good                    1 if fit_flag ok and muv_fit_err < 0.5
  <band>_flux, <band>_flux_err    the flux used for every one of Isaac's 19
                                  bands (total or aperture, uJy), NaN where
                                  the band is not imaged

Where the numbers come from
---------------------------
  filter pivots and half-power edges   JDox (NIRCam), SVO FPS (HST, NIRCam medium)
  power law, lambda0 = 1500 AA, 3000 AA  Paper 1 Sect. 2.4 / Pirie et al. (2025)
  AB magnitudes                        Oke & Gunn (1983)
  M_UV distance modulus + 2.5 log(1+z) Hogg et al. (2002)
  H0 = 70, Om = 0.3                    Paper 1
  sigma_beta < 0.5                     Paper 1 Sect. 5
  1250 AA cut                          adopted here

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv
  jwst_catalogs/cosmos_primer_minerva_production.fits

Outputs
-------
  jwst_catalogs/muv_beta_fits.csv              one row per source (overwritten)
  MUSE_subcubes/<ID>/<ID>_muv_beta_fit.png     one plot per source

Usage
-----
python fit_muv_beta.py
python fit_muv_beta.py --dry-run             # print the fits, write nothing
python fit_muv_beta.py --ids 23070 50803     # replace only these rows
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

import p2_common as pc

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
ISAAC_FITS = f"{CAT_DIR}/cosmos_primer_minerva_production.fits"
OUT_CSV = f"{CAT_DIR}/muv_beta_fits.csv"
FIG_NAME = "muv_beta_fit.png"

# ---------------------------------------------------------------------------
# Filter wavelengths, copied from the cited pages, not estimated.
#
# NIRCam wide bands: JWST User Documentation (JDox), "NIRCam Filters" table,
#   pivot and blue and red half-power wavelengths, micron.
# NIRCam medium bands: SVO Filter Profile Service, JWST/NIRCam.<FILT>,
#   WavelengthPivot, WavelengthCen, FWHM in AA (edges = Cen -/+ FWHM/2).
# HST/ACS: SVO, HST/ACS_WFC.<FILT>, same PARAMs.
# Only Isaac's 19 bands are listed.
# ---------------------------------------------------------------------------
NIRCAM_JDOX = {   # filter -> (pivot, lambda-, lambda+), micron
    "F090W": (0.901, 0.795, 1.005),
    "F115W": (1.154, 1.013, 1.282),
    "F150W": (1.501, 1.331, 1.668),
    "F200W": (1.990, 1.755, 2.227),
    "F277W": (2.786, 2.423, 3.132),
    "F356W": (3.563, 3.135, 3.981),
    "F410M": (4.092, 3.866, 4.302),
    "F444W": (4.421, 3.881, 4.982),
}
NIRCAM_SVO = {    # filter -> (pivot, centre, FWHM), AA
    "F140M": (14053.232536752, 14055.471891322, 1471.7453241243),
    "F162M": (16272.470828066, 16279.44469614, 1710.0509900877),
    "F182M": (18451.671760304, 18451.438545284, 2450.0535142807),
    "F210M": (20954.513115415, 20970.264415414, 2092.9556602331),
    "F250M": (25032.325766786, 25035.347980395, 1824.9614459531),
    "F300M": (29891.21134832, 29930.663423494, 3276.8424330832),
    "F360M": (36241.760882275, 36217.330681244, 3855.305234812),
    "F460M": (46299.277451031, 46317.872804161, 2329.5607509154),
}
HST_SVO = {       # filter -> (pivot, centre, FWHM), AA
    "F435W": (4329.8468812521, 4369.6175314903, 900.04349333548),
    "F606W": (5921.8841189137, 5962.1998266378, 2253.4024368618),
    "F814W": (8045.5275046827, 8102.9069995926, 2098.146800223),
}


def build_filter_table():
    """filter -> (instrument, pivot, blue edge, red edge) in AA, sorted by pivot."""
    tab = {}
    for f, (piv, lo, hi) in NIRCAM_JDOX.items():
        tab[f] = ("NIRCam", piv * 1e4, lo * 1e4, hi * 1e4)
    for f, (piv, cen, fwhm) in NIRCAM_SVO.items():
        tab[f] = ("NIRCam", piv, cen - fwhm / 2, cen + fwhm / 2)
    for f, (piv, cen, fwhm) in HST_SVO.items():
        tab[f] = ("ACS", piv, cen - fwhm / 2, cen + fwhm / 2)
    return dict(sorted(tab.items(), key=lambda kv: kv[1][1]))


FILTERS = build_filter_table()
BANDS = [f.lower() for f in FILTERS]       # Isaac's column names

LAM0 = 1500.0                  # rest-frame anchor, AA
C_ANG = 2.99792458e18          # speed of light, AA / s
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", type=int, nargs="+",
                   help="fit only these IDs and replace their rows in --out-csv")
    p.add_argument("--dry-run", action="store_true",
                   help="print the band selection and fits, write nothing")
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--isaac", default=ISAAC_FITS)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR,
                   help=f"plots go to <subcube-dir>/<ID>/<ID>_{FIG_NAME}")
    p.add_argument("--phot", default="total", choices=["total", "aperture"])
    p.add_argument("--lya-cut", type=float, default=1250.0,
                   help="rest-frame blue edge limit, AA (default 1250)")
    p.add_argument("--uv-max", type=float, default=3000.0,
                   help="rest-frame pivot limit, AA (default 3000)")
    p.add_argument("--min-bands", type=int, default=2)
    p.add_argument("--max-beta-err", type=float, default=0.5)
    p.add_argument("--max-muv-err", type=float, default=0.5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_isaac(path, ids, phot):
    """Isaac's rows for the IDs, indexed by Number, -99 set to NaN, fluxes
    scaled to total if asked. Returns the table and the IDs not found."""
    need = ["Number", "Flag", "total_correction"] + BANDS + [f"{b}_err" for b in BANDS]
    t = Table.read(path)
    missing = [c for c in need if c not in t.colnames]
    if missing:
        raise SystemExit(f"ERROR {path} has no {missing}")
    num = np.asarray(t["Number"]).astype(int)
    keep = np.isin(num, list(ids))
    df = t[keep][need].to_pandas()
    df["Number"] = df["Number"].astype(int)
    df = df.set_index("Number")
    for b in BANDS:
        f = df[b].astype(float)
        e = df[f"{b}_err"].astype(float)
        bad = (f <= -98) | (e <= -98) | ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
        f[bad], e[bad] = np.nan, np.nan
        if phot == "total":
            f, e = f * df["total_correction"], e * df["total_correction"]
        df[b], df[f"{b}_err"] = f, e
    return df, sorted(set(ids) - set(df.index))


def phot_affected(flag):
    """1 if any digit of Isaac's Flag is 1, 2 or 3."""
    try:
        s = str(int(flag))
    except (TypeError, ValueError):
        return 0
    return int(any(c in "123" for c in s))


def select_bands(z, row, lya_cut, uv_max):
    """(used, unused) lists of dicts for one source. Missing bands are left out."""
    used, unused = [], []
    for filt, (_, piv, blue, _red) in FILTERS.items():
        b = filt.lower()
        f, e = row[b], row[f"{b}_err"]
        if not (np.isfinite(f) and np.isfinite(e) and e > 0):
            continue
        d = dict(filt=filt, lam_obs=piv, lam_rest=piv / (1 + z),
                 fnu_jy=f * 1e-6, efnu_jy=e * 1e-6)
        ok = (blue / (1 + z) >= lya_cut) and (piv / (1 + z) <= uv_max)
        (used if ok else unused).append(d)
    return used, unused


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def powerlaw(lam_rest, f0, beta):
    return f0 * (lam_rest / LAM0) ** beta


def fit_source(used, z):
    """Weighted power-law fit in f_lambda. Returns a dict of results."""
    lam_obs = np.array([d["lam_obs"] for d in used])
    lam_rest = lam_obs / (1 + z)
    fnu = np.array([d["fnu_jy"] for d in used])
    efnu = np.array([d["efnu_jy"] for d in used])

    flam = fnu * 1e-23 * C_ANG / lam_obs**2          # erg/s/cm2/AA, observed
    eflam = efnu * 1e-23 * C_ANG / lam_obs**2
    scale = np.nanmax(np.abs(flam))
    y, ey = flam / scale, eflam / scale

    p0 = [max(np.median(y), 0.1), -2.0]
    popt, pcov = curve_fit(powerlaw, lam_rest, y, p0=p0, sigma=ey,
                           absolute_sigma=True,
                           bounds=([1e-6, -6.0], [np.inf, 4.0]), maxfev=20000)
    f0, beta = popt
    dof = len(used) - 2
    chi2 = float(np.sum(((y - powerlaw(lam_rest, *popt)) / ey) ** 2))
    err_scale = np.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0
    f0_err, beta_err = np.sqrt(np.diag(pcov)) * err_scale

    lam1500_obs = LAM0 * (1 + z)
    to_fnu_njy = scale * lam1500_obs**2 / C_ANG / 1e-23 * 1e9
    f1500 = f0 * to_fnu_njy
    f1500_err = f0_err * to_fnu_njy
    m1500 = -2.5 * np.log10(f1500 * 1e-9 / 3631.0)
    dl_pc = COSMO.luminosity_distance(z).to(u.pc).value
    muv = m1500 - 5 * np.log10(dl_pc / 10.0) + 2.5 * np.log10(1 + z)
    muv_err = 2.5 / np.log(10) * f0_err / f0
    return dict(beta_fit=float(beta), beta_fit_err=float(beta_err),
                f1500_fit=float(f1500), f1500_fit_err=float(f1500_err),
                m_uv_fit=float(m1500), muv_fit=float(muv), muv_fit_err=float(muv_err),
                chi2=chi2, dof=dof, err_scale=float(err_scale),
                popt=popt, scale=scale)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_fit(sid, z, used, unused, res, lya_cut, uv_max, phot, out_png):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    def njy(ds):
        return ([d["lam_rest"] for d in ds], [d["fnu_jy"] * 1e9 for d in ds],
                [d["efnu_jy"] * 1e9 for d in ds])

    if unused:
        x, y, e = njy(unused)
        ax.errorbar(x, y, yerr=e, fmt="o", mfc="none", color="0.6", ms=6,
                    capsize=2, label="not used")
    if used:
        x, y, e = njy(used)
        ax.errorbar(x, y, yerr=e, fmt="o", color="C0", ms=7, capsize=2,
                    label="used in fit", zorder=3)
        for k, (d, yv) in enumerate(zip(used, y)):
            ax.annotate(d["filt"], (d["lam_rest"], yv), textcoords="offset points",
                        xytext=(4, 6 if k % 2 == 0 else -13), fontsize=7, color="C0")

    if res is not None:
        lam = np.linspace(lya_cut * 0.9, uv_max * 1.05, 400)
        flam = powerlaw(lam, *res["popt"]) * res["scale"]
        fnu_njy = flam * (lam * (1 + z)) ** 2 / C_ANG / 1e-23 * 1e9
        ax.plot(lam, fnu_njy, color="C3", lw=1.5, label="power-law fit")
        ax.plot(LAM0, res["f1500_fit"], "s", color="k", ms=6, zorder=4,
                label=r"$f_\nu$(1500 $\rm \AA$)")
        txt = (rf"$M_{{\rm UV}} = {res['muv_fit']:.2f} \pm {res['muv_fit_err']:.2f}$" "\n"
               rf"$\beta = {res['beta_fit']:.2f} \pm {res['beta_fit_err']:.2f}$" "\n"
               rf"$N_{{\rm bands}} = {len(used)}$")
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(fc="white", ec="0.7"))
    else:
        ax.text(0.97, 0.05, f"no fit ({len(used)} bands in the window)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

    ax.axvspan(0, lya_cut, color="0.9", zorder=0)
    ax.axvline(uv_max, color="0.6", ls=":", lw=1)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(700, 6000)
    ax.set_xticks([1000, 1500, 2000, 3000, 5000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Rest-frame wavelength [$\rm \AA$]")
    ax.set_ylabel(rf"$f_\nu$ [nJy] ({phot})")
    ax.set_title(f"ID {sid}   z_sys = {z:.4f}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    sec = ax.secondary_xaxis("top", functions=(lambda l: l * (1 + z) / 1e4,
                                                lambda l: l * 1e4 / (1 + z)))
    sec.set_xlabel(r"Observed wavelength [$\mu$m]")
    lo, hi = 700 * (1 + z) / 1e4, 6000 * (1 + z) / 1e4
    sec.set_xticks([t for t in (0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5) if lo <= t <= hi])
    sec.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    sec.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    a = parse_args()
    print("fit_muv_beta.py")
    for p_ in (a.zsys, a.isaac):
        print(f"  reading  {os.path.abspath(p_)}")
    tag = "  (dry run, nothing written)" if a.dry_run else ""
    print(f"  writing  {os.path.abspath(a.out_csv)}{tag}")
    print(f"  writing  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME}{tag}")
    print(f"  {a.phot} fluxes, rest-frame window blue edge >= {a.lya_cut:.0f} AA, "
          f"pivot <= {a.uv_max:.0f} AA, at least {a.min_bands} bands\n")

    sample = pc.load_sample(a.zsys, ids=a.ids, need=("ID", "z_sys"))
    isaac, not_found = load_isaac(a.isaac, sample["ID"].tolist(), a.phot)
    if not_found:
        print(f"  WARNING not in Isaac's catalogue: {not_found}\n")

    rows = []
    for _, s in sample.iterrows():
        sid, z = int(s["ID"]), float(s["z_sys"])
        out = dict(ID=sid, z_sys=z, isaac_flag=np.nan, phot_flag=np.nan,
                   total_correction=np.nan,
                   n_bands=0, bands_used="",
                   beta_fit=np.nan, beta_fit_err=np.nan,
                   muv_fit=np.nan, muv_fit_err=np.nan, m_uv_fit=np.nan,
                   f1500_fit=np.nan, f1500_fit_err=np.nan,
                   chi2=np.nan, dof=np.nan, err_scale=np.nan, fit_flag="")
        if sid not in isaac.index:
            out["fit_flag"] = "no_photometry"
            print(f"  {sid:>6d}  z={z:.4f}  not in Isaac's catalogue")
            rows.append(out)
            continue

        r = isaac.loc[sid]
        out["isaac_flag"] = int(r["Flag"])
        out["phot_flag"] = phot_affected(r["Flag"])
        out["total_correction"] = float(r["total_correction"])
        for b in BANDS:
            out[f"{b}_flux"], out[f"{b}_flux_err"] = r[b], r[f"{b}_err"]

        used, unused = select_bands(z, r, a.lya_cut, a.uv_max)
        out["n_bands"] = len(used)
        out["bands_used"] = " ".join(d["filt"] for d in used)
        res = None
        if len(used) < a.min_bands:
            out["fit_flag"] = "too_few_bands"
        else:
            try:
                res = fit_source(used, z)
                for k in ("beta_fit", "beta_fit_err", "muv_fit", "muv_fit_err",
                          "m_uv_fit", "f1500_fit", "f1500_fit_err", "chi2", "dof",
                          "err_scale"):
                    out[k] = res[k]
                out["fit_flag"] = "ok"
            except (RuntimeError, ValueError) as err:
                out["fit_flag"] = "fit_failed"
                print(f"  {sid:>6d}  fit failed: {err}")

        msg = (f"  {sid:>6d}  z={z:.4f}  flag={out['isaac_flag']}  "
               f"{out['n_bands']} bands [{out['bands_used']}]  ")
        msg += (f"beta={out['beta_fit']:.2f}+/-{out['beta_fit_err']:.2f}  "
                f"M_UV={out['muv_fit']:.2f}+/-{out['muv_fit_err']:.2f}"
                if res is not None else out["fit_flag"])
        print(msg)

        if not a.dry_run:
            png = os.path.abspath(f"{a.subcube_dir}/{sid}/{sid}_{FIG_NAME}")
            plot_fit(sid, z, used, unused, res, a.lya_cut, a.uv_max, a.phot, png)
        rows.append(out)

    df = pd.DataFrame(rows)
    ok = df["fit_flag"] == "ok"
    df["beta_fit_good"] = (ok & (df["beta_fit_err"] < a.max_beta_err)).astype(int)
    df["muv_fit_good"] = (ok & (df["muv_fit_err"] < a.max_muv_err)).astype(int)
    front = [c for c in df.columns if not c.endswith(("_flux", "_flux_err"))]
    flux = [c for b in BANDS for c in (f"{b}_flux", f"{b}_flux_err") if c in df.columns]
    df = df[front + flux]

    print(f"\n  sources          {len(df)}")
    for flag, n in df["fit_flag"].value_counts().items():
        print(f"  {flag:<16} {n}")
    print(f"  beta_fit_good    {int(df['beta_fit_good'].sum())}  "
          f"(beta_fit_err < {a.max_beta_err:g})")
    print(f"  muv_fit_good     {int(df['muv_fit_good'].sum())}  "
          f"(muv_fit_err < {a.max_muv_err:g})")
    poor = df[ok & (df["beta_fit_good"] == 0)].sort_values("beta_fit_err", ascending=False)
    if len(poor):
        print("  poorly constrained beta  " + ", ".join(
            f"{i} ({e:.2f}, {n} bands)" for i, e, n in
            zip(poor["ID"], poor["beta_fit_err"], poor["n_bands"])))
    if df["phot_flag"].eq(1).any():
        print(f"  Isaac flag 1-3   {df.loc[df['phot_flag'] == 1, 'ID'].tolist()}")

    if a.dry_run:
        print("\nDry run, nothing written")
        return

    path = os.path.abspath(a.out_csv)
    if a.ids and os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["ID"].astype(int).isin(df["ID"])]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values("ID").reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nWrote  {path}")
    print(f"Wrote  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_NAME} "
          f"({len(sample)} plots)")


if __name__ == "__main__":
    main()