#!/usr/bin/env python3
"""
build_sed_catalog.py

Read Ken's BAGPIPES SED fits of the z_sys sample, check them against the
sample, and write one CSV of SED properties with plain column names for the
Lya catalogue (build_lya_catalog.py) to join.

The fits were run on Isaac's PRIMER + MINERVA photometry, keyed on Isaac's
Number (= ID here), at a fixed redshift (zred_fixed). They are assumed to use
total fluxes, as the interim muv_fit does, so no aperture correction is
applied (to be confirmed with Ken).

Checks
------
  1. IDs. Every Number appears once, and the SED IDs and the z_sys sample IDs
     match. Missing sources on either side are listed.
  2. Positions. RA/Dec of the SED table against the z_sys sample. Anything
     more than --pos-tol (0.1") apart is listed, which would mean the table
     is keyed on a different catalogue version.
  3. Redshift. zred_fixed against the current z_sys, as a velocity. Sources
     more than --dz-tol-kms (100 km/s) apart are listed and get
     sed_zred_ok = 0. An SED fit hardly changes over 100 km/s, so this only
     catches a fit made at an old or wrong redshift.
  4. Posteriors. Every p16 <= p50 <= p84 and finite. sed_ok = 0 otherwise.

Columns of the output CSV
-------------------------
Each quantity q has sed_<q> (the posterior median, p50), sed_<q>_err_lo
(p50 - p16) and sed_<q>_err_hi (p84 - p50), both positive, and sed_<q>_err,
their mean, for plots that need one number. For M_UV err_lo is towards the
brighter (more negative) side.

  ID, z_sys, sed_zred, dv_sed_zsys_kms, sed_zred_ok, sed_ok,
  sed_muv          M_UV, AB mag                  (MUV)
  sed_logM         log10 M*/Msun, surviving mass (log10_mass_surviving)
  sed_logsfr10     log10 SFR/(Msun/yr), 10 Myr   (log10_sfr10)
  sed_logsfr100    log10 SFR/(Msun/yr), 100 Myr  (log10_sfr100)
  sed_logssfr10    log10 sSFR/yr, 10 Myr         (log10_ssfr10_surviving)
  sed_logssfr100   log10 sSFR/yr, 100 Myr        (log10_ssfr100_surviving)
  sed_av           A_V, mag                      (A_V)
  sed_aha          A_Halpha, mag                 (A_Ha)
  sed_logxion      log10 xi_ion/(Hz/erg)         (log10_xion)
  sed_log_evidence, sed_log_likelihood           (carried for reference)

The 10 Myr SFR and sSFR are often poorly constrained, as they depend on a
recent burst the photometry barely sees. The 100 Myr values are the ones to
use for binning.

Comparison plot
---------------
sed_muv against the interim photometric muv_fit (fit_muv_beta.py, total
fluxes), with the 1:1 line, and muv_fit - sed_muv against z_sys. Open points
have muv_fit_err >= 0.5. The median offset, robust scatter and every source
more than --muv-flag (0.5 mag) apart are printed.

Inputs
------
  jwst_catalogs/primer_minerva_full_in_muse_with_grating_zsys_ceridwen_output.fits
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  jwst_catalogs/muv_beta_fits.csv                   muv_fit (optional, plot only)

Outputs
-------
  jwst_catalogs/sed_properties.csv
  plots/sed_muv_comparison.png

Usage
-----
python build_sed_catalog.py --dry-run     # checks only, nothing written
python build_sed_catalog.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.table import Table

import p2_common as pc

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
SED_FITS = f"{CAT_DIR}/primer_minerva_full_in_muse_with_grating_zsys_ceridwen_output.fits"
SED_CSV = f"{CAT_DIR}/sed_properties.csv"
MUV_FITS = f"{CAT_DIR}/muv_beta_fits.csv"
PLOT = f"{pc.P2_ROOT}/plots/sed_muv_comparison.png"

# output name -> column stem in Ken's table (stem_p16, stem_p50, stem_p84)
QUANTITIES = {
    "muv": "MUV",
    "logM": "log10_mass_surviving",
    "logsfr10": "log10_sfr10",
    "logsfr100": "log10_sfr100",
    "logssfr10": "log10_ssfr10_surviving",
    "logssfr100": "log10_ssfr100_surviving",
    "av": "A_V",
    "aha": "A_Ha",
    "logxion": "log10_xion",
}
CARRY = {"sed_log_evidence": "log_evidence", "sed_log_likelihood": "log_likelihood"}

try:
    import cmasher as cmr
    CMAP = cmr.get_sub_cmap("cmr.torch", 0.15, 0.85)
except ImportError:
    CMAP = plt.get_cmap("viridis")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sed-fits", default=SED_FITS)
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--muv-csv", default=MUV_FITS)
    p.add_argument("--out", default=SED_CSV)
    p.add_argument("--plot", default=PLOT)
    p.add_argument("--dz-tol-kms", type=float, default=100.0,
                   help="flag fits whose zred_fixed is further than this from z_sys")
    p.add_argument("--pos-tol", type=float, default=0.1,
                   help="arcsec, list sources whose RA/Dec differ by more than this")
    p.add_argument("--muv-flag", type=float, default=0.5,
                   help="mag, list sources whose muv_fit and sed_muv differ by more")
    p.add_argument("--dry-run", action="store_true", help="checks only, write nothing")
    return p.parse_args()


def robust(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med))), int(v.size)


def build(sed, zs, a):
    """Checked SED table keyed on ID, plus a list of problems for printing."""
    sed = sed.copy()
    sed["ID"] = sed["Number"].astype(int)
    problems = []

    # 1. IDs
    dup = sed.loc[sed["ID"].duplicated(), "ID"].tolist()
    if dup:
        raise SystemExit(f"ERROR Numbers repeated in the SED table: {dup}")
    only_sed = sorted(set(sed["ID"]) - set(zs["ID"]))
    only_zs = sorted(set(zs["ID"]) - set(sed["ID"]))
    print(f"\n  IDs        {len(sed)} in the SED table, {len(zs)} in the z_sys sample, "
          f"{len(set(sed['ID']) & set(zs['ID']))} in both")
    if only_sed:
        print(f"    in the SED table only, dropped: {only_sed}")
        problems.append("ids")
    if only_zs:
        print(f"    in the z_sys sample only, no SED: {only_zs}")
        problems.append("ids")

    m = zs[["ID", "ra", "dec", "z_sys"]].merge(sed, on="ID", how="left")
    has = m["Number"].notna()

    # 2. positions
    cosd = np.cos(np.deg2rad(m["dec"]))
    sep = np.hypot((m["RA"] - m["ra"]) * cosd, m["Dec"] - m["dec"]) * 3600.0
    far = has & (sep > a.pos_tol)
    print(f"  positions  max separation {np.nanmax(sep[has]) if has.any() else np.nan:.3f}\"")
    if far.any():
        print(f"    more than {a.pos_tol:g}\" apart: " + ", ".join(
            f"{int(i)} ({s:.2f}\")" for i, s in zip(m.loc[far, "ID"], sep[far])))
        problems.append("positions")

    # 3. redshift
    dv = pc.dz_to_kms(m["zred_fixed"], m["z_sys"])
    big = has & (np.abs(dv) > a.dz_tol_kms)
    med_abs = np.nanmedian(np.abs(dv[has])) if has.any() else np.nan
    max_abs = np.nanmax(np.abs(dv[has])) if has.any() else np.nan
    print(f"  redshift   |zred_fixed - z_sys| median {med_abs:.1f} km/s, max {max_abs:.1f} km/s")
    if big.any():
        print(f"    more than {a.dz_tol_kms:g} km/s apart (sed_zred_ok = 0):")
        for _, r in m[big].iterrows():
            print(f"      {int(r['ID']):>6d}  z_sys {r['z_sys']:.4f}  zred_fixed "
                  f"{r['zred_fixed']:.4f}  {pc.dz_to_kms(r['zred_fixed'], r['z_sys']):8.0f} km/s")
        problems.append("redshift")

    # 4. posteriors
    out = pd.DataFrame({"ID": m["ID"], "z_sys": m["z_sys"], "sed_zred": m["zred_fixed"],
                        "dv_sed_zsys_kms": dv})
    out["sed_zred_ok"] = (has & ~big).astype(int)
    bad = ~has.copy()
    for name, stem in QUANTITIES.items():
        cols = [f"{stem}_p{p}" for p in (16, 50, 84)]
        miss = [c for c in cols if c not in m.columns]
        if miss:
            raise SystemExit(f"ERROR {a.sed_fits} has no {miss}")
        p16, p50, p84 = (m[c].astype(float) for c in cols)
        fin = np.isfinite(p16) & np.isfinite(p50) & np.isfinite(p84)
        order = fin & (p16 <= p50) & (p50 <= p84)
        odd = has & ~order
        if odd.any():
            print(f"    {name}: non-finite or unordered percentiles for {m.loc[odd, 'ID'].tolist()}")
            problems.append("posteriors")
        bad |= odd
        out[f"sed_{name}"] = p50
        out[f"sed_{name}_err_lo"] = p50 - p16
        out[f"sed_{name}_err_hi"] = p84 - p50
        out[f"sed_{name}_err"] = 0.5 * (p84 - p16)
    for new, old in CARRY.items():
        out[new] = m[old].astype(float) if old in m.columns else np.nan
    out["sed_ok"] = (~bad).astype(int)
    print(f"  posteriors {int(out['sed_ok'].sum())} of {len(out)} sources with every "
          "p16 <= p50 <= p84 finite")
    return out.sort_values("ID").reset_index(drop=True), problems


def summarise(out):
    print("\n  posterior medians across the sample (median, 16th to 84th percentile, "
          "median 1 sigma error)")
    ok = out["sed_ok"] == 1
    for name in QUANTITIES:
        v = out.loc[ok, f"sed_{name}"]
        e = out.loc[ok, f"sed_{name}_err"]
        if v.notna().any():
            print(f"    sed_{name:<12s} {v.median():8.2f}   {v.quantile(0.16):8.2f} to "
                  f"{v.quantile(0.84):8.2f}   err {e.median():.2f}")
    for name in ("logsfr10", "logsfr100"):
        e = out.loc[ok, f"sed_{name}_err"]
        print(f"    sed_{name} error > 0.5 dex for {int((e > 0.5).sum())} sources")


def comparison_plot(out, muv_csv, path, flag, write):
    if not os.path.exists(muv_csv):
        print(f"\n  {muv_csv} not found, no comparison plot")
        return
    uv = pd.read_csv(muv_csv)
    uv["ID"] = uv["ID"].astype(int)
    if "fit_flag" in uv.columns:
        uv = uv[uv["fit_flag"] == "ok"]
    d = out.merge(uv[["ID", "muv_fit", "muv_fit_err"]], on="ID", how="inner")
    d = d[(d["sed_ok"] == 1) & d["muv_fit"].notna() & d["sed_muv"].notna()]
    if d.empty:
        print("\n  no sources with both M_UV, no comparison plot")
        return
    d["dmuv"] = d["muv_fit"] - d["sed_muv"]
    good = d["muv_fit_err"] < 0.5
    med, sig, n = robust(d.loc[good, "dmuv"])
    print(f"\n  M_UV  muv_fit - sed_muv for {n} sources with muv_fit_err < 0.5: "
          f"median {med:+.2f} mag, robust sigma {sig:.2f} mag")
    far = d[np.abs(d["dmuv"]) > flag].sort_values("dmuv")
    if len(far):
        print(f"    differ by more than {flag:g} mag:")
        for _, r in far.iterrows():
            print(f"      {int(r['ID']):>6d}  z {r['z_sys']:.2f}  muv_fit {r['muv_fit']:6.2f} "
                  f"+/- {r['muv_fit_err']:.2f}  sed_muv {r['sed_muv']:6.2f}  "
                  f"diff {r['dmuv']:+.2f}")
    if not write:
        return

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 3.1))
    fig.subplots_adjust(wspace=0.35)
    sed_err = np.vstack([d["sed_muv_err_lo"], d["sed_muv_err_hi"]])
    for ax, y, yerr in ((ax1, d["muv_fit"], d["muv_fit_err"]),
                        (ax2, d["dmuv"], np.hypot(d["muv_fit_err"], d["sed_muv_err"]))):
        x = d["sed_muv"] if ax is ax1 else d["z_sys"]
        xerr = sed_err if ax is ax1 else None
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.6",
                    elinewidth=0.6, zorder=1)
        c = CMAP(norm(d["z_sys"]))
        ax.scatter(x[good], y[good], c=c[good.values], s=18, edgecolor="black",
                   linewidths=0.4, zorder=3)
        ax.scatter(x[~good], y[~good], facecolors="white", edgecolors=c[~good.values],
                   s=18, linewidths=1.0, zorder=3)
    lo = min(d["sed_muv"].min(), d["muv_fit"].min()) - 0.3
    hi = max(d["sed_muv"].max(), d["muv_fit"].max()) + 0.3
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax1.set_xlim(hi, lo)
    ax1.set_ylim(hi, lo)
    ax1.set_xlabel(r"$M_{\rm UV}$ SED")
    ax1.set_ylabel(r"$M_{\rm UV}$ power-law fit")
    ax1.set_title(f"N = {len(d)}", fontsize=9)
    ax2.axhline(0, color="k", ls="--", lw=0.8)
    if np.isfinite(med):
        ax2.axhline(med, color="0.4", ls=":", lw=0.8,
                    label=f"median {med:+.2f}, $\\sigma$ {sig:.2f}")
        ax2.legend(frameon=False, fontsize=7, loc="upper right")
    ax2.set_xlabel(r"$z_{\rm sys}$")
    ax2.set_ylabel(r"$M_{\rm UV}$ fit $-$ SED")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    fig.colorbar(sm, ax=[ax1, ax2], pad=0.02, fraction=0.03).set_label(r"$z_{\rm sys}$")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote  {path}")


def main():
    a = parse_args()
    print("build_sed_catalog.py")
    print(f"  reading  {os.path.abspath(a.sed_fits)}")
    print(f"  reading  {os.path.abspath(a.zsys)}")
    print(f"  reading  {os.path.abspath(a.muv_csv)}"
          f"{'' if os.path.exists(a.muv_csv) else '  (not found, no comparison plot)'}")
    if a.dry_run:
        print("  dry run, nothing will be written")

    sed = Table.read(a.sed_fits).to_pandas()
    zs = pc.load_sample(a.zsys, need=("ID", "ra", "dec", "z_sys"))
    out, problems = build(sed, zs, a)
    summarise(out)
    comparison_plot(out, a.muv_csv, a.plot, a.muv_flag, not a.dry_run)

    if problems:
        print(f"\n  CHECK  problems found in: {sorted(set(problems))}")
    else:
        print("\n  all checks passed")
    if a.dry_run:
        print("  dry run, nothing written")
        return
    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote  {path}")


if __name__ == "__main__":
    main()