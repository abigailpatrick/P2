#!/usr/bin/env python3
"""
measure_lya_fesc.py

Lya escape fractions of the z_sys sample (Paper 1 Eq. 7),

    f_esc = F_Lya / (8.7 F_Ha,corr)

with F_Lya the observed (not dust-corrected) Lya flux in the 0.6" MUSE
aperture and F_Ha,corr the total, dust-corrected Halpha flux from
build_ha_flux_corrections.py. 8.7 is the Case B Lya/Halpha ratio
(--lya-ha-int). Both lines are at the same redshift, so the flux ratio is
the luminosity ratio. Run after build_lya_catalog.py and
build_ha_flux_corrections.py, then rerun build_lya_catalog.py to join the
result.

Lya detections (lya_det = 1)
----------------------------
F_Lya = lya_flux from lya_catalog.csv (the preferred fit, the whole line,
blue + red for a double), with lya_flux_err.

Lya non-detections (lya_det = 0), 5 sigma upper limits (Paper 1 Eq. 6)
----------------------------------------------------------------------
    F_lim = 5 sigma_rms sqrt(N_pix) dlambda
sigma_rms is the per-pixel noise, the square root of the mean variance of
the aperture spectrum over the window, dlambda the pixel width and
N_pix = FWHM / dlambda. The window is centred on Lya at z_sys (air
wavelength, as the cube) shifted by the median Delta_v of the detections
(--ul-centre median_dv, or systemic), and is the median observed FWHM of
the detections wide (fwhm_obs_kms, so the LSF is included). The limit is
f_esc < F_lim / (8.7 F_Ha,corr) with the best F_Ha,corr.

Errors
------
--n-mc draws. F_Lya is drawn from a Gaussian with its error, F_Ha,corr is
taken draw by draw from ha_flux_corrections_mc.npz, so the slit-factor,
line-flux and dust uncertainties all carry through. The value is the best
estimate, the errors value - p16 and p84 - value of the draws (clipped at
0), and fesc_lya_err their mean. Limits have no error.

Every source also gets fesc_lya_sed, the same but with Halpha corrected by
the SED A_Ha whatever its case, so the dust methods can be compared.

Columns of lya_fesc.csv
-----------------------
  ID, z_sys, det_rule (the rule of the lya_catalog.csv used), in_analysis
  fesc_status        detection, upper_limit, no_ha (no usable Halpha),
                     not_tested (Lya not tested), no_lya_flux
  fesc_lya, fesc_lya_err_lo, fesc_lya_err_hi, fesc_lya_err, fesc_lya_is_limit
  fesc_lya_sed, fesc_lya_sed_err_lo, fesc_lya_sed_err_hi
  lya_flux_used, lya_flux_used_err      1e-20 erg/s/cm2, the flux of a detection
  lya_flux_lim_5sig                     1e-20 erg/s/cm2, every tested source
  ha_case, ha_source, dust_method, balmer_spectra, a_ha, a_ha_err_lo, a_ha_err_hi,
  ha_flux_corr, ha_flux_corr_err_lo, ha_flux_corr_err_hi   (from build_ha_flux_corrections.py)

Sample statistics (printed)
---------------------------
For in_analysis == 1 with Halpha, the median f_esc with the reverse
Kaplan-Meier estimator (lifelines, as Paper 1), detections as values and
non-detections as upper limits, with --n-boot bootstrap errors. Also by
ha_case, with SED dust throughout, and for detections alone.

Inputs
------
  MUSE_catalogs/lya_catalog.csv
  jwst_catalogs/ha_flux_corrections.csv, ha_flux_corrections_mc.npz
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz        for the upper limits

Outputs
-------
  MUSE_catalogs/lya_fesc.csv
  plots/lya_fesc.png      f_esc against z and against M_UV, symbol by Halpha case

Usage
-----
python measure_lya_fesc.py
python measure_lya_fesc.py --ul-centre systemic
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p2_common as pc

HA_CSV = f"{pc.P2_ROOT}/jwst_catalogs/ha_flux_corrections.csv"
HA_NPZ = f"{pc.P2_ROOT}/jwst_catalogs/ha_flux_corrections_mc.npz"
OUT_CSV = f"{pc.MUSE_CAT_DIR}/lya_fesc.csv"
OUT_PNG = f"{pc.P2_ROOT}/plots/lya_fesc.png"
HA_KEEP = ["ha_case", "ha_source", "dust_method", "balmer_spectra", "a_ha", "a_ha_err_lo",
           "a_ha_err_hi", "ha_flux_corr", "ha_flux_corr_err_lo", "ha_flux_corr_err_hi",
           "ha_flux_corr_sed"]
CASE_STYLE = {1: ("o", "#2a9d8f", "case 1, spec + Balmer"), 2: ("s", "#3b6fb6", "case 2, spec + SED"),
              3: ("^", "#e39b3b", "case 3, phot + SED")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.LYA_CATALOG)
    p.add_argument("--ha-csv", default=HA_CSV)
    p.add_argument("--ha-npz", default=HA_NPZ)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--plot", default=OUT_PNG)
    p.add_argument("--lya-ha-int", type=float, default=8.7, help="Case B Lya/Halpha (default 8.7)")
    p.add_argument("--muse-flux-unit", type=float, default=1e-20,
                   help="erg/s/cm2 per unit of lya_flux (default 1e-20)")
    p.add_argument("--ul-sigma", type=float, default=5.0)
    p.add_argument("--ul-centre", choices=["median_dv", "systemic"], default="median_dv")
    p.add_argument("--n-mc", type=int, default=5000,
                   help="Lya flux draws (Halpha draws come from the npz, recycled if fewer)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args()


def errs(val, draws):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size < 10 or not np.isfinite(val):
        return np.nan, np.nan
    p16, p84 = np.percentile(d, [16, 84])
    return max(val - p16, 0.0), max(p84 - val, 0.0)


def flux_limit(sid, centre, fwhm_kms, nsig, subcube_dir):
    """5 sigma flux limit (spectrum units) in a window of FWHM about centre."""
    path = pc.muse_path(sid, "spectrum", subcube_dir)
    if not os.path.exists(path):
        return np.nan
    d = np.load(path, allow_pickle=True)
    w, v = np.asarray(d["wave"], float), np.asarray(d["var"], float)
    dl = float(np.median(np.diff(w)))
    width = centre * fwhm_kms / pc.C_KMS
    m = (np.abs(w - centre) <= width / 2) & np.isfinite(v) & (v > 0)
    if m.sum() < 2:
        return np.nan
    sig = np.sqrt(np.mean(v[m]))
    return float(nsig * sig * np.sqrt(width / dl) * dl)


def km_median(vals, is_lim):
    """Reverse Kaplan-Meier median of left-censored data (upper limits),
    as Paper 1: negate, fit, negate. NaN if lifelines is missing."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return np.nan
    vals, is_lim = np.asarray(vals, float), np.asarray(is_lim, bool)
    if vals.size < 3 or (~is_lim).sum() == 0:
        return np.nan
    k = KaplanMeierFitter()
    k.fit(-vals, event_observed=~is_lim)
    m = k.median_survival_time_
    return float(-m) if np.isfinite(m) else np.nan


def km_report(lab, d, col, n_boot, rng):
    d = d[d[col].notna()]
    vals, lim = d[col].to_numpy(float), d["fesc_lya_is_limit"].to_numpy(int) == 1
    med = km_median(vals, lim)
    if not np.isfinite(med):
        return
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boots.append(km_median(vals[i], lim[i]))
    lo, hi = np.nanpercentile(boots, [16, 84])
    print(f"    {lab:<34s} N {len(d):>3d} ({int((~lim).sum())} det, {int(lim.sum())} limits)  "
          f"median {med:.3f} (+{hi - med:.3f} -{med - lo:.3f})")


def main():
    a = parse_args()
    print("measure_lya_fesc.py")
    for p_ in (a.catalog, a.ha_csv, a.ha_npz):
        print(f"  reading  {os.path.abspath(p_)}")
    print(f"  reading  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_lya_spectrum.npz (limits)")
    for p_ in (a.out, a.plot):
        print(f"  writing  {os.path.abspath(p_)}")

    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    rule = pc.det_rule(cat)
    ha = pd.read_csv(a.ha_csv)
    ha["ID"] = ha["ID"].astype(int)
    npz = np.load(a.ha_npz)
    draw_ix = {int(i): k for k, i in enumerate(npz["ID"])}
    rng = np.random.default_rng(a.seed)

    ana_det = cat[(cat["in_analysis"] == 1) & (cat["lya_det"] == 1)]
    fwhm = float(np.nanmedian(ana_det["fwhm_obs_kms"])) if len(ana_det) else 300.0
    dv_c = float(np.nanmedian(ana_det["delta_v"])) if (a.ul_centre == "median_dv" and len(ana_det)) else 0.0
    print(f"  rule {rule}, f_esc = F_Lya / ({a.lya_ha_int:g} F_Ha,corr)")
    print(f"  upper limits {a.ul_sigma:g} sigma, window {fwhm:.0f} km/s (median observed FWHM of "
          f"{len(ana_det)} detections) centred {dv_c:+.0f} km/s from systemic Lya")

    # the catalogue may already carry these columns (blank, or from an earlier
    # run), so drop them and take them fresh from the Halpha file
    old = [c for c in HA_KEEP + ["fesc_lya", "fesc_status"] if c in cat.columns]
    m = cat.drop(columns=old).merge(ha[["ID"] + HA_KEEP], on="ID", how="left")
    rows = []
    for _, r in m.iterrows():
        sid = int(r["ID"])
        o = dict(ID=sid, z_sys=r["z_sys"], det_rule=rule, in_analysis=r["in_analysis"],
                 fesc_lya_is_limit=0)
        for c in HA_KEEP:
            o[c] = r[c]
        tested = r["lya_status"] == "ok"
        if tested:
            centre = pc.lya_obs_air(r["z_sys"]) * (1 + dv_c / pc.C_KMS)
            o["lya_flux_lim_5sig"] = flux_limit(sid, centre, fwhm, a.ul_sigma, a.subcube_dir)
        has_ha = np.isfinite(r["ha_flux_corr"]) and r["ha_flux_corr"] > 0
        if not tested:
            o["fesc_status"] = "not_tested"
        elif not has_ha:
            o["fesc_status"] = "no_ha"
        elif r["lya_det"] == 1:
            f, e = r["lya_flux"], r["lya_flux_err"]
            if not np.isfinite(f):
                o["fesc_status"] = "no_lya_flux"
            else:
                o["fesc_status"] = "detection"
                o.update(lya_flux_used=f, lya_flux_used_err=e)
                e = e if np.isfinite(e) else 0.0
                k = draw_ix.get(sid)
                for tag, hcol, key in (("", "ha_flux_corr", "ha_flux_corr"),
                                       ("_sed", "ha_flux_corr_sed", "ha_flux_corr_sed")):
                    hv = r[hcol]
                    if not (np.isfinite(hv) and hv > 0):
                        continue
                    hd = npz[key][k] if k is not None else np.full(1, hv)
                    hd = np.resize(hd, a.n_mc)
                    ld = rng.normal(f, e, a.n_mc) * a.muse_flux_unit
                    val = f * a.muse_flux_unit / (a.lya_ha_int * hv)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dd = ld / (a.lya_ha_int * hd)
                    lo, hi = errs(val, dd)
                    o.update({f"fesc_lya{tag}": val, f"fesc_lya{tag}_err_lo": lo,
                              f"fesc_lya{tag}_err_hi": hi})
                o["fesc_lya_err"] = np.nanmean([o.get("fesc_lya_err_lo", np.nan),
                                                o.get("fesc_lya_err_hi", np.nan)])
        else:
            lim = o.get("lya_flux_lim_5sig", np.nan)
            o["fesc_status"] = "upper_limit" if np.isfinite(lim) else "no_lya_flux"
            if np.isfinite(lim):
                o["fesc_lya_is_limit"] = 1
                o["fesc_lya"] = lim * a.muse_flux_unit / (a.lya_ha_int * r["ha_flux_corr"])
                hs = r["ha_flux_corr_sed"]
                if np.isfinite(hs) and hs > 0:
                    o["fesc_lya_sed"] = lim * a.muse_flux_unit / (a.lya_ha_int * hs)
        rows.append(o)

    cols = ["ID", "z_sys", "det_rule", "in_analysis", "fesc_status", "fesc_lya", "fesc_lya_err_lo",
            "fesc_lya_err_hi", "fesc_lya_err", "fesc_lya_is_limit", "fesc_lya_sed",
            "fesc_lya_sed_err_lo", "fesc_lya_sed_err_hi", "lya_flux_used", "lya_flux_used_err",
            "lya_flux_lim_5sig"] + [c for c in HA_KEEP if c != "ha_flux_corr_sed"]
    out = pd.DataFrame(rows).reindex(columns=cols).sort_values("ID").reset_index(drop=True)
    report(out, cat, a, rng)
    for p_ in (a.out, a.plot):
        os.makedirs(os.path.dirname(os.path.abspath(p_)), exist_ok=True)
    out.to_csv(a.out, index=False)
    plot(out, cat, a.plot, rule)
    print(f"\nWrote  {os.path.abspath(a.out)}")
    print(f"Wrote  {os.path.abspath(a.plot)}")


def report(out, cat, a, rng):
    ana = out[out["in_analysis"] == 1]
    print("\n  status (in_analysis == 1)")
    for st, n in ana["fesc_status"].value_counts().items():
        print(f"    {st:<12s} {n:>3d}")
    nh = ana[ana["fesc_status"] == "no_ha"]
    if len(nh):
        lyd = cat.set_index("ID").reindex(nh["ID"])["lya_det"]
        print(f"    no_ha among Lya detections: {nh.loc[lyd.values == 1, 'ID'].tolist()}")

    d = ana[ana["fesc_status"] == "detection"].sort_values("fesc_lya")
    if len(d):
        print(f"\n  detections ({len(d)})")
        print(f"    {'ID':>6s} {'z':>5s} {'case':>4s} {'dust':>6s} {'spectra':>9s} {'A_Ha':>5s} "
              f"{'f_esc':>7s} {'-':>6s} {'+':>6s} {'f_esc SED':>9s}")
        for _, r in d.iterrows():
            print(f"    {int(r['ID']):>6d} {r['z_sys']:5.2f} {int(r['ha_case']):>4d} "
                  f"{r['dust_method']:>6s} {str(r['balmer_spectra']):>9s} {r['a_ha']:5.2f} "
                  f"{r['fesc_lya']:7.3f} {r['fesc_lya_err_lo']:6.3f} {r['fesc_lya_err_hi']:6.3f} "
                  f"{r['fesc_lya_sed']:9.3f}")
        print("\n  detections, median f_esc by Halpha case")
        for c in (1, 2, 3):
            s = d[d["ha_case"] == c]
            if len(s):
                print(f"    case {c}  N {len(s):>3d}  median {s['fesc_lya'].median():.3f}  "
                      f"(SED dust {s['fesc_lya_sed'].median():.3f})")
        c1 = d[(d["ha_case"] == 1) & d["fesc_lya_sed"].notna()]
        if len(c1):
            r_ = c1["fesc_lya"] / c1["fesc_lya_sed"]
            print(f"    case 1, f_esc Balmer / f_esc SED: median {r_.median():.2f}, "
                  f"16-84th {r_.quantile(.16):.2f} to {r_.quantile(.84):.2f}")
        hi = d[d["fesc_lya"] > 1]
        if len(hi):
            print(f"    f_esc > 1: {hi['ID'].tolist()} (check the slit factor, dust and apertures)")
    u = ana[ana["fesc_status"] == "upper_limit"]
    if len(u):
        print(f"\n  upper limits ({len(u)}): f_esc < {u['fesc_lya'].min():.3f} to < "
              f"{u['fesc_lya'].max():.3f}, median < {u['fesc_lya'].median():.3f}")

    print("\n  sample median f_esc, reverse Kaplan-Meier with bootstrap errors (in_analysis == 1)")
    if importlib.util.find_spec("lifelines") is None:
        print("    lifelines not installed, skipped (pip install lifelines)")
        return
    use = ana[ana["fesc_status"].isin(["detection", "upper_limit"])]
    km_report("all, preferred dust", use, "fesc_lya", a.n_boot, rng)
    km_report("all, SED dust throughout", use.assign(fesc_lya=use["fesc_lya_sed"]), "fesc_lya",
              a.n_boot, rng)
    for c in (1, 2, 3):
        km_report(f"case {c}", use[use["ha_case"] == c], "fesc_lya", a.n_boot, rng)


def plot(out, cat, path, rule):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    d = out[(out["in_analysis"] == 1) & out["fesc_status"].isin(["detection", "upper_limit"])]
    muv = cat.set_index("ID")["sed_muv"] if "sed_muv" in cat.columns else None
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.0), sharey=True)
    for ax, xk in zip(axes, ("z_sys", "muv")):
        for case, (mk, col, lab) in CASE_STYLE.items():
            s = d[d["ha_case"] == case]
            x = s["z_sys"].to_numpy(float) if xk == "z_sys" else (
                muv.reindex(s["ID"]).to_numpy(float) if muv is not None else np.full(len(s), np.nan))
            det = s["fesc_lya_is_limit"].to_numpy(int) == 0
            y = s["fesc_lya"].to_numpy(float)
            yerr = np.nan_to_num(np.vstack([s["fesc_lya_err_lo"], s["fesc_lya_err_hi"]]).astype(float))
            ax.errorbar(x[det], y[det], yerr=yerr[:, det], fmt=mk, ms=4.5, mfc=col, mec="k",
                        ecolor="0.6", lw=0.6, label=f"{lab} ({int(det.sum())})")
            if (~det).any():
                ax.errorbar(x[~det], y[~det], yerr=0.4 * y[~det], uplims=True, fmt=mk, ms=4.5,
                            mfc="white", mec=col, ecolor=col, color=col, lw=0.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"$z_{\rm sys}$" if xk == "z_sys" else r"$M_{\rm UV}$ (SED)")
        if xk == "muv":
            ax.invert_xaxis()
    axes[0].set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
    axes[0].legend(frameon=False, fontsize=6, loc="best")
    fig.suptitle(f"{rule}, open symbols are 5$\\sigma$ upper limits", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()