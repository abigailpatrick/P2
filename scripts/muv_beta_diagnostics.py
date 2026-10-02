#!/usr/bin/env python3
"""
Sample-level checks on the beta / M_UV fits written by fit_muv_beta.py.

Looking at fits one by one cannot tell a noisy source from a problem with the
method. This script asks four questions of the whole sample.

  1. Are the fits statistically acceptable?
     chi2 of each fit, converted to a p-value for its degrees of freedom.
     p < 0.01 means the points scatter more than their errors allow.

  2. Is the HST photometry consistent with NIRCam?
     For each source with >= 2 NIRCam filters in the window, the power law is
     fit to the NIRCam points alone and used to predict each HST point. The
     ratio data / prediction is averaged over sources (inverse-variance
     weighted, with the prediction error included). A ratio of 1 means HST
     and NIRCam agree. A ratio of 1.15 means that HST filter reads 15 per
     cent high relative to NIRCam, i.e. a calibration or aperture-correction
     offset. NIRCam is the anchor, so this cannot tell whether NIRCam itself
     is offset, only whether the two disagree. The median normalised residual
     (data - model)/error from the full fit is also given for every filter.

  3. Does beta depend on which telescope we use?
     Each source is refit with NIRCam filters only. beta(all) against
     beta(NIRCam) should follow the 1:1 line within the errors.

  4. Is beta actually constrained?
     sigma_beta against the lever arm, lambda_max / lambda_min (rest frame) of
     the used filters. A short lever arm gives a poorly constrained slope even
     when every point has a small error.

Flags written per source (a source can have several, separated by ';')
  poor_fit      chi2 p-value < 0.01
  sigma_beta    sigma_beta >= 0.5, the Paper 1 threshold
  short_lever   lever arm < 1.5
  beta_extreme  beta < -3 or beta > 1, outside the range of normal stellar populations
  few_filters   fewer than 3 filters, so no degrees of freedom to test the fit

Inputs
  muv_beta_by_JELS_ID.csv (from fit_muv_beta.py) and the four FITS catalogues,
  which are re-read only to get the flux units.

Outputs
  --out-csv   per-source diagnostics and flags
  --out-png   six-panel summary figure

Usage
  python checks/muv_beta_diagnostics.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fit_muv_beta as F  # noqa: E402

P2 = "/ceph/cephfs/apatrick/P2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits-csv", default=f"{F.CAT_DIR}/muv_beta_by_JELS_ID.csv")
    p.add_argument("--cat-dir", default=F.CAT_DIR)
    p.add_argument("--aper", default="600")
    p.add_argument("--lya-cut", type=float, default=1250.0)
    p.add_argument("--uv-max", type=float, default=3000.0)
    p.add_argument("--out-csv", default=f"{F.CAT_DIR}/muv_beta_diagnostics.csv")
    p.add_argument("--out-png", default=f"{P2}/plots/muv_beta_diagnostics.png")
    return p.parse_args()


def model_fnu_njy(lam_rest, f1500_njy, beta):
    """f_lambda propto lambda^beta, so f_nu propto lambda^(beta + 2)."""
    return f1500_njy * (lam_rest / F.LAM0) ** (beta + 2)


def main():
    a = parse_args()
    print(f"  fits csv  {os.path.abspath(a.fits_csv)}")
    print(f"  out csv   {os.path.abspath(a.out_csv)}")
    print(f"  out png   {os.path.abspath(a.out_png)}\n")

    fits = pd.read_csv(a.fits_csv)
    cats = {n: (t, s) for n, t, s in F.load_catalogues(a.cat_dir, a.aper)}
    print()

    rows, resid = [], []
    for _, r in fits[fits["fit_flag"] == "ok"].iterrows():
        sid, z = int(r["ID"]), float(r["z"])
        tab, scales = cats[r["phot_catalogue"]]
        used, _ = F.select_filters(z, scales, tab.loc[sid], a.aper, a.lya_cut, a.uv_max)

        # residuals against the fit already in the csv
        lam = np.array([d["lam_rest"] for d in used])
        fnu = np.array([d["fnu_jy"] for d in used]) * 1e9
        efnu = np.array([d["efnu_jy"] for d in used]) * 1e9
        norm = (fnu - model_fnu_njy(lam, r["f1500_nJy"], r["beta"])) / efnu
        for d, n in zip(used, norm):
            resid.append(dict(ID=sid, filt=d["filt"], inst=F.FILTERS[d["filt"]][0],
                              resid=n, ratio=np.nan, ratio_err=np.nan))

        dof = len(used) - 2
        chi2 = float(np.sum(norm**2))
        pval = chi2_dist.sf(chi2, dof) if dof > 0 else np.nan

        # NIRCam-only refit
        nc = [d for d in used if F.FILTERS[d["filt"]][0] == "NIRCam"]
        b_nc, eb_nc = np.nan, np.nan
        if len(nc) >= 2:
            try:
                res = F.fit_source(nc, z)
                b_nc, eb_nc = res["beta"], res["beta_err"]
                # predict each HST point from the NIRCam-only fit
                frac_f0 = res["M_UV_err"] * np.log(10) / 2.5
                for k, d in enumerate(used):
                    if F.FILTERS[d["filt"]][0] == "NIRCam":
                        continue
                    pred = model_fnu_njy(lam[k], res["f1500_nJy"], res["beta"])
                    frac_pred = np.hypot(frac_f0, np.log(lam[k] / F.LAM0) * res["beta_err"])
                    # error depends on the prediction, not the data, so high
                    # and low ratios get the same weight
                    rec = next(x for x in reversed(resid)
                               if x["ID"] == sid and x["filt"] == d["filt"])
                    rec["ratio"] = fnu[k] / pred
                    rec["ratio_err"] = np.hypot(efnu[k] / pred, frac_pred)
            except (RuntimeError, ValueError):
                pass

        lever = lam.max() / lam.min()
        flags = []
        if np.isfinite(pval) and pval < 0.01:
            flags.append("poor_fit")
        if r["beta_err"] >= 0.5:
            flags.append("sigma_beta")
        if lever < 1.5:
            flags.append("short_lever")
        if r["beta"] < -3 or r["beta"] > 1:
            flags.append("beta_extreme")
        if len(used) < 3:
            flags.append("few_filters")

        rows.append(dict(ID=sid, z=z, n_filters=len(used), beta=r["beta"],
                         beta_err=r["beta_err"], M_UV=r["M_UV"], M_UV_err=r["M_UV_err"],
                         chi2=chi2, dof=dof, chi2_nu=chi2 / dof if dof > 0 else np.nan,
                         p_value=pval, lever_arm=lever,
                         n_nircam=len(nc), beta_nircam=b_nc, beta_nircam_err=eb_nc,
                         flags=";".join(flags)))

    df = pd.DataFrame(rows)
    rf = pd.DataFrame(resid)

    # ------------------------------------------------------------------ print
    print(f"{len(df)} fitted sources\n")
    print("Fit quality (sources with dof >= 1)")
    t = df[df["dof"] > 0]
    print(f"  median chi2_nu        {t['chi2_nu'].median():.2f}   (expect about 1)")
    print(f"  p < 0.01              {(t['p_value'] < 0.01).sum()} of {len(t)}"
          f"   (expect about {0.01 * len(t):.1f} by chance)")
    print(f"  dof = 0, untestable   {(df['dof'] == 0).sum()}\n")

    print("Per-filter checks")
    print("  ratio  HST only. Weighted mean of data / prediction from the NIRCam-only")
    print("         fit. 1.00 = agrees with NIRCam, 1.15 = reads 15 per cent high")
    print("  scat   chi2_nu of the ratios about their mean. >> 1 means more scatter")
    print("         between sources than the errors allow")
    print("  resid  median (data - model)/error in the full fit, expect about 0")
    g = rf.groupby("filt")["resid"]
    lo = rf.dropna(subset=["ratio", "ratio_err"])

    def wmean(x):
        w = 1 / x["ratio_err"] ** 2
        m = np.sum(w * x["ratio"]) / w.sum()
        return pd.Series({"N_ratio": len(x), "ratio": m, "ratio_err": 1 / np.sqrt(w.sum()),
                          "scat": np.sum(w * (x["ratio"] - m) ** 2) / max(len(x) - 1, 1)})
    lw = lo.groupby("filt")[["ratio", "ratio_err"]].apply(wmean) if len(lo) else pd.DataFrame()
    summ = pd.DataFrame({"inst": rf.groupby("filt")["inst"].first(),
                         "N": g.size(), "resid": g.median()}).join(lw)
    summ = summ.loc[[f for f in F.FILTERS if f in summ.index]]
    print(summ.to_string(float_format=lambda x: f"{x:6.2f}", na_rep="     -"))
    print()

    both = df.dropna(subset=["beta_nircam"])
    d = both["beta"] - both["beta_nircam"]
    print("beta(all) - beta(NIRCam only)")
    print(f"  N = {len(both)}, median = {d.median():+.2f}, "
          f"NMAD = {1.4826 * np.median(np.abs(d - d.median())):.2f}\n")

    print("Flags")
    for f in ["poor_fit", "sigma_beta", "short_lever", "beta_extreme", "few_filters"]:
        ids = df.loc[df["flags"].str.contains(f), "ID"].tolist()
        print(f"  {f:13} {len(ids):3d}  {ids}")
    clean = df["flags"] == ""
    print(f"  no flags      {clean.sum():3d}")
    print(f"\nbeta of unflagged sources: median {df.loc[clean, 'beta'].median():.2f}, "
          f"16-84th {df.loc[clean, 'beta'].quantile(0.16):.2f} to "
          f"{df.loc[clean, 'beta'].quantile(0.84):.2f}")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    a1, a2, a3, a4, a5, a6 = ax.ravel()

    order = list(summ.index)
    data = [rf.loc[rf["filt"] == f, "resid"].values for f in order]
    bp = a1.boxplot(data, positions=range(len(order)), showfliers=True, patch_artist=True)
    for patch, f in zip(bp["boxes"], order):
        patch.set_facecolor("C0" if summ.loc[f, "inst"] == "NIRCam" else "C1")
    for k, f in enumerate(order):
        if np.isfinite(summ.loc[f].get("ratio", np.nan)):
            a1.text(k, 0.97, f"{summ.loc[f, 'ratio']:.2f}", ha="center", va="top",
                    fontsize=9, color="C1", fontweight="bold",
                    transform=matplotlib.transforms.blended_transform_factory(
                        a1.transData, a1.transAxes))
    a1.set_xticks(range(len(order)))
    a1.set_xticklabels(order, rotation=45)
    a1.axhline(0, color="k", lw=0.8)
    a1.axhspan(-1, 1, color="0.9", zorder=0)
    lo_, hi_ = a1.get_ylim()
    a1.set_ylim(lo_, hi_ + 0.15 * (hi_ - lo_))
    a1.set_ylabel("(data - model) / error")
    a1.set_title("Residuals (blue NIRCam, orange HST)\nnumbers: HST / NIRCam-predicted", fontsize=10)

    ok = t["chi2_nu"].notna()
    for k, c in zip(sorted(t["dof"].unique()), plt.cm.viridis(np.linspace(0, 0.9, 6))):
        sel = t[t["dof"] == k]
        a2.hist(np.log10(sel["chi2_nu"]), bins=np.linspace(-2, 2, 25), histtype="step",
                lw=1.5, color=c, label=f"dof = {k}")
    a2.axvline(0, color="k", ls="--", lw=0.8)
    a2.set_xlabel(r"$\log_{10}\,\chi^2_\nu$")
    a2.set_title(r"Fit quality ($\chi^2_\nu \approx 1$ expected)")
    a2.legend(fontsize=8)

    a3.errorbar(both["beta_nircam"], both["beta"], xerr=both["beta_nircam_err"],
                yerr=both["beta_err"], fmt="o", ms=4, alpha=0.6, lw=0.8)
    lim = [-4, 1.5]
    a3.plot(lim, lim, "k--", lw=0.8)
    a3.set_xlim(lim); a3.set_ylim(lim)
    a3.set_xlabel(r"$\beta$ (NIRCam only)")
    a3.set_ylabel(r"$\beta$ (NIRCam + HST)")
    a3.set_title("Does HST change beta?")

    sc = a4.scatter(df["lever_arm"], df["beta_err"], c=df["n_filters"], cmap="viridis", s=25)
    a4.axhline(0.5, color="C3", ls="--", lw=0.8, label=r"$\sigma_\beta = 0.5$")
    a4.axvline(1.5, color="0.5", ls=":", lw=0.8)
    a4.set_yscale("log")
    a4.set_xlabel(r"lever arm $\lambda_{\rm max}/\lambda_{\rm min}$ (rest)")
    a4.set_ylabel(r"$\sigma_\beta$")
    a4.set_title("Is beta constrained?")
    a4.legend(fontsize=8)
    fig.colorbar(sc, ax=a4, label="N filters")

    flag = df["flags"] != ""
    a5.errorbar(df.loc[~flag, "z"], df.loc[~flag, "beta"], yerr=df.loc[~flag, "beta_err"],
                fmt="o", ms=4, color="C0", label="no flag")
    a5.errorbar(df.loc[flag, "z"], df.loc[flag, "beta"], yerr=df.loc[flag, "beta_err"],
                fmt="o", ms=4, mfc="none", color="0.5", label="flagged")
    a5.axhline(-1.92, color="C3", ls="--", lw=0.8, label=r"$\beta=-1.92$ (Paper 1)")
    a5.set_ylim(-4.5, 2)
    a5.set_xlabel("z")
    a5.set_ylabel(r"$\beta$")
    a5.legend(fontsize=8)

    a6.errorbar(df.loc[~flag, "M_UV"], df.loc[~flag, "beta"], xerr=df.loc[~flag, "M_UV_err"],
                yerr=df.loc[~flag, "beta_err"], fmt="o", ms=4, color="C0")
    a6.errorbar(df.loc[flag, "M_UV"], df.loc[flag, "beta"], fmt="o", ms=4,
                mfc="none", color="0.5")
    a6.set_ylim(-4.5, 2)
    a6.invert_xaxis()
    a6.set_xlabel(r"$M_{\rm UV}$")
    a6.set_ylabel(r"$\beta$")
    a6.set_title(r"$\beta$ - $M_{\rm UV}$ (brighter to the right)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_png)), exist_ok=True)
    fig.savefig(a.out_png, dpi=130)
    plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"\nwrote {os.path.abspath(a.out_csv)}")
    print(f"wrote {os.path.abspath(a.out_png)}")


if __name__ == "__main__":
    main()