#!/usr/bin/env python3
"""
muv_beta_diagnostics.py

Checks on the photometric M_UV and beta fits from fit_muv_beta.py.

Panels of plots/muv_beta_diagnostics.png
----------------------------------------
  (a) beta_fit against z_sys. Filled points have beta_fit_err < 0.5
      (beta_fit_good), open points do not. Colour is the number of bands in
      the fit. Below z ~ 3.5 only F814W, F090W and F115W sit in the rest-UV
      window, so beta is expected to be poorly constrained there. The dashed
      line is beta = -1.92 (Pirie et al. 2025, the Paper 1 fixed slope).
  (b) muv_fit against a single-band M_UV. The single band is the one whose
      rest-frame pivot is closest to 1500 AA, among the bands that pass the
      1250 AA cut and have data, and M_UV is
          -2.5 log10(f_nu / 3631 Jy) - 5 log10(D_L / 10 pc) + 2.5 log10(1 + z).
      For a flat f_nu (beta = -2) the two agree whatever the band. Otherwise
      they differ by about -2.5 (beta + 2) log10(lambda_rest / 1500), so the
      points are coloured by the band's rest wavelength.
  (c) beta_fit_err against the number of bands, with the 0.5 limit.

The flux columns used are the ones fit_muv_beta.py wrote (<band>_flux, total
or aperture, whichever it was run with).

Inputs
------
  jwst_catalogs/muv_beta_fits.csv

Outputs
-------
  plots/muv_beta_diagnostics.png
  jwst_catalogs/muv_beta_diagnostics.csv   ID, z_sys, n_bands, single_band,
                                           single_band_lam_rest, muv_single,
                                           muv_single_err, dmuv (muv_fit - muv_single)

Usage
-----
python muv_beta_diagnostics.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import p2_common as pc
from fit_muv_beta import FILTERS, LAM0

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
BETA_PIRIE = -1.92


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits-csv", default=f"{CAT_DIR}/muv_beta_fits.csv")
    p.add_argument("--out-csv", default=f"{CAT_DIR}/muv_beta_diagnostics.csv")
    p.add_argument("--out-png", default=f"{pc.P2_ROOT}/plots/muv_beta_diagnostics.png")
    p.add_argument("--lya-cut", type=float, default=1250.0)
    p.add_argument("--max-beta-err", type=float, default=0.5)
    return p.parse_args()


def single_band_muv(row, lya_cut):
    """(band, rest pivot, M_UV, error) from the band nearest rest 1500 AA."""
    z = row["z_sys"]
    best = None
    for filt, (_, piv, blue, _red) in FILTERS.items():
        b = filt.lower()
        f, e = row.get(f"{b}_flux", np.nan), row.get(f"{b}_flux_err", np.nan)
        if not (np.isfinite(f) and np.isfinite(e) and e > 0 and f > 0):
            continue
        if blue / (1 + z) < lya_cut:
            continue
        d = abs(np.log(piv / (1 + z) / LAM0))
        if best is None or d < best[0]:
            best = (d, filt, piv / (1 + z), f, e)
    if best is None:
        return None, np.nan, np.nan, np.nan
    _, filt, lam_rest, f, e = best
    dl_pc = COSMO.luminosity_distance(z).to(u.pc).value
    m = -2.5 * np.log10(f * 1e-6 / 3631.0)
    muv = m - 5 * np.log10(dl_pc / 10.0) + 2.5 * np.log10(1 + z)
    return filt, lam_rest, muv, 2.5 / np.log(10) * e / f


def main():
    a = parse_args()
    print("muv_beta_diagnostics.py")
    print(f"  reading  {os.path.abspath(a.fits_csv)}")
    df = pd.read_csv(a.fits_csv)
    df["ID"] = df["ID"].astype(int)
    ok = df["fit_flag"] == "ok"
    d = df[ok].copy()

    sb = d.apply(lambda r: single_band_muv(r, a.lya_cut), axis=1, result_type="expand")
    d[["single_band", "single_band_lam_rest", "muv_single", "muv_single_err"]] = sb
    d["dmuv"] = d["muv_fit"] - d["muv_single"]
    good = d["beta_fit_err"] < a.max_beta_err

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) beta against z
    ax = axes[0]
    nb = d["n_bands"].astype(int)
    levels = np.arange(nb.min() - 0.5, nb.max() + 1.5)
    cmap = plt.get_cmap("viridis", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)
    ax.errorbar(d["z_sys"], d["beta_fit"], yerr=d["beta_fit_err"], fmt="none",
                ecolor="0.7", elinewidth=0.7, zorder=1)
    sc = ax.scatter(d.loc[good, "z_sys"], d.loc[good, "beta_fit"], c=nb[good],
                    cmap=cmap, norm=norm, s=28, edgecolor="k", lw=0.4, zorder=3,
                    label=rf"$\sigma_\beta < {a.max_beta_err:g}$")
    ax.scatter(d.loc[~good, "z_sys"], d.loc[~good, "beta_fit"], facecolor="white",
               edgecolor=cmap(norm(nb[~good])), s=28, lw=1.0, zorder=3,
               label=rf"$\sigma_\beta \geq {a.max_beta_err:g}$")
    ax.axhline(BETA_PIRIE, color="0.3", ls="--", lw=0.8, label=r"$\beta=-1.92$")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, ticks=np.arange(nb.min(), nb.max() + 1))
    cb.set_label("bands in fit")
    # limits from the points, so a few huge error bars don't flatten the plot
    ax.set_ylim(max(d["beta_fit"].min() - 0.7, -7), min(d["beta_fit"].max() + 0.7, 4))
    ax.set_xlabel(r"$z_{\rm sys}$")
    ax.set_ylabel(r"$\beta_{\rm fit}$")
    ax.legend(frameon=False, fontsize=8, loc="best")

    # (b) muv_fit against the single-band M_UV
    ax = axes[1]
    s = d.dropna(subset=["muv_single"])
    lo = np.nanmin([s["muv_fit"].min(), s["muv_single"].min()]) - 0.3
    hi = np.nanmax([s["muv_fit"].max(), s["muv_single"].max()]) + 0.3
    ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=0.8)
    ax.errorbar(s["muv_single"], s["muv_fit"], xerr=s["muv_single_err"],
                yerr=s["muv_fit_err"], fmt="none", ecolor="0.7", elinewidth=0.7)
    sc = ax.scatter(s["muv_single"], s["muv_fit"], c=s["single_band_lam_rest"],
                    cmap="plasma", norm=Normalize(1250, 3000), s=28,
                    edgecolor="k", lw=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"single band rest pivot [$\rm \AA$]")
    ax.set_xlim(hi, lo)
    ax.set_ylim(hi, lo)
    ax.set_xlabel(r"$M_{\rm UV}$, band nearest 1500 $\rm \AA$")
    ax.set_ylabel(r"$M_{\rm UV,fit}$")
    if len(s):
        ax.text(0.04, 0.96, rf"median $\Delta = {s['dmuv'].median():+.2f}$ mag",
                transform=ax.transAxes, va="top", fontsize=8)

    # (c) sigma_beta against the number of bands
    ax = axes[2]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.12, 0.12, len(d))
    ax.scatter(d["n_bands"] + jit, d["beta_fit_err"], s=22, color="C0",
               edgecolor="k", lw=0.4)
    ax.axhline(a.max_beta_err, color="C3", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(nb.min(), nb.max() + 1))
    ax.set_xlabel("bands in fit")
    ax.set_ylabel(r"$\sigma_{\beta}$")

    fig.suptitle(f"Photometric M_UV and beta, {int(ok.sum())} of {len(df)} fitted, "
                 f"{int(good.sum())} with sigma_beta < {a.max_beta_err:g}", fontsize=10)
    fig.tight_layout()

    png = os.path.abspath(a.out_png)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    print(f"\n  fitted              {int(ok.sum())} of {len(df)}")
    print(f"  sigma_beta < {a.max_beta_err:g}    {int(good.sum())}")
    for n, grp in d.groupby("n_bands"):
        print(f"    {int(n)} bands  N={len(grp):<3d} median sigma_beta {grp['beta_fit_err'].median():.2f}"
              f"  z {grp['z_sys'].min():.2f} to {grp['z_sys'].max():.2f}")
    if len(s):
        print(f"  muv_fit - single band  median {s['dmuv'].median():+.2f}, "
              f"scatter (NMAD) {1.4826 * np.median(np.abs(s['dmuv'] - s['dmuv'].median())):.2f} mag")
        big = s[np.abs(s["dmuv"]) > 0.5]
        if len(big):
            print("  differ by > 0.5 mag  " + ", ".join(
                f"{i} ({v:+.2f}, {b})" for i, v, b in zip(big["ID"], big["dmuv"], big["single_band"])))

    out = d[["ID", "z_sys", "n_bands", "single_band", "single_band_lam_rest",
             "muv_single", "muv_single_err", "muv_fit", "dmuv", "beta_fit",
             "beta_fit_err"]]
    csv = os.path.abspath(a.out_csv)
    out.to_csv(csv, index=False)
    print(f"\nWrote  {png}")
    print(f"Wrote  {csv}")


if __name__ == "__main__":
    main()