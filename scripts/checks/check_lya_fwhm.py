#!/usr/bin/env python3
"""
check_lya_fwhm.py

Is the Lya FWHM from fit_lya_properties_grating.py a fair measure of the
lines, or does the fit make them wider than they are? Four checks, one panel
each, for the detections (lya_det == 1 in lya_catalog.csv, in_analysis == 1).

(a) Direct width against the model width
    The FWHM is measured straight from each aperture spectrum: the spectrum
    is lightly smoothed (Gaussian, --smooth-pix, default 1 pixel), the peak is
    found within +/- --peak-search AA of the fitted peak, and the two
    half-maximum crossings either side are found by linear interpolation.
    The convolved model (what MUSE sees) is evaluated on the same pixels,
    smoothed the same way and measured with the same code. Both therefore
    carry the same smoothing and pixel sampling, so any difference between
    them is the fit, not the method. The direct error comes from --n-mc
    noise realisations of the spectrum. Sources where a crossing is not
    found inside the fit window get direct_flag = no_crossing.
(b) FWHM against S/N
    If low-S/N fits come out wider, the intrinsic FWHM falls as lya_snr
    rises. Spearman rho and p are printed and shown.
(c) FWHM against the continuum residual
    The model has no continuum term, so a positive residual in the fit
    window would be absorbed into the line wings. The residual is the mean
    flux in the fit window where the convolved model is below --outer-frac
    (5 per cent) of its peak, given as S/N and as a fraction of the line
    peak.
(d) FWHM against how often the MC refits put alpha on a bound
    frac_alpha_at_bound (from mc_lya_errors_grating.py) counts refits with
    alpha at EITHER bound, 0 (symmetric) or alpha_max (15, close to a
    half-Gaussian with a long red tail). A value near 0.5 means the refits
    jump between the two, i.e. the data do not constrain the skewness.
    Points are coloured by the fitted alpha. Sources with
    frac_sigma_at_bound > 0.1 are ringed.

Only sources with in_analysis == 1 are used, as in plot_lya_science.py, so
the close pairs and interlopers are left out.

Inputs
------
  MUSE_catalogs/lya_catalog.csv               lya_det, lya_snr, in_analysis
  MUSE_catalogs/lya_fits_mc.csv               fit parameters, fwhm_*, frac_*_at_bound
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz    wave (air), flux, var

Outputs
-------
  plots/lya_fwhm_checks_<rule>.png
  MUSE_catalogs/lya_fwhm_checks_<rule>.csv    one row per source checked
<rule> is the detection rule of the catalogue (det_rule, e.g. p95).

Usage (from scripts/checks/, or anywhere, it finds scripts/ itself)
-----
python check_lya_fwhm.py
python check_lya_fwhm.py --sample all
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import p2_common as pc                      # noqa: E402
import fit_lya_properties_grating as flp    # noqa: E402

LYA_CATALOG = getattr(pc, "LYA_CATALOG", f"{pc.MUSE_CAT_DIR}/lya_catalog.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=["detections", "non_detections", "all"],
                   default="detections")
    p.add_argument("--catalog", default=LYA_CATALOG)
    p.add_argument("--fits-csv", default=f"{pc.MUSE_CAT_DIR}/lya_fits_mc.csv")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--out-png", default=None,
                   help="default plots/lya_fwhm_checks_<rule>.png")
    p.add_argument("--out-csv", default=None,
                   help="default MUSE_catalogs/lya_fwhm_checks_<rule>.csv")
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="AA either side of best_center, as the fit (default 25)")
    p.add_argument("--smooth-pix", type=float, default=1.0,
                   help="Gaussian smoothing sigma in pixels for the direct width (default 1)")
    p.add_argument("--peak-search", type=float, default=5.0,
                   help="AA either side of the fitted peak to look for the data peak")
    p.add_argument("--outer-frac", type=float, default=0.05,
                   help="continuum residual from pixels where the model is below "
                        "this fraction of its peak (default 0.05)")
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def half_max_width(wave, prof, peak_guess, search):
    """FWHM in AA from the half-maximum crossings either side of the highest
    point within +/- search of peak_guess. NaN if a crossing is not found."""
    near = np.where(np.abs(wave - peak_guess) <= search)[0]
    if near.size == 0 or not np.any(np.isfinite(prof[near])):
        return np.nan, np.nan
    ip = near[np.nanargmax(prof[near])]
    pk = prof[ip]
    if not (np.isfinite(pk) and pk > 0):
        return np.nan, np.nan
    half = 0.5 * pk

    left = np.nan
    for i in range(ip, 0, -1):
        if prof[i - 1] < half:
            x0, x1, y0, y1 = wave[i - 1], wave[i], prof[i - 1], prof[i]
            left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break
    right = np.nan
    for i in range(ip, len(prof) - 1):
        if prof[i + 1] < half:
            x0, x1, y0, y1 = wave[i], wave[i + 1], prof[i], prof[i + 1]
            right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break
    return right - left, wave[ip]


def to_kms(width, wave):
    return width / wave * flp.C_KMS


def check_source(row, a, rng):
    out = dict(fwhm_direct_kms=np.nan, fwhm_direct_err_kms=np.nan,
               fwhm_direct_model_kms=np.nan, direct_over_model=np.nan,
               cont_resid=np.nan, cont_resid_err=np.nan, cont_resid_snr=np.nan,
               cont_resid_frac=np.nan, n_outer=0, direct_flag="")
    path = pc.muse_path(row["ID"], "spectrum", a.subcube_dir)
    if not os.path.exists(path):
        out["direct_flag"] = "no_spectrum"
        return out
    d = np.load(path, allow_pickle=True)
    wave = np.asarray(d["wave"], float)
    flux = np.asarray(d["flux"], float)
    var = np.asarray(d["var"], float)

    c = float(row["best_center"])
    sel = ((wave >= c - a.fit_window) & (wave <= c + a.fit_window)
           & np.isfinite(flux) & np.isfinite(var) & (var > 0))
    if sel.sum() < 10:
        out["direct_flag"] = "too_few_pixels"
        return out
    w, f, s = wave[sel], flux[sel], np.sqrt(var[sel])

    model = flp.skew_model_lsf(w, row["flux_fit"], row["mu"], row["sigma"],
                               row["alpha_skew"], row["sigma_lsf_ang"])
    pk_fit = float(row["lya_peak_wave"])

    def smooth(y):
        return gaussian_filter1d(y, a.smooth_pix) if a.smooth_pix > 0 else y

    wd, wpk = half_max_width(w, smooth(f), pk_fit, a.peak_search)
    wm, _ = half_max_width(w, smooth(model), pk_fit, a.peak_search)
    if np.isfinite(wm):
        out["fwhm_direct_model_kms"] = to_kms(wm, pk_fit)
    if np.isfinite(wd):
        out["fwhm_direct_kms"] = to_kms(wd, wpk)
        draws = []
        for _ in range(a.n_mc):
            wi, wpi = half_max_width(w, smooth(f + rng.normal(0, s)), pk_fit, a.peak_search)
            if np.isfinite(wi):
                draws.append(to_kms(wi, wpi))
        if len(draws) >= 10:
            out["fwhm_direct_err_kms"] = float(np.std(draws, ddof=1))
        out["direct_flag"] = "ok"
        if np.isfinite(wm):
            out["direct_over_model"] = out["fwhm_direct_kms"] / out["fwhm_direct_model_kms"]
    else:
        out["direct_flag"] = "no_crossing"

    mpk = np.nanmax(model)
    outer = model < a.outer_frac * mpk
    out["n_outer"] = int(outer.sum())
    if outer.sum() >= 3:
        m = float(np.mean(f[outer]))
        e = float(np.sqrt(np.sum(s[outer] ** 2)) / outer.sum())
        out.update(cont_resid=m, cont_resid_err=e, cont_resid_snr=m / e,
                   cont_resid_frac=m / mpk if mpk > 0 else np.nan)
    return out


def corr_text(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return "N < 5", np.nan, np.nan
    rho, p = spearmanr(x[ok], y[ok])
    return rf"Spearman $\rho={rho:+.2f}$, $p={p:.2g}$, N={ok.sum()}", rho, p


def main():
    a = parse_args()
    flag = "lya_det"
    print("check_lya_fwhm.py")
    rule = pc.det_rule(pd.read_csv(a.catalog))
    a.out_png = a.out_png or f"{pc.P2_ROOT}/plots/lya_fwhm_checks_{rule}.png"
    a.out_csv = a.out_csv or f"{pc.MUSE_CAT_DIR}/lya_fwhm_checks_{rule}.csv"
    for p_ in (a.catalog, a.fits_csv):
        print(f"  reading  {os.path.abspath(p_)}")
    print(f"  reading  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_lya_spectrum.npz")
    print(f"  writing  {os.path.abspath(a.out_png)}")
    print(f"  writing  {os.path.abspath(a.out_csv)}")
    print(f"  sample {a.sample}, in_analysis == 1, smoothing "
          f"{a.smooth_pix:g} pix, outer pixels below {a.outer_frac:g} of the model peak\n")

    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    if flag not in cat.columns or "in_analysis" not in cat.columns:
        raise SystemExit(f"{a.catalog} has no lya_det or in_analysis. Rerun build_lya_catalog.py")
    fits = pd.read_csv(a.fits_csv)
    fits["ID"] = fits["ID"].astype(int)
    need = ["ID", "best_center", "flux_fit", "mu", "sigma", "alpha_skew", "sigma_lsf_ang",
            "lya_peak_wave", "fwhm_int_kms", "fwhm_obs_kms", "fwhm_lsf_kms", "resolved",
            "fit_success"]
    missing = [c for c in need if c not in fits.columns]
    if missing:
        raise SystemExit(f"{a.fits_csv} has no {missing}")
    extra = [c for c in ("fwhm_int_kms_err_mc", "frac_alpha_at_bound",
                         "frac_sigma_at_bound", "skew_at_max") if c in fits.columns]

    df = cat[["ID", "z_sys", "lya_snr", flag, "in_analysis", "pair_status"]].merge(
        fits[need + extra], on="ID", how="left")
    df = df[df["fit_success"].astype(str).str.lower().isin(["true", "1"])]
    if a.sample == "detections":
        df = df[df[flag] == 1]
    elif a.sample == "non_detections":
        df = df[df[flag] == 0]
    else:
        df = df[df[flag].notna()]
    cp = df.loc[df["in_analysis"] == 0, "ID"].tolist()
    if cp:
        print(f"  close pairs and interlopers left out: {cp}")
    df = df[df["in_analysis"] == 1]
    df = df.sort_values("ID").reset_index(drop=True)
    print(f"  {len(df)} sources\n")

    rng = np.random.default_rng(a.seed)
    res = pd.DataFrame([check_source(r, a, rng) for _, r in df.iterrows()])
    df = pd.concat([df, res], axis=1)
    for c in ("frac_alpha_at_bound", "frac_sigma_at_bound", "fwhm_int_kms_err_mc"):
        if c not in df.columns:
            df[c] = np.nan

    print(f"  {'ID':>6} {'S/N':>5} {'int':>5} {'obs':>5} {'direct':>6} {'model':>6} "
          f"{'d/m':>5} {'resid S/N':>9} {'resid/peak':>10} {'alpha':>5} {'a@bnd':>5}")
    for _, r in df.iterrows():
        print(f"  {r['ID']:>6d} {r['lya_snr']:5.1f} {r['fwhm_int_kms']:5.0f} "
              f"{r['fwhm_obs_kms']:5.0f} {r['fwhm_direct_kms']:6.0f} "
              f"{r['fwhm_direct_model_kms']:6.0f} {r['direct_over_model']:5.2f} "
              f"{r['cont_resid_snr']:9.1f} {r['cont_resid_frac']:10.2f} "
              f"{r['alpha_skew']:5.1f} {r['frac_alpha_at_bound']:5.2f}"
              + ("" if r["direct_flag"] == "ok" else f"  {r['direct_flag']}"))

    ok = df["direct_flag"] == "ok"
    ratio = df.loc[ok, "direct_over_model"].dropna()
    print("")
    if len(ratio):
        q16, q50, q84 = np.percentile(ratio, [16, 50, 84])
        print(f"  (a) direct / model width   median {q50:.2f}  (16-84: {q16:.2f}-{q84:.2f}), "
              f"N={len(ratio)}")
        print(f"      median FWHM  direct {df.loc[ok, 'fwhm_direct_kms'].median():.0f}  "
              f"model {df.loc[ok, 'fwhm_direct_model_kms'].median():.0f} km/s (both smoothed), "
              f"fwhm_obs {df['fwhm_obs_kms'].median():.0f}, fwhm_int {df['fwhm_int_kms'].median():.0f}")
    t_b, _, _ = corr_text(np.log10(df["lya_snr"].to_numpy(float)), df["fwhm_int_kms"].to_numpy(float))
    t_c, _, _ = corr_text(df["cont_resid_snr"].to_numpy(float), df["fwhm_int_kms"].to_numpy(float))
    t_d, _, _ = corr_text(df["frac_alpha_at_bound"].to_numpy(float), df["fwhm_int_kms"].to_numpy(float))
    plain = lambda t: t.replace("$", "").replace(r"\rho", "rho")   # noqa: E731
    print(f"  (b) FWHM_int vs log S/N    {plain(t_b)}")
    print(f"  (c) FWHM_int vs residual   {plain(t_c)}")
    print(f"      residual S/N > 2: {int((df['cont_resid_snr'] > 2).sum())}, "
          f"< -2: {int((df['cont_resid_snr'] < -2).sum())}")
    print(f"  (d) FWHM_int vs alpha@bound {plain(t_d)}")
    print(f"      alpha at a bound (0 or max) in >50% of refits: "
          f"{int((df['frac_alpha_at_bound'] > 0.5).sum())}")
    bad = df.loc[~ok, ["ID", "direct_flag"]].values.tolist()
    if bad:
        print(f"  no direct width: {bad}")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.5))
    snr = df["lya_snr"].to_numpy(float)

    A = ax[0, 0]
    d = df[ok]
    lim = [0, 1.08 * np.nanmax([d["fwhm_direct_kms"].max(), d["fwhm_direct_model_kms"].max(), 100])]
    A.plot(lim, lim, color="0.4", ls="--", lw=0.8, label="1:1")
    A.errorbar(d["fwhm_direct_model_kms"], d["fwhm_direct_kms"],
               yerr=d["fwhm_direct_err_kms"], fmt="none", ecolor="0.7", elinewidth=0.7)
    sc = A.scatter(d["fwhm_direct_model_kms"], d["fwhm_direct_kms"], c=np.log10(d["lya_snr"]),
                   cmap="viridis", s=26, edgecolor="k", lw=0.4, zorder=3)
    fig.colorbar(sc, ax=A, pad=0.02).set_label(r"log$_{10}\,\Sigma_{\rm Ly\alpha}$")
    A.set_xlim(lim)
    A.set_ylim(lim)
    A.set_xlabel("model FWHM, measured like the data [km/s]")
    A.set_ylabel("direct FWHM from the spectrum [km/s]")
    A.set_title("(a) direct against model width", fontsize=10)
    if len(ratio):
        A.text(0.04, 0.96, f"median direct/model = {np.median(ratio):.2f}",
               transform=A.transAxes, va="top", fontsize=8)
    A.legend(frameon=False, loc="lower right", fontsize=8)

    B = ax[0, 1]
    B.errorbar(snr, df["fwhm_int_kms"], yerr=df["fwhm_int_kms_err_mc"], fmt="o", ms=4,
               color="C0", ecolor="0.7", elinewidth=0.7, mec="k", mew=0.4)
    B.set_xscale("log")
    B.set_xlabel(r"$\Sigma_{\rm Ly\alpha}$")
    B.set_ylabel(r"FWHM$_{\rm int}$ [km/s]")
    B.set_title("(b) width against S/N", fontsize=10)
    B.text(0.96, 0.96, t_b, transform=B.transAxes, va="top", ha="right",
           fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

    C = ax[1, 0]
    sc = C.scatter(df["cont_resid_snr"], df["fwhm_int_kms"], c=df["cont_resid_frac"],
                   cmap="coolwarm", vmin=-0.3, vmax=0.3, s=26, edgecolor="k", lw=0.4)
    fig.colorbar(sc, ax=C, pad=0.02).set_label("residual / line peak")
    C.axvline(0, color="0.5", lw=0.8)
    for x in (-2, 2):
        C.axvline(x, color="0.7", ls=":", lw=0.8)
    C.set_xlabel("continuum residual in the fit window [S/N]")
    C.set_ylabel(r"FWHM$_{\rm int}$ [km/s]")
    C.set_title("(c) width against continuum residual", fontsize=10)
    C.text(0.96, 0.96, t_c, transform=C.transAxes, va="top", ha="right",
           fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

    D = ax[1, 1]
    sc = D.scatter(df["frac_alpha_at_bound"], df["fwhm_int_kms"], c=df["alpha_skew"],
                   cmap="magma", vmin=0, vmax=15, s=26, edgecolor="k", lw=0.4, zorder=3)
    ring = df["frac_sigma_at_bound"] > 0.1
    if ring.any():
        D.scatter(df.loc[ring, "frac_alpha_at_bound"], df.loc[ring, "fwhm_int_kms"],
                  facecolor="none", edgecolor="C2", s=90, lw=1.2, zorder=4,
                  label=r"$\sigma$ at bound in >10% of refits")
        D.legend(frameon=False, loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=D, pad=0.02).set_label(r"fitted $\alpha$")
    D.set_xlim(-0.05, 1.05)
    D.set_xlabel(r"fraction of MC refits with $\alpha$ at a bound (0 or 15)")
    D.set_ylabel(r"FWHM$_{\rm int}$ [km/s]")
    D.set_title("(d) width against skewness at a bound", fontsize=10)
    D.text(0.96, 0.96, t_d, transform=D.transAxes, va="top", ha="right",
           fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

    lab = {"detections": "detections", "non_detections": "non-detections", "all": "all tested"}
    fig.suptitle(f"Lya FWHM checks, {rule} {lab[a.sample]}, N = {len(df)}", fontsize=11)
    fig.tight_layout()
    png = os.path.abspath(a.out_png)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    cols = ["ID", "z_sys", "lya_snr", flag, "fwhm_int_kms", "fwhm_int_kms_err_mc",
            "fwhm_obs_kms", "fwhm_lsf_kms", "resolved", "fwhm_direct_kms",
            "fwhm_direct_err_kms", "fwhm_direct_model_kms", "direct_over_model",
            "cont_resid", "cont_resid_err", "cont_resid_snr", "cont_resid_frac", "n_outer",
            "alpha_skew", "frac_alpha_at_bound", "frac_sigma_at_bound", "direct_flag"]
    csv = os.path.abspath(a.out_csv)
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    df[[c for c in cols if c in df.columns]].to_csv(csv, index=False)
    print(f"\nWrote  {png}")
    print(f"Wrote  {csv}")


if __name__ == "__main__":
    main()