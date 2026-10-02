#!/usr/bin/env python3
"""
Science plots for one manually labelled Lya group (a, b, c or d), or all four.

Reads
-----
  lya_group_<label>.csv              from group_lya_by_manual.py
                                     (ID, fwhm_kms, delta_v_kms, delta_v_err_kms)
  delta_v_from_best_zsys_line.csv    z_sys, merged on ID
  lya_properties_mc.csv              fwhm_kms_err_mc, merged on ID
  muv_beta_by_JELS_ID.csv            M_UV, M_UV_err, merged on ID (from
                                     fit_muv_beta.py). Only M_UV with
                                     fit_flag == ok and M_UV_err < --muv-max-err
                                     is used. Other sources keep their FWHM and
                                     dv but drop out of the M_UV plots.

Writes to --outdir, every name prefixed group_<label>_
------------------------------------------------------
Plots ending _zbins use three redshift bins (--z-edges, default 3.5 4.5).
Every one of them also has a _zbins2 version with two bins split at
--z-split (default 3.5).

  hist_fwhm.png, hist_z.png, hist_dv.png, hist_muv.png
                            distributions
  hist_fwhm_zbins[2].png    FWHM split by redshift
  hist_dv_zbins[2].png      velocity offset split by redshift
  dv_vs_fwhm.png            Delta_v against FWHM, coloured by z_sys
  dv_vs_fwhm_zbins[2].png   the same, split by redshift
  fwhm_models.png           FWHM histogram (bins uniform in log10 FWHM) with a
                            log-normal fit (median and bootstrap 16-84 band),
                            Prieto-Lyon+25, Mason+18 (FWHM = dv) and Mason+19
                            (Verhamme-scaled). Drawn as in Prieto-Lyon+25
                            Fig. 6, each curve peaks at its median. The Mason
                            curves use the median M_UV and median z_sys of the
                            sources in the plot.
  fwhm_models_zbins[2].png  the same, split by redshift, Mason curves at each
                            bin's median M_UV and z_sys
  fwhm_models_muvbins2.png  the same, split into two M_UV bins at the sample
                            median M_UV (or --muv-edges2)
  fwhm_models_muvbins3.png  the same, three M_UV bins at the sample tertiles
                            (or --muv-edges3)
  dv_vs_fwhm_verhamme.png   Delta_v against FWHM with the Verhamme+18 relation
                            (1 sigma band) and the 1:1 line
  dv_vs_fwhm_verhamme_zbins[2].png  the same, split by redshift
  dv_vs_muv.png             Delta_v against M_UV, coloured by z_sys, with the
                            Mason+18 relation drawn at integer z across the
                            sample (cf. Prieto-Lyon+25 Fig. 7 right)
  dv_vs_muv_zbins[2].png    the same, split by redshift, Mason+18 at each bin's
                            median z with its 0.24 dex scatter
  fwhm_vs_muv.png           FWHM against M_UV, coloured by z_sys, with Mason+19
                            at integer z and the Prieto-Lyon+25 median
                            (cf. Prieto-Lyon+25 Fig. 7 left)
  fwhm_vs_muv_zbins[2].png  the same, split by redshift, Mason+19 at each bin's
                            median z with its 0.30 dex scatter

--muv fixes the M_UV used for the Mason curves instead of the sample median.
--density-scaling adds the Prieto-Lyon+25 (1+z)^(2/3) scaled Mason+19 curve.
--label all runs every group whose CSV exists.

Histogram bin widths are fixed (--fwhm-bin, --dv-bin, --z-bin) and shared
across panels so the bins can be compared directly.

FWHM is the observed width of the skewed-Gaussian fit, not corrected for the
MUSE line spread function.

Usage
-----
python plot_lya_science.py --label a
python plot_lya_science.py --label all
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    import cmasher as cmr
    _HAVE_CMASHER = True
except ImportError:
    _HAVE_CMASHER = False

LABELS = ["a", "b", "c", "d"]

# ---------------------------------------------------------------------------
# Matplotlib style, MNRAS (matched to the P1 figures)
# ---------------------------------------------------------------------------
MNRAS_COL_WIDTH_IN = 3.46    # single column, 88 mm
MNRAS_PAGE_WIDTH_IN = 7.09   # double column, 180 mm

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

LABEL_FWHM = r"FWHM$_{\rm Ly\alpha}$ [km s$^{-1}$]"
LABEL_DV = r"$\Delta v_{\rm Ly\alpha}$ [km s$^{-1}$]"
LABEL_Z = r"$z_{\rm sys}$"
LABEL_MUV = r"$M_{\rm UV}$"


def get_cmap():
    """Redshift colour map, cmasher torch trimmed so neither end is too pale."""
    if _HAVE_CMASHER:
        return cmr.get_sub_cmap("cmr.torch", 0.15, 0.85)
    return plt.get_cmap("viridis")


def get_hist_colour():
    if _HAVE_CMASHER:
        return cmr.take_cmap_colors("cmr.torch", 1, cmap_range=(0.45, 0.45))[0]
    return "steelblue"


def panel_width(n):
    """Figure width for a row of n panels."""
    return {1: MNRAS_COL_WIDTH_IN, 2: 5.0}.get(n, MNRAS_PAGE_WIDTH_IN)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_group(group_csv, delta_v_csv, properties_csv, muv_csv, muv_max_err):
    """Group CSV with z_sys, the FWHM MC error and M_UV merged on."""
    grp = pd.read_csv(group_csv)
    grp["ID"] = grp["ID"].astype(int)

    dv = pd.read_csv(delta_v_csv)[["ID", "z_sys"]]
    dv["ID"] = dv["ID"].astype(int)

    props = pd.read_csv(properties_csv)[["ID", "fwhm_kms_err_mc"]]
    props["ID"] = props["ID"].astype(int)

    drop = [c for c in ("z_sys", "fwhm_kms_err_mc", "M_UV", "M_UV_err") if c in grp.columns]
    df = (grp.drop(columns=drop)
             .merge(dv, on="ID", how="left")
             .merge(props, on="ID", how="left"))

    if muv_csv and os.path.exists(muv_csv):
        m = pd.read_csv(muv_csv)
        m["ID"] = m["ID"].astype(int)
        good = (m["fit_flag"] == "ok") & (m["M_UV_err"] < muv_max_err)
        m.loc[~good, ["M_UV", "M_UV_err"]] = np.nan
        df = df.merge(m[["ID", "M_UV", "M_UV_err"]].drop_duplicates("ID"),
                      on="ID", how="left")
    else:
        print(f"[WARN] M_UV csv not found ({muv_csv}), M_UV plots will be empty")
        df["M_UV"] = np.nan
        df["M_UV_err"] = np.nan

    for col in ("z_sys", "fwhm_kms", "fwhm_kms_err_mc",
                "delta_v_kms", "delta_v_err_kms", "M_UV", "M_UV_err"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bin_masks(values, edges, var="z"):
    """Masks and panel labels for len(edges)+1 bins. NaN values fall in none."""
    edges = sorted(edges)
    sym = "z" if var == "z" else r"M_{\rm UV}"
    fmt = (lambda e: f"{e:g}") if var == "z" else (lambda e: f"{e:.2f}")
    masks, labels = [], []
    for i in range(len(edges) + 1):
        if i == 0:
            masks.append(values < edges[0])
            labels.append(rf"${sym} < {fmt(edges[0])}$")
        elif i == len(edges):
            masks.append(values >= edges[-1])
            labels.append(rf"${sym} \geq {fmt(edges[-1])}$")
        else:
            masks.append((values >= edges[i - 1]) & (values < edges[i]))
            labels.append(rf"${fmt(edges[i - 1])} \leq {sym} < {fmt(edges[i])}$")
    return masks, labels


def zbin_masks(z, edges):
    return bin_masks(z, edges, "z")


def fixed_bins(values, width):
    """Bin edges of a fixed width covering all finite values."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0, width])
    lo = np.floor(v.min() / width) * width
    hi = np.ceil(v.max() / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + 0.5 * width, width)


def median_or(values, fallback):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else fallback


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


def plain(lab):
    """Strip mathtext from a panel label for terminal printing."""
    return (lab.replace("$", "").replace(r"\leq", "<=").replace(r"\geq", ">=")
               .replace(r"M_{\rm UV}", "M_UV"))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hist(values, bins, xlabel, title, path):
    v = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.7))
    ax.hist(v, bins=bins, color=get_hist_colour(), edgecolor="black", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N")
    ax.set_title(f"{title}  (N = {len(v)})", fontsize=9)
    save(fig, path)


def plot_hist_zbins(values, z, bins, edges, xlabel, title, path):
    masks, labels = zbin_masks(z, edges)
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.4),
                             sharex=True, sharey=True)
    for ax, m, lab in zip(axes, masks, labels):
        v = values[m & np.isfinite(values)]
        ax.hist(v, bins=bins, color=get_hist_colour(), edgecolor="black", lw=0.6)
        ax.set_title(f"{lab}  (N = {len(v)})", fontsize=9)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("N")
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.subplots_adjust(wspace=0.08)
    save(fig, path)


def draw_points(ax, df, xcol, ycol, xerrcol, yerrcol, norm, cmap):
    """Error bars in grey behind, points coloured by z_sys on top."""
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    xerr = np.nan_to_num(df[xerrcol].to_numpy(), nan=0.0)
    yerr = np.nan_to_num(df[yerrcol].to_numpy(), nan=0.0)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.6",
                elinewidth=0.6, capsize=0, zorder=1)
    return ax.scatter(x, y, c=df["z_sys"], cmap=cmap, norm=norm, s=18,
                      edgecolor="black", linewidths=0.4, zorder=3)


def draw_scatter(ax, df, norm, cmap):
    return draw_points(ax, df, "fwhm_kms", "delta_v_kms", "fwhm_kms_err_mc",
                       "delta_v_err_kms", norm, cmap)


def plot_dv_vs_fwhm(df, path, title):
    d = df.dropna(subset=["fwhm_kms", "delta_v_kms", "z_sys"])
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.9))
    sc = draw_scatter(ax, d, norm, cmap)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(LABEL_Z)
    ax.set_xlabel(LABEL_FWHM)
    ax.set_ylabel(LABEL_DV)
    ax.set_title(f"{title}  (N = {len(d)})", fontsize=9)
    save(fig, path)


def plot_dv_vs_fwhm_zbins(df, edges, path, title):
    d = df.dropna(subset=["fwhm_kms", "delta_v_kms", "z_sys"])
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    masks, labels = zbin_masks(d["z_sys"].to_numpy(), edges)
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.6),
                             sharex=True, sharey=True)
    sc = None
    for ax, m, lab in zip(axes, masks, labels):
        sub = d[m]
        if len(sub):
            sc = draw_scatter(ax, sub, norm, cmap)
        ax.set_title(f"{lab}  (N = {len(sub)})", fontsize=9)
        ax.set_xlabel(LABEL_FWHM)
    axes[0].set_ylabel(LABEL_DV)
    fig.subplots_adjust(wspace=0.08)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.03)
        cb.set_label(LABEL_Z)
    fig.suptitle(title, fontsize=9, y=1.02)
    save(fig, path)


# ---------------------------------------------------------------------------
# Literature models
# ---------------------------------------------------------------------------
# Mason et al. 2018 (ApJ 856, 2) Eq. 3, approximate Delta_v(M_UV, z):
#   log10 dv = 0.32 * gamma * (M_UV + 20.0 + 0.26 z) + 2.34
#   gamma = -0.3 if M_UV >= -20.0 - 0.26 z, else -0.7
# Mason et al. 2019 (MNRAS 485, 3947) App. C4: FWHM prior is log-normal,
#   median from the above Delta_v scaled by Verhamme et al. 2018,
#   FWHM = (34 + dv) / 0.9, with 0.3 dex scatter.
# Prieto-Lyon et al. 2025/26 (arXiv:2509.18302) Eq. 5:
#   log10 FWHM ~ N(2.39, 0.30), no M_UV dependence. Their FWHM is corrected
#   for instrumental resolution in quadrature.
# Prieto-Lyon et al. Sec. 6.1 density scaling: FWHM and dv ~ (1+z)^(2/3) at
#   fixed halo mass. Normalised here to z = 2 (the Mason calibration epoch),
#   which reproduces their quoted ~1.6x at z ~ 5.
# Verhamme et al. 2018 (MNRAS 478, L60) Eq. 2:
#   V_peak = 0.9(+/-0.14) FWHM - 34(+/-60) km/s, intrinsic scatter 72 km/s.
#   Their FWHM is NOT corrected for instrumental broadening.

MASON18_SIGMA_DEX = 0.24   # Mason+18 Eq. 1 sigma_v
MASON19_SIGMA_DEX = 0.30
PL25_MU, PL25_SIGMA = 2.39, 0.30
VERH_SLOPE, VERH_SLOPE_ERR = 0.9, 0.14
VERH_INT, VERH_INT_ERR = -34.0, 60.0
VERH_SCATTER = 72.0

COL_THIS = "black"
COL_MASON = "#d9534f"
COL_MASON18 = "#8e44ad"
COL_PL = "#3b6fb6"
COL_MASON_DENS = "#e39b3b"
COL_VERH = "#2a9d8f"


def mason18_dv(muv, z):
    """Mason et al. 2018 Eq. 3 median Lya velocity offset, km/s.

    Works for a scalar or an array of M_UV. Returns a float for scalar input.
    """
    muv = np.asarray(muv, dtype=float)
    x = muv + 20.0 + 0.26 * z
    gamma = np.where(muv >= -20.0 - 0.26 * z, -0.3, -0.7)
    out = 10.0 ** (0.32 * gamma * x + 2.34)
    return float(out) if out.ndim == 0 else out


def mason19_fwhm_median(muv, z):
    """Mason et al. 2019 median Lya FWHM, km/s (M18 dv scaled by V18)."""
    return (34.0 + mason18_dv(muv, z)) / 0.9


def density_factor(z):
    """(1+z)^(2/3) density scaling, normalised to z = 2."""
    return ((1.0 + z) / 3.0) ** (2.0 / 3.0)


def lognormal_pdf_log(x, mu_log10, sigma_log10):
    """Density per dex, p(log10 x), for log10(x) ~ N(mu, sigma).

    This is the convention of Prieto-Lyon+25 Fig. 6: plotted against linear
    x, each curve peaks at its median, 10**mu. Scaled by N x (bin width in
    dex) it gives expected counts per log-uniform histogram bin.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = x > 0
    lx = np.log10(x[ok])
    out[ok] = (np.exp(-0.5 * ((lx - mu_log10) / sigma_log10) ** 2)
               / (sigma_log10 * np.sqrt(2 * np.pi)))
    return out


def fit_lognormal_band(values, grid, n_boot, rng):
    """Log-normal fit to values, with a bootstrap 16-84 band on the curve.

    Returns (mu, sigma, curve_median, curve_lo, curve_hi) on grid, where the
    curves are densities per dex.
    """
    lv = np.log10(values[np.isfinite(values) & (values > 0)])
    mu, sig = lv.mean(), lv.std(ddof=1)
    curves = np.empty((n_boot, grid.size))
    for i in range(n_boot):
        s = rng.choice(lv, size=lv.size, replace=True)
        s_sig = s.std(ddof=1)
        if not np.isfinite(s_sig) or s_sig <= 0:
            s_sig = sig
        curves[i] = lognormal_pdf_log(grid, s.mean(), s_sig)
    lo, med, hi = np.percentile(curves, [16, 50, 84], axis=0)
    return mu, sig, med, lo, hi


def draw_fwhm_models(ax, fwhm, z_med, muv, log_edges, xmax, n_boot, rng,
                     density):
    """FWHM histogram with this-work, Prieto-Lyon, Mason+18 and Mason+19.

    Histogram bins are uniform in log10 FWHM, drawn on a linear axis, so each
    log-normal peaks at its median as in Prieto-Lyon+25 Fig. 6 while staying
    normalised to the counts. Returns a dict of medians for printing.
    """
    v = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
    dlog = log_edges[1] - log_edges[0]
    ax.hist(v, bins=10 ** log_edges, color=get_hist_colour(), alpha=0.55,
            edgecolor="black", lw=0.6)
    info = {"n": len(v)}
    if len(v) < 3:
        return info

    grid = np.linspace(1.0, xmax, 800)
    scale = len(v) * dlog

    mu, sig, med, lo, hi = fit_lognormal_band(v, grid, n_boot, rng)
    ax.fill_between(grid, lo * scale, hi * scale, color="0.5", alpha=0.35, lw=0)
    ax.plot(grid, med * scale, color=COL_THIS, ls="--", lw=1.2,
            label="This work")
    info["this_median"] = 10 ** mu
    info["this_sigma"] = sig

    ax.plot(grid, lognormal_pdf_log(grid, PL25_MU, PL25_SIGMA) * scale,
            color=COL_PL, ls=":", lw=1.3, label="Prieto-Lyon+25")

    m18 = mason18_dv(muv, z_med)
    ax.plot(grid, lognormal_pdf_log(grid, np.log10(m18),
                                    MASON18_SIGMA_DEX) * scale,
            color=COL_MASON18, ls="-.", lw=1.2,
            label=r"Mason+18 (FWHM = $\Delta v$)")
    info["mason18_median"] = m18

    m19 = mason19_fwhm_median(muv, z_med)
    ax.plot(grid, lognormal_pdf_log(grid, np.log10(m19),
                                    MASON19_SIGMA_DEX) * scale,
            color=COL_MASON, ls="-.", lw=1.2, label="Mason+19 (V18 scaled)")
    info["mason19_median"] = m19

    if density:
        d_med = m19 * density_factor(z_med)
        ax.plot(grid, lognormal_pdf_log(grid, np.log10(d_med),
                                        MASON19_SIGMA_DEX) * scale,
                color=COL_MASON_DENS, ls="-.", lw=1.0,
                label=r"Mason+19 $\times\,[(1+z)/3]^{2/3}$")
        info["mason_density_median"] = d_med
    return info


def plot_fwhm_models(df, muv, log_edges, xmax, n_boot, rng, density, path,
                     title):
    """Single panel. muv is the M_UV used for the Mason curves."""
    z_med = float(np.nanmedian(df["z_sys"]))
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.8))
    info = draw_fwhm_models(ax, df["fwhm_kms"].to_numpy(), z_med, muv,
                            log_edges, xmax, n_boot, rng, density)
    ax.set_xlabel(LABEL_FWHM)
    ax.set_ylabel("N")
    ax.set_xlim(0, xmax)
    ax.set_title(f"{title}  (N = {info['n']})", fontsize=9)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=6,
                    title=rf"Mason at $M_{{\rm UV}}={muv:.2f}$, $z={z_med:.2f}$",
                    title_fontsize=6)
    leg._legend_box.align = "right"
    save(fig, path)
    return z_med, info


def plot_fwhm_models_bins(df, masks, labels, muv_fixed, muv_fallback,
                          log_edges, xmax, n_boot, rng, density, path,
                          suptitle):
    """One panel per bin. Mason curves at each bin's median z_sys and M_UV
    (or muv_fixed if given). A bin with no M_UV uses muv_fallback."""
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.6),
                             sharex=True, sharey=True)
    out = []
    for ax, m, lab in zip(axes, masks, labels):
        sub = df[m]
        z_med = float(np.nanmedian(sub["z_sys"])) if len(sub) else np.nan
        muv = muv_fixed if muv_fixed is not None else median_or(sub["M_UV"], muv_fallback)
        info = (draw_fwhm_models(ax, sub["fwhm_kms"].to_numpy(), z_med, muv,
                                 log_edges, xmax, n_boot, rng, density)
                if len(sub) and np.isfinite(z_med) else {"n": len(sub)})
        ax.set_title(f"{lab}\n(N = {info['n']})", fontsize=8)
        ax.set_xlabel(LABEL_FWHM)
        if np.isfinite(z_med):
            ax.text(0.97, 0.55, rf"$z_{{\rm med}}={z_med:.2f}$" "\n"
                    rf"$M_{{\rm UV}}={muv:.2f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        out.append((lab, z_med, muv, info))
    axes[0].set_ylabel("N")
    axes[0].set_xlim(0, xmax)
    axes[0].set_xticks(np.arange(0, xmax, 200))
    axes[-1].legend(frameon=False, loc="upper right", fontsize=5.5)
    fig.suptitle(suptitle, fontsize=9, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    save(fig, path)
    return out


def draw_verhamme(ax, fmax):
    """Verhamme+18 relation with 1 sigma band, and the 1:1 line.

    Band: slope and intercept errors (treated as independent) plus the 72 km/s
    intrinsic scatter, added in quadrature. An approximation of the band in
    Prieto-Lyon+25 Fig. 13.
    """
    f = np.linspace(0, fmax, 200)
    v = VERH_SLOPE * f + VERH_INT
    sig = np.sqrt((VERH_SLOPE_ERR * f) ** 2 + VERH_INT_ERR ** 2
                  + VERH_SCATTER ** 2)
    ax.fill_between(f, v - sig, v + sig, color=COL_VERH, alpha=0.15, lw=0)
    ax.plot(f, v, color=COL_VERH, ls="--", lw=1.0, label="Verhamme+18")
    ax.plot(f, f, color="black", ls="--", lw=0.8, label="1:1")


def dv_fwhm_limit(df):
    x = df["fwhm_kms"] + df["fwhm_kms_err_mc"].fillna(0)
    y = df["delta_v_kms"] + df["delta_v_err_kms"].fillna(0)
    return 1.05 * float(np.nanmax([x.max(), y.max()]))


def plot_dv_vs_fwhm_verhamme(df, path, title):
    d = df.dropna(subset=["fwhm_kms", "delta_v_kms", "z_sys"])
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    lim = dv_fwhm_limit(d)
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 3.0))
    draw_verhamme(ax, lim)
    sc = draw_scatter(ax, d, norm, cmap)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(LABEL_Z)
    ax.set_xlim(0, lim)
    ax.set_ylim(min(0, d["delta_v_kms"].min() * 1.1), lim)
    ax.set_xlabel(LABEL_FWHM)
    ax.set_ylabel(LABEL_DV)
    ax.set_title(f"{title}  (N = {len(d)})", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    save(fig, path)


def plot_dv_vs_fwhm_verhamme_zbins(df, edges, path, title):
    d = df.dropna(subset=["fwhm_kms", "delta_v_kms", "z_sys"])
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    masks, labels = zbin_masks(d["z_sys"].to_numpy(), edges)
    lim = dv_fwhm_limit(d)
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.7),
                             sharex=True, sharey=True)
    sc = None
    for ax, m, lab in zip(axes, masks, labels):
        draw_verhamme(ax, lim)
        sub = d[m]
        if len(sub):
            sc = draw_scatter(ax, sub, norm, cmap)
        ax.set_title(f"{lab}  (N = {len(sub)})", fontsize=9)
        ax.set_xlabel(LABEL_FWHM)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(min(0, d["delta_v_kms"].min() * 1.1), lim)
    axes[0].set_ylabel(LABEL_DV)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6)
    fig.subplots_adjust(wspace=0.08)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.03)
        cb.set_label(LABEL_Z)
    fig.suptitle(title, fontsize=9, y=1.02)
    save(fig, path)


# ---------------------------------------------------------------------------
# Delta_v and FWHM against M_UV (cf. Prieto-Lyon+25 Fig. 7)
# ---------------------------------------------------------------------------

VS_MUV = {
    # quantity -> (column, error column, axis label, model function, model
    #              scatter in dex, model legend label)
    "dv": ("delta_v_kms", "delta_v_err_kms", LABEL_DV, mason18_dv,
           MASON18_SIGMA_DEX, "Mason+18"),
    "fwhm": ("fwhm_kms", "fwhm_kms_err_mc", LABEL_FWHM, mason19_fwhm_median,
             MASON19_SIGMA_DEX, "Mason+19"),
}


def muv_limits(d):
    lo = (d["M_UV"] - d["M_UV_err"].fillna(0)).min() - 0.2
    hi = (d["M_UV"] + d["M_UV_err"].fillna(0)).max() + 0.2
    return hi, lo   # faint on the left, bright on the right


def y_limits(d, col, ecol):
    top = (d[col] + d[ecol].fillna(0)).max()
    bot = (d[col] - d[ecol].fillna(0)).min()
    return min(0.0, 1.1 * bot), 1.15 * top


def draw_pl25_fwhm(ax, label=True):
    ax.axhline(10 ** PL25_MU, color=COL_PL, ls=":", lw=1.2,
               label="Prieto-Lyon+25" if label else None)


def plot_vs_muv(df, quantity, path, title):
    """Single panel: quantity against M_UV, coloured by z_sys, with the Mason
    relation at integer redshifts spanning the sample."""
    col, ecol, ylab, model, _, mlab = VS_MUV[quantity]
    d = df.dropna(subset=[col, "M_UV", "z_sys"])
    if len(d) == 0:
        print(f"  [skip] {os.path.basename(path)}: no sources with M_UV")
        return
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    xlim = muv_limits(d)
    grid = np.linspace(min(xlim), max(xlim), 300)

    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 3.0))
    zs = np.arange(np.round(d["z_sys"].min()), np.round(d["z_sys"].max()) + 1)
    for zz in zs:
        ax.plot(grid, model(grid, zz), color=cmap(norm(zz)), lw=1.1, zorder=2,
                label=rf"{mlab}, $z={zz:g}$")
    if quantity == "fwhm":
        draw_pl25_fwhm(ax)
    sc = draw_points(ax, d, "M_UV", col, "M_UV_err", ecol, norm, cmap)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(LABEL_Z)
    ax.set_xlim(*xlim)
    ax.set_ylim(*y_limits(d, col, ecol))
    ax.set_xlabel(LABEL_MUV)
    ax.set_ylabel(ylab)
    ax.set_title(f"{title}  (N = {len(d)})", fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    save(fig, path)


def plot_vs_muv_zbins(df, quantity, edges, path, title):
    """One panel per redshift bin, Mason relation at each bin's median z with
    its log-normal scatter as a band."""
    col, ecol, ylab, model, sig_dex, mlab = VS_MUV[quantity]
    d = df.dropna(subset=[col, "M_UV", "z_sys"])
    if len(d) == 0:
        print(f"  [skip] {os.path.basename(path)}: no sources with M_UV")
        return
    cmap = get_cmap()
    norm = Normalize(vmin=d["z_sys"].min(), vmax=d["z_sys"].max())
    masks, labels = zbin_masks(d["z_sys"].to_numpy(), edges)
    xlim = muv_limits(d)
    grid = np.linspace(min(xlim), max(xlim), 300)
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.8),
                             sharex=True, sharey=True)
    sc = None
    for ax, m, lab in zip(axes, masks, labels):
        sub = d[m]
        if len(sub):
            z_med = float(sub["z_sys"].median())
            med = model(grid, z_med)
            c = cmap(norm(z_med))
            ax.fill_between(grid, med * 10 ** -sig_dex, med * 10 ** sig_dex,
                            color=c, alpha=0.15, lw=0, zorder=0)
            ax.plot(grid, med, color=c, lw=1.1, zorder=2,
                    label=rf"{mlab}, $z={z_med:.2f}$")
            if quantity == "fwhm":
                draw_pl25_fwhm(ax)
            sc = draw_points(ax, sub, "M_UV", col, "M_UV_err", ecol, norm, cmap)
            ax.legend(frameon=False, loc="upper left", fontsize=5.5)
        ax.set_title(f"{lab}  (N = {len(sub)})", fontsize=9)
        ax.set_xlabel(LABEL_MUV)
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*y_limits(d, col, ecol))
    axes[0].set_ylabel(ylab)
    fig.subplots_adjust(wspace=0.08)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.03)
        cb.set_label(LABEL_Z)
    fig.suptitle(title, fontsize=9, y=1.02)
    save(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Science plots for one manually labelled Lya group.")
    p.add_argument("--label", default="a",
                   help="Manual group label to plot (a, b, c, d), or 'all'. Default a.")
    p.add_argument("--group-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs",
                   help="Directory holding lya_group_<label>.csv.")
    p.add_argument("--delta-v-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/delta_v_from_best_zsys_line.csv")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv")
    p.add_argument("--muv-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/muv_beta_by_JELS_ID.csv")
    p.add_argument("--muv-max-err", type=float, default=0.5,
                   help="Only use M_UV with M_UV_err below this. Default 0.5.")
    p.add_argument("--outdir", default="/ceph/cephfs/apatrick/P2/plots")
    p.add_argument("--z-edges", type=float, nargs=2, default=[3.5, 4.5],
                   help="Edges of the three redshift bins. Default 3.5 4.5.")
    p.add_argument("--z-split", type=float, default=3.5,
                   help="Edge of the two redshift bins (_zbins2 plots). Default 3.5.")
    p.add_argument("--muv-edges2", type=float, nargs=1, default=None,
                   help="Edge of the two M_UV bins. Default: sample median.")
    p.add_argument("--muv-edges3", type=float, nargs=2, default=None,
                   help="Edges of the three M_UV bins. Default: sample tertiles.")
    p.add_argument("--fwhm-bin", type=float, default=50.0,
                   help="FWHM histogram bin width, km/s. Default 50.")
    p.add_argument("--dv-bin", type=float, default=100.0,
                   help="Velocity offset histogram bin width, km/s. Default 100.")
    p.add_argument("--muv-bin", type=float, default=0.25,
                   help="M_UV histogram bin width, mag. Default 0.25.")
    p.add_argument("--muv", type=float, default=None,
                   help="Fix the M_UV used for the Mason FWHM curves. Default: "
                        "the median M_UV of the sources in each plot or bin.")
    p.add_argument("--fwhm-model-max", type=float, default=800.0,
                   help="Upper FWHM limit for the model-comparison plots, km/s.")
    p.add_argument("--fwhm-log-bin", type=float, default=0.1,
                   help="Bin width in dex for the model-comparison "
                        "histograms. Default 0.1.")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="Bootstrap draws for the log-normal fit band.")
    p.add_argument("--density-scaling", action="store_true",
                   help="Also draw Mason+19 scaled by [(1+z)/3]^(2/3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-bin", type=float, default=0.2,
                   help="Redshift histogram bin width. Default 0.2.")
    return p.parse_args()


def run_group(label, args):
    group_csv = os.path.abspath(os.path.join(args.group_dir, f"lya_group_{label}.csv"))
    delta_v_path = os.path.abspath(args.delta_v_csv)
    properties_path = os.path.abspath(args.properties_csv)
    muv_path = os.path.abspath(args.muv_csv)
    outdir = os.path.abspath(args.outdir)
    edges3 = list(args.z_edges)
    edges2 = [args.z_split]

    print("[CONFIG]")
    print(f"  group           {label}")
    print(f"  group csv       {group_csv}")
    print(f"  delta_v csv     {delta_v_path}")
    print(f"  properties csv  {properties_path}")
    print(f"  M_UV csv        {muv_path}  (M_UV_err < {args.muv_max_err:g})")
    print(f"  output dir      {outdir}")
    print(f"  z bins          three: {edges3[0]:g}, {edges3[1]:g}   two: {edges2[0]:g}")
    print("")

    if not os.path.exists(group_csv):
        print(f"[SKIP] {group_csv} not found. Run group_lya_by_manual.py first.\n")
        return

    df = load_group(group_csv, delta_v_path, properties_path, muv_path,
                    args.muv_max_err)
    n_no_z = int(df["z_sys"].isna().sum())
    n_muv = int(df["M_UV"].notna().sum())
    print(f"[INFO] {len(df)} sources in group {label}, {n_muv} with a usable M_UV")
    if n_no_z:
        print(f"[WARN] {n_no_z} sources have no z_sys and are left out of "
              f"the redshift-split and coloured plots")

    z = df["z_sys"].to_numpy()
    fwhm = df["fwhm_kms"].to_numpy()
    dv = df["delta_v_kms"].to_numpy()
    muv_arr = df["M_UV"].to_numpy()

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"group_{label}_")
    title = f"Group {label}"
    zsets = [(edges3, "zbins"), (edges2, "zbins2")]

    print("")
    print("[PLOTS]")
    plot_hist(fwhm, fixed_bins(fwhm, args.fwhm_bin), LABEL_FWHM, title, stem + "hist_fwhm.png")
    plot_hist(z, fixed_bins(z, args.z_bin), LABEL_Z, title, stem + "hist_z.png")
    plot_hist(dv, fixed_bins(dv, args.dv_bin), LABEL_DV, title, stem + "hist_dv.png")
    if n_muv:
        plot_hist(muv_arr, fixed_bins(muv_arr, args.muv_bin), LABEL_MUV, title,
                  stem + "hist_muv.png")

    for edges, tag in zsets:
        plot_hist_zbins(fwhm, z, fixed_bins(fwhm, args.fwhm_bin), edges, LABEL_FWHM,
                        title, stem + f"hist_fwhm_{tag}.png")
        plot_hist_zbins(dv, z, fixed_bins(dv, args.dv_bin), edges, LABEL_DV,
                        title, stem + f"hist_dv_{tag}.png")

    plot_dv_vs_fwhm(df, stem + "dv_vs_fwhm.png", title)
    for edges, tag in zsets:
        plot_dv_vs_fwhm_zbins(df, edges, stem + f"dv_vs_fwhm_{tag}.png", title)

    # Literature comparison, FWHM histograms
    rng = np.random.default_rng(args.seed)
    xmax = args.fwhm_model_max
    lv = np.log10(fwhm[np.isfinite(fwhm) & (fwhm > 0)])
    lb = args.fwhm_log_bin
    log_edges = np.arange(np.floor(lv.min() / lb) * lb,
                          max(np.log10(xmax), lv.max()) + lb, lb)

    dz = df.dropna(subset=["z_sys"])
    muv_all = args.muv if args.muv is not None else median_or(dz["M_UV"], -20.0)
    if args.muv is None and not np.isfinite(dz["M_UV"]).any():
        print("[WARN] no usable M_UV, Mason curves fall back to M_UV = -20")
    muv_note = (rf"$M_{{\rm UV}}={args.muv:g}$ fixed" if args.muv is not None
                else r"median $M_{\rm UV}$ per bin")

    z_all, info_all = plot_fwhm_models(
        dz, muv_all, log_edges, xmax, args.n_boot, rng, args.density_scaling,
        stem + "fwhm_models.png", title)
    rows = [("all", z_all, muv_all, info_all)]

    for edges, tag in zsets:
        masks, labels = zbin_masks(dz["z_sys"].to_numpy(), edges)
        out = plot_fwhm_models_bins(
            dz, masks, labels, args.muv, muv_all, log_edges, xmax, args.n_boot,
            rng, args.density_scaling, stem + f"fwhm_models_{tag}.png",
            f"{title}  (Mason at each bin's median $z$, {muv_note})")
        rows += [(f"[{tag}] {plain(l)}", zm, mu, inf) for l, zm, mu, inf in out]

    m_ok = dz["M_UV"].dropna()
    if len(m_ok) >= 4:
        e2 = list(args.muv_edges2) if args.muv_edges2 else [float(m_ok.median())]
        e3 = (list(args.muv_edges3) if args.muv_edges3
              else [float(q) for q in m_ok.quantile([1 / 3, 2 / 3])])
        for edges, tag in ((e2, "muvbins2"), (e3, "muvbins3")):
            masks, labels = bin_masks(dz["M_UV"].to_numpy(), edges, "muv")
            out = plot_fwhm_models_bins(
                dz, masks, labels, args.muv, muv_all, log_edges, xmax,
                args.n_boot, rng, args.density_scaling,
                stem + f"fwhm_models_{tag}.png",
                f"{title}  (Mason at each bin's median $z$ and {muv_note})")
            rows += [(f"[{tag}] {plain(l)}", zm, mu, inf) for l, zm, mu, inf in out]
    else:
        print(f"  [skip] M_UV-binned FWHM plots: only {len(m_ok)} sources with M_UV")

    plot_dv_vs_fwhm_verhamme(df, stem + "dv_vs_fwhm_verhamme.png", title)
    for edges, tag in zsets:
        plot_dv_vs_fwhm_verhamme_zbins(df, edges,
                                       stem + f"dv_vs_fwhm_verhamme_{tag}.png", title)

    # Delta_v and FWHM against M_UV
    for q in ("dv", "fwhm"):
        plot_vs_muv(df, q, stem + f"{q}_vs_muv.png", title)
        for edges, tag in zsets:
            plot_vs_muv_zbins(df, q, edges, stem + f"{q}_vs_muv_{tag}.png", title)

    print("")
    print("[MODELS] Mason+18 and Mason+19 evaluated at the M_UV and z listed")
    print(f"         Prieto-Lyon+25 median FWHM = {10**PL25_MU:.0f} km/s (all z)")
    for lab, zm, mu, info in rows:
        if info.get("n", 0) < 3:
            print(f"  {lab:<34s} N={info.get('n', 0)}  too few for a fit")
            continue
        line = (f"  {lab:<34s} N={info['n']:<3d} z_med={zm:.2f} M_UV={mu:.2f}  "
                f"this work={info['this_median']:.0f}  "
                f"Mason+18={info['mason18_median']:.0f}  "
                f"Mason+19={info['mason19_median']:.0f}")
        if "mason_density_median" in info:
            line += f"  density-scaled={info['mason_density_median']:.0f}"
        print(line + "  km/s")
    print("")


def main():
    args = parse_args()
    label = args.label.strip().lower()
    labels = LABELS if label == "all" else [label]
    for lab in labels:
        run_group(lab, args)
    print("[DONE]")


if __name__ == "__main__":
    main()