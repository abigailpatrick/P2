#!/usr/bin/env python
"""
checks/check_ha_corrections.py

Consistency checks of the slit-loss and dust corrections made by
build_ha_flux_corrections.py, and of the escape fractions from
measure_lya_fesc.py. Nothing here changes a value. It makes figures, a
per-band slit factor table and a list of sources worth a look.

Sample: in_analysis == 1 in lya_catalog.csv (--sample all for every source in
ha_flux_corrections.csv, --sample detections for the Lya detections only).
Lya detections are filled symbols, non-detections open.

1. Slit factors in every band (slit_factors.png)
------------------------------------------------
build_ha_flux_corrections.py keeps one factor per line. Here the factor
    S = f_nu(Isaac total photometry) / f_nu(spectrum through the band)
is recomputed with the same functions in every NIRCam band a spectrum covers
(coverage >= --min-coverage, S/N >= 3 in both), whether or not it contains a
line. A band counts as nominal if its half-maximum range lies inside the
grating's nominal range. Bands beyond it are shown as grey crosses and left
out of every number below.
  a  S against band pivot, one line per spectrum. A flat line means the
     correction does not depend on wavelength. A rise means the slit loses
     more light at longer wavelengths (the PSF grows).
  b  S in G395 against G235 for sources with both, each band's S moved to
     3 micron with the sample median slope and then the median taken. They
     should agree if both spectra sample the galaxy the same way.
  c  max / min S across the nominal bands of one spectrum.
  d  S used for Halpha against Isaac's total_correction (aperture to total,
     larger for extended sources).
  e  S used for Halpha against z_sys, sample-median factors as crosses.
  f  S(Hbeta) against S(Halpha) for the case 1 same-spectrum decrements.
The median slope d ln S / d ln lambda of the spectra with >= 2 nominal bands
is printed, with the bias it would put on a same-spectrum decrement if it
were left uncorrected:
    R_true / R_obs = S_Ha / S_Hb = (6565 / 4863)^slope
    Delta A_Ha     = 2.5 k_Ha / (k_Hb - k_Ha) log10(R_true / R_obs)

2. Balmer decrement and SED attenuation (balmer_dust.png)
---------------------------------------------------------
  a  A_Ha Balmer against A_Ha SED, colour Hbeta S/N, circle same spectrum,
     square different spectra.
  b  Halpha / Hbeta against z_sys, with the Case B 2.86.
  c  slit-corrected / observed decrement (= S_Ha / S_Hb) against z_sys, for
     the same-spectrum decrements.
  d  A_Ha Balmer - SED against Hbeta S/N.
  e  A_Ha Balmer - SED against SED stellar mass.
  f  A_Ha used, by case.

3. Photometric against spectral Halpha (ha_phot_vs_spec.png)
------------------------------------------------------------
Where both exist, the photometric Halpha (band excess) and the slit-corrected
spectral Halpha are independent totals.
  a  photometric against spectral total Halpha, colour by band.
  b  their ratio against the slit factor used. A trend means the slit
     correction is over- or under-correcting.
  c  their ratio against z_sys.
  d  their ratio against the photometric S/N.

4. Escape fractions (fesc_checks.png), if lya_catalog.csv has fesc_lya
-----------------------------------------------------------------------
  a  f_esc (dust method used) against f_esc with the SED A_Ha, case 1 only
     (cases 2 and 3 are the same by construction).
  b  f_esc against the Halpha slit factor.
  c  f_esc against A_Ha used.
  d  f_esc by case, upper limits as down arrows.

Flags (ha_correction_flags.csv)
-------------------------------
One row per source with any of
  slit_out     direct S_Ha outside --slit-warn (0.8 to 3)
  slit_median  S_Ha is the sample median, with the reason
  grating      S at 3 micron of G235 and G395 differ by more than --ratio-warn (1.5)
  chromatic    max / min S across the nominal bands of a spectrum > --ratio-warn
  balmer_low   decrement below 2.86 by more than 2 sigma
  dust_offset  |A_Ha Balmer - SED| > 0.5 mag and > 3 sigma
  phot_spec    photometric / spectral Halpha outside --phot-warn (0.5 to 2),
               photometric S/N >= 3
  fesc_gt1     f_esc above 1
  fesc_dust    f_esc Balmer / f_esc SED outside 1/3 to 3

Inputs
------
  jwst_catalogs/ha_flux_corrections.csv          build_ha_flux_corrections.py
  MUSE_catalogs/lya_catalog.csv                  in_analysis, lya_det, sed_logM, fesc_*
  jwst_catalogs/cosmos_primer_minerva_production.fits
  jwst_spectra_pm/<grating>_<filter>/<ID>_<grating>_<filter>_spectra_lime.fits
  jwst_catalogs/nircam_filters/JWST_NIRCam.<FILT>.dat

Outputs
-------
  plots/ha_checks/slit_summary.png        S used against z by grating, and photometric /
                                          slit-corrected spectral Halpha in dex (the one-figure check)
  plots/ha_checks/dust_balmer_vs_sed.png  A_Ha and f_esc, Balmer against SED, case 1, with offsets
  plots/ha_checks/slit_factors.png, balmer_dust.png, ha_phot_vs_spec.png, fesc_checks.png
  jwst_catalogs/ha_slit_factors_all_bands.csv
  jwst_catalogs/ha_correction_flags.csv

Usage
-----
python checks/check_ha_corrections.py
python checks/check_ha_corrections.py --sample all
python checks/check_ha_corrections.py --reuse-bands     # skip re-reading the spectra
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import p2_common as pc                          # noqa: E402
import build_ha_flux_corrections as bh          # noqa: E402
from fit_muv_beta import FILTERS, load_isaac    # noqa: E402

CAT_DIR = f"{pc.P2_ROOT}/jwst_catalogs"
REF_UM = 3.0       # common wavelength for comparing gratings
# slit_source values that are a source's own factor (not the sample median);
# "direct" is the label of older ha_flux_corrections.csv files
OWN_S = ("line_band", "interp", "spec_median", "median", "direct")
FALLBACK_S = ("other_grating", "grating_median", "sample_median")
GR_COL = {"G235": "#2a9d8f", "G395": "#e07a1f"}
CASE_COL = {1: "#2a9d8f", 2: "#3b6fb6", 3: "#e39b3b", 0: "0.6"}
BAND_COL = dict(zip(bh.SLIT_FILTERS, plt.cm.viridis(np.linspace(0, 0.95, len(bh.SLIT_FILTERS)))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ha-csv", default=f"{CAT_DIR}/ha_flux_corrections.csv")
    p.add_argument("--catalog", default=pc.LYA_CATALOG)
    p.add_argument("--isaac", default=bh.ISAAC_FITS)
    p.add_argument("--spec-root", default=bh.SPEC_ROOT)
    p.add_argument("--filter-dir", default=bh.FILTER_DIR)
    p.add_argument("--sample", choices=["analysis", "detections", "all"], default="analysis")
    p.add_argument("--min-coverage", type=float, default=0.70)
    p.add_argument("--slit-snr-min", type=float, default=3.0)
    p.add_argument("--slit-warn", type=float, nargs=2, default=[0.8, 3.0])
    p.add_argument("--ratio-warn", type=float, default=1.5)
    p.add_argument("--phot-warn", type=float, nargs=2, default=[0.5, 2.0])
    p.add_argument("--reuse-bands", action="store_true",
                   help="read --bands-csv instead of recomputing it from the spectra")
    p.add_argument("--bands-csv", default=f"{CAT_DIR}/ha_slit_factors_all_bands.csv")
    p.add_argument("--slit-bands-csv", default=f"{CAT_DIR}/ha_slit_bands.csv",
                   help="the bands build_ha_flux_corrections.py used, for the band-to-band scatter")
    p.add_argument("--flags-csv", default=f"{CAT_DIR}/ha_correction_flags.csv")
    p.add_argument("--outdir", default=f"{pc.P2_ROOT}/plots/ha_checks")
    return p.parse_args()


# ============================================================================
# Inputs
# ============================================================================

def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def load_inputs(a):
    ha = pd.read_csv(a.ha_csv)
    ha["ID"] = ha["ID"].astype(int)
    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    keep = ["ID", "in_analysis", "lya_det", "sed_logM", "fesc_lya", "fesc_lya_err_lo",
            "fesc_lya_err_hi", "fesc_lya_is_limit", "fesc_status", "fesc_lya_sed"]
    cat = cat[[c for c in keep if c in cat.columns]]
    d = ha.merge(cat, on="ID", how="left")
    d["in_analysis"] = num(d, "in_analysis").fillna(0).astype(int)
    d["det"] = (num(d, "lya_det") == 1)
    if a.sample == "analysis":
        d = d[d["in_analysis"] == 1]
    elif a.sample == "detections":
        d = d[(d["in_analysis"] == 1) & d["det"]]
    return d.reset_index(drop=True)


def band_nominal(curve, grating):
    """1 if the band's half-maximum range is inside the grating's nominal range."""
    lam, t = curve
    on = lam[t >= 0.5 * t.max()]
    lo, hi = pc.NIRSPEC_NOMINAL_UM[grating]
    return int(on.min() / 1e4 >= lo and on.max() / 1e4 <= hi)


def all_band_factors(d, a):
    """S in every usable band of every spectrum of the sample."""
    curves = bh.load_filters(a.filter_dir)
    isaac, _ = load_isaac(a.isaac, d["ID"].tolist(), "total")
    rows = []
    for _, s in d.iterrows():
        sid = int(s["ID"])
        if sid not in isaac.index:
            continue
        prow = isaac.loc[sid]
        fl = int(prow["Flag"]) if np.isfinite(prow["Flag"]) else -1
        for gr in pc.GRATINGS:
            if not os.path.exists(bh.spectrum_path(a.spec_root, sid, gr)):
                continue
            try:
                w, f, e = bh.load_spectrum(a.spec_root, sid, gr)
            except (OSError, KeyError):
                continue
            for filt in bh.SLIT_FILTERS:
                fp, ep = prow.get(filt.lower(), np.nan), prow.get(f"{filt.lower()}_err", np.nan)
                if not (np.isfinite(fp) and np.isfinite(ep) and ep > 0):
                    continue
                lam_f, t_f = curves[filt]
                with np.errstate(invalid="ignore", divide="ignore"):
                    fs, es, cov = bh.synthetic_fnu(w, f, e, lam_f, t_f)
                if not (np.isfinite(fs) and np.isfinite(es)):
                    continue
                if cov < a.min_coverage or not (es > 0 and fs / es >= a.slit_snr_min
                                                and fp / ep >= a.slit_snr_min):
                    continue
                fac = fp / fs
                rows.append(dict(ID=sid, z_sys=s["z_sys"], grating=gr, band=filt,
                                 pivot_um=FILTERS[filt][1] / 1e4, S=fac,
                                 S_err=fac * np.hypot(ep / fp, es / fs), coverage=cov,
                                 nominal=band_nominal(curves[filt], gr), isaac_flag=fl,
                                 total_correction=float(prow["total_correction"])))
    return pd.DataFrame(rows, columns=["ID", "z_sys", "grating", "band", "pivot_um", "S", "S_err",
                                       "coverage", "nominal", "isaac_flag", "total_correction"])


def per_spectrum(b):
    """Median S, max/min and log slope for every spectrum, nominal bands only."""
    out = []
    for (sid, gr), g in b[b["nominal"] == 1].groupby(["ID", "grating"]):
        S, lam = g["S"].to_numpy(float), g["pivot_um"].to_numpy(float)
        slope = np.nan
        if len(g) >= 2 and np.ptp(lam) > 0:
            wgt = (g["S"] / g["S_err"]).to_numpy(float)
            slope = np.polyfit(np.log(lam), np.log(S), 1, w=wgt)[0]
        out.append(dict(ID=sid, grating=gr, n_bands=len(g), S_med=float(np.median(S)),
                        S_spread=float(S.max() / S.min()), slope=slope))
    sp = pd.DataFrame(out, columns=["ID", "grating", "n_bands", "S_med", "S_spread", "slope", "S_3um"])
    # S moved to 3 micron with the sample median slope, so spectra in
    # different gratings are compared at one wavelength
    med = float(np.nanmedian(sp["slope"])) if sp["slope"].notna().any() else 0.0
    nb = b[b["nominal"] == 1]
    s3 = (nb["S"] * (REF_UM / nb["pivot_um"]) ** med).groupby([nb["ID"], nb["grating"]]).median()
    sp["S_3um"] = [s3.get((i, g), np.nan) for i, g in zip(sp["ID"], sp["grating"])]
    sp.attrs["slope_med"] = med
    return sp


# ============================================================================
# Plot helpers
# ============================================================================

def style():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "font.size": 8})


def gfam(g):
    return str(g)[:4] if isinstance(g, str) else ""


def pts(ax, x, y, det, col, xerr=None, yerr=None, mk="o", ms=4, label=None):
    """Filled for Lya detections, open for non-detections."""
    x, y, det = np.asarray(x, float), np.asarray(y, float), np.asarray(det, bool)
    for m, fill in ((det, True), (~det, False)):
        if not m.any():
            continue
        xe = None if xerr is None else np.asarray(xerr)[..., m]
        ye = None if yerr is None else np.asarray(yerr)[..., m]
        ax.errorbar(x[m], y[m], xerr=xe, yerr=ye, fmt=mk, ms=ms, mfc=col if fill else "white",
                    mec=col if not fill else "k", mew=0.6, ecolor="0.7", elinewidth=0.6,
                    label=label if fill else None)


def e2(d, col):
    lo, hi = num(d, f"{col}_err_lo"), num(d, f"{col}_err_hi")
    return np.nan_to_num(np.vstack([lo.to_numpy(float), hi.to_numpy(float)]))


def rho_text(ax, x, y, loc=(0.03, 0.95)):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 5:
        r, p = spearmanr(x[ok], y[ok])
        ax.text(*loc, rf"$\rho_s$ = {r:+.2f}, p = {p:.2g}, N = {ok.sum()}",
                transform=ax.transAxes, va="top", fontsize=7)


def det_legend(ax):
    ax.plot([], [], "o", mfc="0.4", mec="k", ms=4, label=r"Ly$\alpha$ det")
    ax.plot([], [], "o", mfc="white", mec="0.4", ms=4, label="non-det")


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# ============================================================================
# Figures
# ============================================================================

def fig_slit(d, b, sp, path):
    det_of = dict(zip(d["ID"], d["det"]))
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.4))
    ax = ax.ravel()

    # a: S against wavelength, one line per spectrum
    a_ = ax[0]
    for (sid, gr), g in b.groupby(["ID", "grating"]):
        g = g.sort_values("pivot_um")
        col = GR_COL.get(gfam(gr), "k")
        nom = g[g["nominal"] == 1]
        a_.plot(nom["pivot_um"], nom["S"], "-", color=col, lw=0.6, alpha=0.6)
        pts(a_, nom["pivot_um"], nom["S"], [det_of.get(sid, False)] * len(nom), col, ms=3)
        ext = g[g["nominal"] == 0]
        a_.plot(ext["pivot_um"], ext["S"], "x", color="0.6", ms=4)
    for gf, col in GR_COL.items():
        a_.plot([], [], "-", color=col, label=gf)
    a_.plot([], [], "x", color="0.6", label="beyond nominal range")
    a_.axhline(1, color="k", ls=":", lw=0.6)
    logy(a_)
    a_.set_xlabel(r"band pivot [$\mu$m]")
    a_.set_ylabel(r"$S$ = phot / spectrum")
    a_.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=6)
    a_.set_title("(a) every band of every spectrum", fontsize=8)

    # b: G395 against G235
    a_ = ax[1]
    m = sp.assign(fam=sp["grating"].map(gfam)).groupby(["ID", "fam"])["S_3um"].median().unstack()
    if {"G235", "G395"} <= set(m.columns):
        both = m.dropna(subset=["G235", "G395"])
        pts(a_, both["G235"], both["G395"], [det_of.get(i, False) for i in both.index], "#8c4a8c")
        lim = [0.3, 10]
        a_.plot(lim, lim, "k--", lw=0.8)
        a_.fill_between(lim, np.array(lim) / 1.5, np.array(lim) * 1.5, color="0.9", zorder=0)
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        logx(a_)
        logy(a_)
        a_.set_title(f"(b) sources with both, N = {len(both)}", fontsize=8)
    else:
        a_.text(0.5, 0.5, "no source with both G235 and G395", ha="center", transform=a_.transAxes)
        a_.set_title("(b)", fontsize=8)
    a_.set_xlabel(rf"$S$ at {REF_UM:g} $\mu$m, G235")
    a_.set_ylabel(rf"$S$ at {REF_UM:g} $\mu$m, G395")

    # c: spread within a spectrum
    a_ = ax[2]
    v = sp.loc[sp["n_bands"] >= 2, "S_spread"].to_numpy(float)
    if v.size:
        a_.hist(v, bins=np.linspace(1, max(2.5, v.max() * 1.05), 20), color="#8c4a8c",
                edgecolor="k", lw=0.5)
    a_.set_xlabel(r"max / min $S$ within one spectrum")
    a_.set_ylabel("N spectra")
    a_.set_title(f"(c) spectra with >= 2 nominal bands, N = {v.size}", fontsize=8)

    # d, e: S used for Halpha
    u = d[d["ha_slit_source"].isin(OWN_S)]
    tc = b.drop_duplicates("ID").set_index("ID")["total_correction"]
    a_ = ax[3]
    for gf, col in GR_COL.items():
        g = u[u["ha_grating"].map(gfam) == gf]
        pts(a_, tc.reindex(g["ID"]).to_numpy(float), g["ha_slit_factor"], g["det"], col,
            yerr=num(g, "ha_slit_factor_err").to_numpy(float), label=gf)
    rho_text(a_, tc.reindex(u["ID"]).to_numpy(float), u["ha_slit_factor"])
    a_.set_xlabel("Isaac total_correction")
    a_.set_ylabel(r"$S_{\rm H\alpha}$ used")
    a_.set_title(r"(d) own H$\alpha$ factors", fontsize=8)
    a_.legend(frameon=False, fontsize=6, loc="lower right")

    a_ = ax[4]
    for gf, col in GR_COL.items():
        g = u[u["ha_grating"].map(gfam) == gf]
        pts(a_, g["z_sys"], g["ha_slit_factor"], g["det"], col,
            yerr=num(g, "ha_slit_factor_err").to_numpy(float), label=gf)
    med = d[d["ha_slit_source"].isin(FALLBACK_S)]
    a_.plot(med["z_sys"], med["ha_slit_factor"], "x", color="0.4", ms=5,
            label=f"fallback, no own factor ({len(med)})")
    a_.set_xlabel(r"$z_{\rm sys}$")
    a_.set_ylabel(r"$S_{\rm H\alpha}$ used")
    a_.legend(frameon=False, fontsize=6)
    a_.set_title("(e)", fontsize=8)

    # f: Hbeta against Halpha factor in one spectrum
    a_ = ax[5]
    s = d[(d["balmer_spectra"] == "same") & d["ha_slit_source"].isin(OWN_S)
          & d.get("hb_slit_source", pd.Series("", index=d.index)).isin(OWN_S)]
    if len(s):
        pts(a_, s["ha_slit_factor"], s["hb_slit_factor"], s["det"], "#2a9d8f",
            xerr=num(s, "ha_slit_factor_err").to_numpy(float),
            yerr=num(s, "hb_slit_factor_err").to_numpy(float))
        lim = [0.3, 10]
        a_.plot(lim, lim, "k--", lw=0.8)
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        logx(a_)
        logy(a_)
    a_.set_xlabel(r"$S_{\rm H\alpha}$")
    a_.set_ylabel(r"$S_{\rm H\beta}$")
    a_.set_title(f"(f) case 1 same spectrum, N = {len(s)}", fontsize=8)
    det_legend(ax[2])
    ax[2].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_balmer(d, path):
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.4))
    ax = ax.ravel()
    c1 = d[d["a_ha_balmer"].notna()].copy()
    c1["dA"] = num(c1, "a_ha_balmer") - num(c1, "a_ha_sed")
    hb_snr = num(c1, "hb_snr").to_numpy(float)
    norm = matplotlib.colors.LogNorm(vmin=3, vmax=max(30, np.nanmax(hb_snr) if hb_snr.size else 30))

    a_ = ax[0]
    sc = None
    for bs, mk in (("same", "o"), ("different", "s")):
        g = c1[(c1["balmer_spectra"] == bs) & c1["a_ha_sed"].notna()]
        if not len(g):
            continue
        a_.errorbar(g["a_ha_sed"], g["a_ha_balmer"], xerr=e2(g, "a_ha_sed"), yerr=e2(g, "a_ha_balmer"),
                    fmt="none", ecolor="0.75", elinewidth=0.6, zorder=1)
        for m, ec in ((g["det"], "k"), (~g["det"], "0.6")):
            gg = g[m]
            sc = a_.scatter(gg["a_ha_sed"], gg["a_ha_balmer"], c=num(gg, "hb_snr"), cmap="plasma",
                            norm=norm, marker=mk, s=22, edgecolor=ec, lw=0.6, zorder=2,
                            alpha=1 if ec == "k" else 0.55)
        a_.scatter([], [], marker=mk, facecolor="0.5", edgecolor="k", label=f"{bs} ({len(g)})")
    lim = max(1.5, np.nanmax(np.r_[1.5, c1["a_ha_balmer"], c1["a_ha_sed"]]) * 1.1) if len(c1) else 1.5
    a_.plot([0, lim], [0, lim], "k--", lw=0.8)
    a_.set_xlim(-0.05, lim)
    a_.set_ylim(-0.05, lim)
    a_.set_xlabel(r"$A_{\rm H\alpha}$ SED [mag]")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ Balmer [mag]")
    a_.legend(frameon=False, fontsize=6, loc="upper left")
    if sc is not None:
        cb = fig.colorbar(sc, ax=a_, label=r"H$\beta$ S/N", pad=0.01)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_title(r"(a) case 1, grey edge = non-det", fontsize=8)

    a_ = ax[1]
    for bs, mk in (("same", "o"), ("different", "s")):
        for gf, col in GR_COL.items():
            g = c1[(c1["balmer_spectra"] == bs) & (c1["balmer_ha_grating"].map(gfam) == gf)]
            pts(a_, g["z_sys"], g["balmer_ratio"], g["det"], col, yerr=e2(g, "balmer_ratio"), mk=mk,
                label=f"{bs}, H$\\alpha$ in {gf}" if len(g) else None)
    a_.axhline(bh.R_INT, color="k", ls="--", lw=0.8)
    a_.set_xlabel(r"$z_{\rm sys}$")
    a_.set_ylabel(r"H$\alpha$ / H$\beta$ used")
    a_.legend(frameon=False, fontsize=6)
    a_.set_title("(b) decrement, dashed 2.86", fontsize=8)

    a_ = ax[2]
    s = c1[(c1["balmer_spectra"] == "same") & c1["balmer_ratio_obs"].notna()
           & c1["balmer_ratio_slitcorr"].notna()]
    for gf, col in GR_COL.items():
        g = s[s["balmer_ha_grating"].map(gfam) == gf]
        pts(a_, g["z_sys"], num(g, "balmer_ratio_slitcorr") / num(g, "balmer_ratio_obs"), g["det"],
            col, label=gf if len(g) else None)
    a_.axhline(1, color="k", ls="--", lw=0.8)
    a_.set_xlabel(r"$z_{\rm sys}$")
    a_.set_ylabel(r"$R_{\rm slit\,corr} / R_{\rm obs}$ = $S_{\rm H\alpha}/S_{\rm H\beta}$")
    a_.legend(frameon=False, fontsize=6)
    a_.set_title(f"(c) same spectrum, N = {len(s)}", fontsize=8)

    g = c1[c1["dA"].notna()]
    sig = np.hypot(0.5 * (num(g, "a_ha_balmer_err_lo") + num(g, "a_ha_balmer_err_hi")),
                   0.5 * (num(g, "a_ha_sed_err_lo") + num(g, "a_ha_sed_err_hi")))
    a_ = ax[3]
    pts(a_, num(g, "hb_snr"), g["dA"], g["det"], "#2a9d8f", yerr=sig.to_numpy(float))
    a_.axhline(0, color="k", ls="--", lw=0.8)
    rho_text(a_, num(g, "hb_snr"), g["dA"])
    a_.set_xlabel(r"H$\beta$ S/N")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ Balmer $-$ SED [mag]")
    a_.set_title("(d)", fontsize=8)

    a_ = ax[4]
    pts(a_, num(g, "sed_logM"), g["dA"], g["det"], "#2a9d8f", yerr=sig.to_numpy(float))
    a_.axhline(0, color="k", ls="--", lw=0.8)
    rho_text(a_, num(g, "sed_logM"), g["dA"])
    a_.set_xlabel(r"SED log $M_\star$ [M$_\odot$]")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ Balmer $-$ SED [mag]")
    a_.set_title("(e)", fontsize=8)

    a_ = ax[5]
    v = num(d, "a_ha")
    bins = np.linspace(0, max(1.5, np.nanmax(np.r_[1.5, v]) * 1.05), 16)
    a_.hist([v[d["ha_case"] == c].dropna() for c in (1, 2, 3)], bins=bins, stacked=True,
            color=[CASE_COL[c] for c in (1, 2, 3)], edgecolor="k", lw=0.4,
            label=[f"case {c} ({int(((d['ha_case'] == c) & v.notna()).sum())})" for c in (1, 2, 3)])
    a_.set_xlabel(r"$A_{\rm H\alpha}$ used [mag]")
    a_.set_ylabel("N")
    a_.legend(frameon=False, fontsize=6)
    a_.set_title("(f)", fontsize=8)
    det_legend(ax[3])
    ax[3].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def phot_spec(d):
    g = d[(d["ha_source"] == "spec") & (num(d, "ha_phot_snr") >= 3) & num(d, "ha_flux_total").notna()].copy()
    g["ratio"] = num(g, "ha_phot_flux") / num(g, "ha_flux_total")
    tot_err = 0.5 * (num(g, "ha_flux_total_err_lo") + num(g, "ha_flux_total_err_hi"))
    g["ratio_err"] = np.abs(g["ratio"]) * np.hypot(num(g, "ha_phot_flux_err") / num(g, "ha_phot_flux"),
                                                   tot_err / num(g, "ha_flux_total"))
    return g


def fig_phot(d, path):
    g = phot_spec(d)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax = ax.ravel()
    a_ = ax[0]
    for band, gg in g.groupby("ha_phot_band"):
        pts(a_, gg["ha_flux_total"], gg["ha_phot_flux"], gg["det"], BAND_COL.get(band, "k"),
            xerr=e2(gg, "ha_flux_total"), yerr=num(gg, "ha_phot_flux_err").to_numpy(float),
            label=f"{band} ({len(gg)})")
    if len(g):
        v = np.r_[g["ha_flux_total"], g["ha_phot_flux"]]
        v = v[v > 0]
        lim = [v.min() / 2, v.max() * 2]
        a_.plot(lim, lim, "k--", lw=0.8)
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_xlabel(r"H$\alpha$ spectral $\times S$ [erg s$^{-1}$ cm$^{-2}$]")
    a_.set_ylabel(r"H$\alpha$ photometric")
    a_.legend(frameon=False, fontsize=6, title="band", title_fontsize=6)
    a_.set_title(f"(a) N = {len(g)}", fontsize=8)

    for a_, x, lab, t in ((ax[1], num(g, "ha_slit_factor"), r"$S_{\rm H\alpha}$ used", "(b)"),
                          (ax[2], g["z_sys"], r"$z_{\rm sys}$", "(c)"),
                          (ax[3], num(g, "ha_phot_snr"), r"photometric H$\alpha$ S/N", "(d)")):
        for band, gg in g.groupby("ha_phot_band"):
            pts(a_, x[gg.index], gg["ratio"], gg["det"], BAND_COL.get(band, "k"),
                yerr=gg["ratio_err"].to_numpy(float))
        a_.axhline(1, color="k", ls="--", lw=0.8)
        a_.axhspan(0.5, 2, color="0.93", zorder=0)
        rho_text(a_, x, g["ratio"])
        a_.set_xlabel(lab)
        a_.set_ylabel("photometric / spectral")
        a_.set_title(t, fontsize=8)
        if len(g):
            logy(a_)
            a_.set_ylim(min(0.3, np.nanmin(g["ratio"].clip(lower=0.05)) / 1.3),
                        max(3.5, np.nanmax(g["ratio"]) * 1.3))
    mm = g[g["ha_slit_source"].isin(FALLBACK_S)] if len(g) else g
    if len(mm):
        ax[1].plot(mm["ha_slit_factor"], mm["ratio"], "x", color="0.3", ms=6, label="fallback S")
        ax[1].legend(frameon=False, fontsize=6)
    det_legend(ax[2])
    ax[2].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_fesc(d, path):
    fe = num(d, "fesc_lya")
    lim_ = num(d, "fesc_lya_is_limit") == 1
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax = ax.ravel()

    a_ = ax[0]
    c1 = d[(d["ha_case"] == 1) & fe.notna() & num(d, "fesc_lya_sed").notna()]
    for m, mk in ((~lim_[c1.index], "o"), (lim_[c1.index], "v")):
        g = c1[m]
        pts(a_, num(g, "fesc_lya_sed"), num(g, "fesc_lya"), g["det"], CASE_COL[1], mk=mk,
            label="limit" if mk == "v" else "measured")
    if len(c1):
        v = np.r_[num(c1, "fesc_lya"), num(c1, "fesc_lya_sed")]
        v = v[v > 0]
        lim = [v.min() / 2, max(1.5, v.max() * 2)]
        a_.plot(lim, lim, "k--", lw=0.8)
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        a_.set_xscale("log")
        a_.set_yscale("log")
    a_.set_xlabel(r"$f_{\rm esc}$ with SED $A_{\rm H\alpha}$")
    a_.set_ylabel(r"$f_{\rm esc}$ with Balmer $A_{\rm H\alpha}$")
    a_.set_title(f"(a) case 1, N = {len(c1)}", fontsize=8)
    a_.plot([], [], "o", mfc=CASE_COL[1], mec="k", ms=4, label="measured")
    a_.plot([], [], "v", mfc="white", mec=CASE_COL[1], ms=4, label="5 sigma limit")
    h, l = a_.get_legend_handles_labels()
    a_.legend(h[-2:], l[-2:], frameon=False, fontsize=6)

    for a_, x, lab, t in ((ax[1], num(d, "ha_slit_factor"), r"$S_{\rm H\alpha}$ used", "(b)"),
                          (ax[2], num(d, "a_ha"), r"$A_{\rm H\alpha}$ used [mag]", "(c)")):
        for c in (1, 2, 3):
            m = (d["ha_case"] == c) & fe.notna()
            mm = m & ~lim_
            g = d[mm]
            pts(a_, x[mm], fe[mm], g["det"], CASE_COL[c], yerr=e2(g, "fesc_lya"),
                label=f"case {c}")
            ml = m & lim_
            a_.plot(x[ml], fe[ml], "v", mfc="white", mec=CASE_COL[c], ms=4)
        ok = fe.notna() & ~lim_
        rho_text(a_, x[ok], fe[ok])
        a_.axhline(1, color="k", ls=":", lw=0.6)
        a_.set_yscale("log")
        a_.set_xlabel(lab)
        a_.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
        a_.set_title(t + " triangles are 5 sigma limits", fontsize=8)
    ax[1].legend(frameon=False, fontsize=6, loc="lower right")

    a_ = ax[3]
    rng = np.random.default_rng(1)
    for c in (1, 2, 3):
        m = (d["ha_case"] == c) & fe.notna()
        xj = c + rng.uniform(-0.15, 0.15, m.sum())
        g = d[m]
        is_l = lim_[m].to_numpy(bool)
        pts(a_, xj[~is_l], fe[m][~is_l], g["det"][~is_l], CASE_COL[c])
        a_.plot(xj[is_l], fe[m][is_l], "v", mfc="white", mec=CASE_COL[c], ms=4)
    a_.axhline(1, color="k", ls=":", lw=0.6)
    a_.set_yscale("log")
    a_.set_xticks([1, 2, 3])
    a_.set_xticklabels(["1 spec+Balmer", "2 spec+SED", "3 phot+SED"])
    a_.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
    a_.set_title("(d) by case", fontsize=8)
    det_legend(ax[3])
    ax[3].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



# ============================================================================
# Two summary figures with headline numbers
# ============================================================================

GRATING_STYLE = {"G235M": ("#2a9d8f", "G235M"), "G235H": ("#1d6b61", "G235H"),
                 "G395M": ("#e07a1f", "G395M"), "G395H": ("#a8540f", "G395H")}
SRC_MARK = {"line_band": "o", "interp": "D", "spec_median": "s", "median": "s", "direct": "o",
            "other_grating": "^"}


def dex_stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(N=0, med=np.nan, p16=np.nan, p84=np.nan, nmad=np.nan)
    return dict(N=int(x.size), med=float(np.median(x)), p16=float(np.percentile(x, 16)),
                p84=float(np.percentile(x, 84)), nmad=bh.nmad(x))


def fig_slit_summary(d, path, slit_bands):
    """One figure for 'are the slit corrections consistent?': S used against z
    by grating, and the end-to-end test, photometric over slit-corrected
    spectral Halpha, in dex. Returns the numbers it prints."""
    sp = d[d["ha_source"] == "spec"].copy()
    own = sp[sp["ha_slit_source"].isin(OWN_S)]
    med_all = float(np.median(own["ha_slit_factor"])) if len(own) else np.nan
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
    a_ = ax[0]
    lines = []
    for g, (col, lab) in GRATING_STYLE.items():
        gg = sp[sp["ha_grating"] == g]
        if not len(gg):
            continue
        for src, mk in SRC_MARK.items():
            m = gg["ha_slit_source"] == src
            if m.any():
                pts(a_, gg.loc[m, "z_sys"], gg.loc[m, "ha_slit_factor"], gg.loc[m, "det"], col, mk=mk,
                    yerr=num(gg[m], "ha_slit_factor_err").to_numpy(float))
        sm = gg["ha_slit_source"].isin(("grating_median", "sample_median"))
        a_.plot(gg.loc[sm, "z_sys"], gg.loc[sm, "ha_slit_factor"], "x", color=col, ms=5)
        go = gg[gg["ha_slit_source"].isin(OWN_S)]["ha_slit_factor"].to_numpy(float)
        if go.size:
            lines.append(f"{lab}: {np.median(go):.2f}, NMAD {bh.nmad(go):.2f} "
                         f"({100 * bh.nmad(go) / np.median(go):.0f} per cent, N {go.size})")
            a_.plot([], [], "o", color=col, label=f"{lab}  median {np.median(go):.2f}, N = {go.size}")
    if np.isfinite(med_all):
        a_.axhline(med_all, color="k", ls="--", lw=0.8)
    a_.axhline(1, color="k", ls=":", lw=0.6)
    for lab, mk in (("band containing the line", "o"), ("interpolated", "D"),
                    ("spectrum median", "s"), ("same source, other grating", "^"),
                    ("grating or sample median", "x")):
        a_.plot([], [], mk, color="0.45", mfc="0.45" if mk != "x" else None, ms=4, label=lab)
    a_.set_yscale("log")
    a_.set_yticks([0.5, 1, 2, 3, 5])
    a_.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_xlabel(r"$z_{\rm sys}$")
    a_.set_ylabel(r"slit factor $S_{\rm H\alpha}$")
    a_.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a_.set_title(f"(a) S used, median {med_all:.2f} (dashed), NMAD "
                 f"{bh.nmad(own['ha_slit_factor'].to_numpy(float)):.2f}", fontsize=7.5)
    # b: end-to-end
    a_ = ax[1]
    g = phot_spec(d)
    r = np.log10(g["ratio"].clip(lower=1e-3).to_numpy(float))
    st = dex_stats(r)
    if st["N"]:
        bins = np.arange(-1.0, 1.05, 0.1)
        a_.hist(np.clip(r, -0.99, 0.99), bins=bins, color="#8c4a8c", edgecolor="k", lw=0.5)
        a_.axvline(0, color="k", ls=":", lw=0.8)
        a_.axvline(st["med"], color="k", ls="--", lw=0.9)
    a_.set_xticks([-1, -0.5, 0, 0.5, 1])
    a_.set_xlabel(r"$\log_{10}$ (photometric / slit-corrected spectral H$\alpha$)")
    a_.set_ylabel("N")
    a_.set_title(f"(b) median {st['med']:+.2f} dex, NMAD {st['nmad']:.2f} dex, N = {st['N']}", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # numbers
    print("\n  SLIT-LOSS CONSISTENCY")
    print(f"    S used for Halpha: {len(own)} own factors, median {med_all:.2f}, NMAD "
          f"{bh.nmad(own['ha_slit_factor'].to_numpy(float)):.2f}; "
          + ", ".join(f"{int((sp['ha_slit_source'] == k).sum())} {k}" for k in FALLBACK_S))
    print("    by grating: " + "; ".join(lines))
    if len(own):
        rel = own["ha_slit_factor"] / own["ha_grating"].map(own.groupby("ha_grating")["ha_slit_factor"].median())
        print(f"    scatter of S about its own grating's median: {100 * bh.nmad(rel.to_numpy(float)):.0f} per cent, "
              f"whole sample {100 * bh.nmad(own['ha_slit_factor'].to_numpy(float)) / med_all:.0f} per cent")
    if slit_bands is not None and len(slit_bands):
        frac = []
        for _, gg in slit_bands[slit_bands["ID"].isin(d["ID"])].groupby(["ID", "grating"]):
            if len(gg) >= 2:
                frac.append(bh.nmad(gg["S"].to_numpy(float)) / np.median(gg["S"]))
        if frac:
            print(f"    band-to-band scatter of S within one spectrum: median {100 * np.median(frac):.0f} per cent "
                  f"({len(frac)} spectra with >= 2 bands)")
    print(f"    photometric / slit-corrected spectral Halpha: median {st['med']:+.2f} dex "
          f"(x{10 ** st['med']:.2f}), 16-84th {st['p16']:+.2f} to {st['p84']:+.2f} dex, "
          f"NMAD {st['nmad']:.2f} dex, N = {st['N']}")
    return st


def fig_dust_compare(d, path):
    """A_Ha and f_esc with Balmer against SED dust for case 1, and the offsets."""
    c1 = d[(d["ha_case"] == 1) & d["a_ha_balmer"].notna() & d["a_ha_sed"].notna()].copy()
    c1["dA"] = num(c1, "a_ha_balmer") - num(c1, "a_ha_sed")
    fe, fs = num(c1, "fesc_lya"), num(c1, "fesc_lya_sed")
    lim = num(c1, "fesc_lya_is_limit") == 1
    c1["dlogf"] = np.log10(fe / fs)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a_ = ax[0]
    pts(a_, c1["a_ha_sed"], c1["a_ha_balmer"], c1["det"], "#2a9d8f", xerr=e2(c1, "a_ha_sed"),
        yerr=e2(c1, "a_ha_balmer"))
    top = max(1.5, float(np.nanmax(np.r_[1.5, c1["a_ha_balmer"], c1["a_ha_sed"]])) * 1.08)
    a_.plot([0, top], [0, top], "k--", lw=0.8)
    sa = dex_stats(c1["dA"])
    sad = dex_stats(c1.loc[c1["det"], "dA"])
    a_.set_xlim(-0.05, top)
    a_.set_ylim(-0.05, top)
    a_.set_xlabel(r"$A_{\rm H\alpha}$ SED [mag]")
    a_.set_ylabel(r"$A_{\rm H\alpha}$ Balmer decrement [mag]")
    a_.set_title(f"(a) Balmer $-$ SED: median {sa['med']:+.2f} mag, NMAD {sa['nmad']:.2f} (N = {sa['N']})",
                 fontsize=7.5)
    det_legend(a_)
    a_.legend(frameon=False, fontsize=6, loc="upper left")
    a_ = ax[1]
    ok = fe.notna() & fs.notna() & (fe > 0) & (fs > 0)
    m = ok & ~lim
    pts(a_, fs[m], fe[m], c1.loc[m, "det"], "#2a9d8f", xerr=e2(c1[m], "fesc_lya_sed"),
        yerr=e2(c1[m], "fesc_lya"))
    a_.scatter(fs[ok & lim], fe[ok & lim], marker="v", facecolors="white", edgecolors="#2a9d8f", s=14,
               label="5$\\sigma$ limits")
    sf = dex_stats(c1.loc[m, "dlogf"])
    if ok.any():
        v = np.r_[fe[ok], fs[ok]]
        lo_, hi_ = v.min() / 2, max(1.0, v.max() * 2)
        a_.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.8)
        if np.isfinite(sf["med"]):
            a_.plot([lo_, hi_], [lo_ * 10 ** sf["med"], hi_ * 10 ** sf["med"]], color="0.4", lw=0.8, ls=":",
                    label=f"median offset {sf['med']:+.2f} dex")
        a_.set_xlim(lo_, hi_)
        a_.set_ylim(lo_, hi_)
    a_.set_xscale("log")
    a_.set_yscale("log")
    a_.set_xlabel(r"$f_{\rm esc}^{\rm Ly\alpha}$ with SED $A_{\rm H\alpha}$")
    a_.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$ with Balmer $A_{\rm H\alpha}$")
    a_.set_title(rf"(b) Ly$\alpha$ detections: median {sf['med']:+.2f} dex, NMAD {sf['nmad']:.2f} (N = {sf['N']})",
                 fontsize=7.5)
    a_.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print("\n  BALMER AGAINST SED DUST (case 1)")
    print(f"    A_Ha Balmer - SED, all case 1: median {sa['med']:+.2f} mag, 16-84th {sa['p16']:+.2f} to "
          f"{sa['p84']:+.2f}, NMAD {sa['nmad']:.2f}, N = {sa['N']}")
    print(f"    A_Ha Balmer - SED, Lya detections: median {sad['med']:+.2f} mag, NMAD {sad['nmad']:.2f}, N = {sad['N']}")
    for bs in ("same", "different"):
        x = dex_stats(c1.loc[c1["balmer_spectra"] == bs, "dA"])
        if x["N"]:
            print(f"      {bs:<10s} spectrum: median {x['med']:+.2f} mag, N = {x['N']}")
    print(f"    log10 f_esc(Balmer) - log10 f_esc(SED), Lya detections: median {sf['med']:+.2f} dex "
          f"(x{10 ** sf['med']:.2f}), 16-84th {sf['p16']:+.2f} to {sf['p84']:+.2f} dex, NMAD {sf['nmad']:.2f}, "
          f"N = {sf['N']}")
    print("    (per source the f_esc offset is exactly -0.4 x the A_Ha offset, as only the dust differs)")
    return sa, sf

# ============================================================================
# Flags and summary
# ============================================================================

def fv(r, k):
    """A float from a row, NaN if missing or not a number."""
    try:
        return float(r.get(k, np.nan))
    except (TypeError, ValueError):
        return np.nan


def flags(d, b, sp, a):
    issues = {i: [] for i in d["ID"]}
    lo, hi = a.slit_warn
    for _, r in d.iterrows():
        i = r["ID"]
        S = fv(r, "ha_slit_factor")
        if r.get("ha_slit_source") in OWN_S and np.isfinite(S) and not (lo <= S <= hi):
            issues[i].append(f"slit_out S_Ha {S:.2f} in {r.get('ha_slit_band')}")
        if r.get("ha_source") == "spec" and r.get("ha_slit_source") in FALLBACK_S:
            note = r.get("ha_slit_note")
            issues[i].append(f"slit_median {r.get('ha_slit_source')}" + (f" ({note})" if isinstance(note, str) and note else ""))
        R, Rhi = fv(r, "balmer_ratio"), fv(r, "balmer_ratio_err_hi")
        if np.isfinite(R) and np.isfinite(Rhi) and Rhi > 0 and (bh.R_INT - R) / Rhi > 2:
            issues[i].append(f"balmer_low R {R:.2f} +{Rhi:.2f}")
        ab, asd = fv(r, "a_ha_balmer"), fv(r, "a_ha_sed")
        if np.isfinite(ab) and np.isfinite(asd):
            s = np.hypot(0.5 * (fv(r, "a_ha_balmer_err_lo") + fv(r, "a_ha_balmer_err_hi")),
                         0.5 * (fv(r, "a_ha_sed_err_lo") + fv(r, "a_ha_sed_err_hi")))
            if abs(ab - asd) > 0.5 and np.isfinite(s) and s > 0 and abs(ab - asd) / s > 3:
                issues[i].append(f"dust_offset Balmer {ab:.2f} SED {asd:.2f} ({abs(ab - asd) / s:.1f} sigma)")
        f, fs = fv(r, "fesc_lya"), fv(r, "fesc_lya_sed")
        if np.isfinite(f) and f > 1:
            issues[i].append(f"fesc_gt1 {f:.2f}" + (" (limit)" if fv(r, "fesc_lya_is_limit") == 1 else ""))
        if r.get("ha_case") == 1 and np.isfinite(f) and np.isfinite(fs) and fs > 0 and not (1 / 3 <= f / fs <= 3):
            issues[i].append(f"fesc_dust Balmer/SED {f / fs:.2f}")
    for _, r in phot_spec(d).iterrows():
        if not (a.phot_warn[0] <= r["ratio"] <= a.phot_warn[1]):
            issues[r["ID"]].append(f"phot_spec {r['ratio']:.2f} in {r['ha_phot_band']}")
    m = sp.assign(fam=sp["grating"].map(gfam)).groupby(["ID", "fam"])["S_3um"].median().unstack()
    if {"G235", "G395"} <= set(m.columns):
        for i, r in m.dropna(subset=["G235", "G395"]).iterrows():
            q = r["G395"] / r["G235"]
            if not (1 / a.ratio_warn <= q <= a.ratio_warn) and i in issues:
                issues[i].append(f"grating S G395/G235 {q:.2f} ({r['G235']:.2f}, {r['G395']:.2f})")
    for _, r in sp[sp["S_spread"] > a.ratio_warn].iterrows():
        if r["ID"] in issues:
            issues[r["ID"]].append(f"chromatic {r['grating']} max/min S {r['S_spread']:.2f} "
                                   f"over {int(r['n_bands'])} bands")
    o = d.set_index("ID")
    rows = [dict(ID=i, z_sys=o.at[i, "z_sys"], lya_det=int(o.at[i, "det"]),
                 in_analysis=int(o.at[i, "in_analysis"]), ha_case=int(o.at[i, "ha_case"]),
                 n_issues=len(v), issues=" | ".join(v)) for i, v in issues.items() if v]
    return pd.DataFrame(rows, columns=["ID", "z_sys", "lya_det", "in_analysis", "ha_case",
                                       "n_issues", "issues"])


def summary(d, b, sp, fl):
    det = d["det"]
    print(f"\n  sources {len(d)} ({int(det.sum())} Lya detections)")
    print("  case                                 all   Lya det")
    for c, lab in ((1, "1  spec Halpha, Balmer"), (2, "2  spec Halpha, SED A_Ha"),
                   (3, "3  phot Halpha, SED A_Ha"), (0, "0  no Halpha")):
        m = d["ha_case"] == c
        print(f"    {lab:<32s} {int(m.sum()):>5d} {int((m & det).sum()):>9d}")
    for bs in ("same", "different"):
        m = (d["ha_case"] == 1) & (d["balmer_spectra"] == bs)
        print(f"      case 1 Balmer, {bs:<17s} {int(m.sum()):>5d} {int((m & det).sum()):>9d}")

    nb = b[b["nominal"] == 1]
    print(f"\n  slit factors in every band: {len(b)} band measurements in "
          f"{b.groupby(['ID', 'grating']).ngroups} spectra, {len(nb)} in nominal bands")
    for gf in GR_COL:
        g = nb[nb["grating"].map(gfam) == gf]
        if len(g):
            print(f"    {gf}  median S {g['S'].median():.2f}, NMAD {bh.nmad(g['S'].to_numpy()):.2f}, "
                  f"{g['ID'].nunique()} sources")
    ext = b[b["nominal"] == 0]
    if len(ext):
        r = []
        for _, e in ext.iterrows():
            same = nb[(nb["ID"] == e["ID"]) & (nb["grating"] == e["grating"])]
            if len(same):
                r.append(e["S"] / same["S"].median())
        if r:
            print(f"    bands beyond the nominal range: {len(ext)}, S / nominal S of the same "
                  f"spectrum median {np.median(r):.2f}, 16-84th {np.percentile(r, 16):.2f} "
                  f"to {np.percentile(r, 84):.2f}")
    sl = sp["slope"].dropna()
    if len(sl):
        med = float(np.median(sl))
        q = (bh.HA_REST / bh.HB_REST) ** med
        dA = 2.5 * bh.K_HA / (bh.K_HB - bh.K_HA) * np.log10(q)
        print(f"    slope d ln S / d ln lambda: median {med:+.2f}, NMAD {bh.nmad(sl.to_numpy()):.2f} "
              f"({len(sl)} spectra)")
        print(f"      so in one spectrum S_Ha / S_Hb = {q:.3f}, and A_Ha from the observed ratio "
              f"is {abs(dA):.2f} mag too {'low' if dA > 0 else 'high'} (the slit-corrected "
              f"ratio, --balmer-basis slitcorr, removes this)")
    m = sp.assign(fam=sp["grating"].map(gfam)).groupby(["ID", "fam"])["S_3um"].median().unstack()
    if {"G235", "G395"} <= set(m.columns):
        both = m.dropna(subset=["G235", "G395"])
        if len(both):
            q = both["G395"] / both["G235"]
            print(f"    G395 / G235 S at {REF_UM:g} um (median slope applied), {len(both)} sources with both: median {q.median():.2f}, "
                  f"16-84th {q.quantile(.16):.2f} to {q.quantile(.84):.2f}")

    c1 = d[d["a_ha_balmer"].notna() & d["a_ha_sed"].notna()]
    if len(c1):
        dA = c1["a_ha_balmer"] - c1["a_ha_sed"]
        print(f"\n  A_Ha Balmer - SED ({len(c1)}): median {dA.median():+.2f}, NMAD "
              f"{bh.nmad(dA.to_numpy()):.2f} mag")
        for bs in ("same", "different"):
            g = dA[c1["balmer_spectra"] == bs]
            if len(g):
                print(f"    {bs:<10s} {len(g):>3d}  median {g.median():+.2f}")
    g = phot_spec(d)
    if len(g):
        print(f"  photometric / spectral Halpha ({len(g)}): median {g['ratio'].median():.2f}, "
              f"NMAD {bh.nmad(g['ratio'].to_numpy()):.2f}")
        for src in dict.fromkeys(g["ha_slit_source"].dropna()):
            gg = g[g["ha_slit_source"] == src]
            if len(gg):
                print(f"    {src:<14s} slit factor {len(gg):>3d}  median {gg['ratio'].median():.2f}")

    print(f"\n  sources with a flag: {len(fl)} ({int((fl['lya_det'] == 1).sum())} Lya detections)")
    kinds = pd.Series([s.split(" ")[0] for v in fl["issues"] for s in v.split(" | ")])
    for k, n in kinds.value_counts().items():
        print(f"    {k:<12s} {n}")
    for lab, sub in (("Lya detections", fl[fl["lya_det"] == 1]), ("non-detections", fl[fl["lya_det"] == 0])):
        if len(sub):
            print(f"\n  flagged {lab}")
            for _, r in sub.sort_values("ID").iterrows():
                print(f"    {int(r['ID']):>6d}  z {r['z_sys']:.3f}  case {r['ha_case']}  {r['issues']}")


def main():
    a = parse_args()
    print("checks/check_ha_corrections.py")
    for k in ("ha_csv", "catalog", "isaac", "spec_root", "filter_dir", "bands_csv", "slit_bands_csv",
              "flags_csv", "outdir"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    print(f"  reading  {a.ha_csv}")
    print(f"  reading  {a.catalog}")
    print(f"  reading  {a.slit_bands_csv}" + ("" if os.path.exists(a.slit_bands_csv) else "  (missing, skipped)"))
    if not a.reuse_bands:
        print(f"  reading  {a.isaac}")
        print(f"  reading  {a.spec_root}/<grating>_<filter>/<ID>_*_spectra_lime.fits")
        print(f"  reading  {a.filter_dir}/JWST_NIRCam.<FILT>.dat")
    os.makedirs(a.outdir, exist_ok=True)
    figs = {k: f"{a.outdir}/{k}.png" for k in ("slit_summary", "dust_balmer_vs_sed", "slit_factors",
                                                 "balmer_dust", "ha_phot_vs_spec", "fesc_checks")}
    print(f"  {'reading' if a.reuse_bands else 'writing'}  {a.bands_csv}")
    print(f"  writing  {a.flags_csv}")
    for p_ in figs.values():
        print(f"  writing  {p_}")
    print(f"  sample {a.sample}")

    d = load_inputs(a)
    if a.reuse_bands:
        b = pd.read_csv(a.bands_csv)
        b = b[b["ID"].isin(d["ID"])]
    else:
        b = all_band_factors(d, a)
        b.to_csv(a.bands_csv, index=False)
    sp = per_spectrum(b)

    style()
    fig_slit(d, b, sp, figs["slit_factors"])
    fig_balmer(d, figs["balmer_dust"])
    fig_phot(d, figs["ha_phot_vs_spec"])
    if num(d, "fesc_lya").notna().any():
        fig_fesc(d, figs["fesc_checks"])
    else:
        print("  no fesc_lya in the catalogue, fesc_checks.png not made "
              "(run measure_lya_fesc.py then build_lya_catalog.py)")
    fl = flags(d, b, sp, a)
    fl.to_csv(a.flags_csv, index=False)
    summary(d, b, sp, fl)
    slit_bands = pd.read_csv(a.slit_bands_csv) if os.path.exists(a.slit_bands_csv) else None
    fig_slit_summary(d, figs["slit_summary"], slit_bands)
    if num(d, "fesc_lya").notna().any():
        fig_dust_compare(d, figs["dust_balmer_vs_sed"])
    print("\n  done")
    for p_ in [a.bands_csv, a.flags_csv] + list(figs.values()):
        if os.path.exists(p_):
            print(f"  {p_}")


if __name__ == "__main__":
    main()