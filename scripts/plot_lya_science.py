#!/usr/bin/env python3
"""
plot_lya_science.py

Every science figure of the Lya detections (and, with --sets, the
non-detections) of the z_sys sample, each with correlation statistics, plus
histograms of every property. The idea is to make every figure with a
physical reason to exist, then choose the ones for the paper.

A source is a detection if lya_det is 1 in lya_catalog.csv, a non-detection
if it is 0 (the rule chosen in build_lya_catalog.py, det_rule). Only
in_analysis == 1 is used, so close pairs and interlopers are left out and
50803 and 51988 are kept. Untested sources (65863) are in neither set. For
non-detections Delta_v and FWHM come from fits to low-S/N spectra, so treat
them as a check.

Reads only MUSE_catalogs/lya_catalog.csv (build_lya_catalog.py). Delta_v,
FWHM and f_blue are those of the preferred fit (lya_model), the skewed
Gaussian for a single line and the red peak for a double. M_UV is the SED
sed_muv by default (--muv-source fit for the interim power-law fit, names
then gain muvfit_), used where its mean error < --muv-max-err (0.5 mag).
The other galaxy properties are Ken's SED values (sed_logM, sed_logsfr100,
sed_logssfr100, sed_av, sed_logxion), drawn with their asymmetric 16th to
84th percentile errors.

Output
------
One folder per detection rule and set, --outdir/<rule>_<set>/ (default
/ceph/cephfs/apatrick/P2/plots/lya_science/snr5_detections/), holding
  <section>_<plot>[_<bins>].png   one figure each, names below
  stats.csv                        one row per figure panel, see Statistics
  all_figures.pdf                  every figure of the set, one per page, to browse
and, once for both sets, --outdir/<rule>_det_vs_nondet/ for the comparison of
detections and non-detections. Reruns overwrite.

Binned versions
---------------
Most scatter plots are also made split into panels, named with a suffix
  _zbins    three redshift bins at --z-edges (3.5 4.5)
  _zbins2   two redshift bins at --z-split (3.5)
  _muvbins  two M_UV bins at the median M_UV of the set
  _massbins two M* bins at the median
  _ssfrbins two sSFR100 bins at the median
  _avbins   two A_V bins at the median
  _model    single and double lines
A split is skipped where it would be one of the plotted axes. --bins limits
which splits are made (default all).

Figures (<section>_<plot>)
--------------------------
hist_   distributions of every property, and hist_overview.png with them all
        hist_fwhm, hist_dv, hist_z, hist_muv, hist_logM, hist_logsfr100,
        hist_logssfr100, hist_av, hist_aha, hist_logxion, hist_snr, hist_flux,
        hist_fblue_int, hist_fblue_data, hist_sep, hist_br, hist_dv_blue,
        hist_ha_fwhm, hist_verhamme_resid. hist_fwhm and hist_dv also _zbins[2].

A. Can the Lya profile recover systemic? (sys_)
  sys_dv_vs_fwhm             Delta_v against FWHM (Verhamme et al. 2018,
                             shell models, both grow with N_HI)
  sys_dv_vs_fwhm_verhamme    with Verhamme+18 Eq. 2 and its band, the 1:1
                             line and a linear fit to our points
  sys_verhamme_resid         Delta_v - Delta_v(Verhamme Eq. 2), histogram with
                             median and robust sigma, and against z and M_UV
  sys_dvred_vs_halfsep       doubles, Delta_v_red against half the peak
                             separation with 1:1 (systemic midway between the
                             peaks, Verhamme et al. 2018)
  sys_recovery_doubles       doubles, how far systemic is from the Eq. 2
                             prediction and from the midpoint of the peaks

B. Radiative transfer through the ISM and CGM (rt_)
  rt_fblue_<kind>_vs_dv      f_blue against Delta_v (low N_HI, small offset
                             and blue flux, Verhamme et al. 2015, CLASSY)
  rt_br_vs_sep, rt_br_vs_dvred   doubles, blue/red flux ratio (outflows
                             suppress the blue peak)
  rt_sep_vs_z                doubles, peak separation (the N_HI and LyC
                             leakage indicator, Verhamme et al. 2015,
                             Izotov et al. 2018) against z
  rt_fwhm_vs_ha_fwhm, rt_dv_vs_ha_fwhm   Lya against the Halpha width, 1:1
                             (radiative-transfer broadening beyond the
                             nebular kinematics, Erb et al. 2014)

C. The IGM and redshift (igm_)
  igm_dv_vs_z                Delta_v against z (IGM removes flux near
                             systemic), best read in _muvbins
  igm_fblue_<kind>_vs_z      f_blue against z (Hayes et al. 2021, zELDA)
  igm_double_frac_vs_z       fraction of doubles in z bins at dBIC > 2, 6
                             and 10, Wilson 68% intervals, a lower limit
  igm_fwhm_vs_z              FWHM against z with (1+z)^(2/3) scaled to the
                             median (Prieto-Lyon et al. 2025) and a power-law
                             fit in log FWHM against log(1+z)

D. Galaxy properties (gal_)
  gal_dv_vs_muv              with Mason+18 (the Delta_v-M_UV relation behind
                             x_HI inferences)
  gal_fwhm_vs_muv            with Mason+19 and Prieto-Lyon+25
  gal_fwhm_models[_bins]     the FWHM distribution with Prieto-Lyon+25,
                             Mason+18 and Mason+19, as before
  gal_<q>_vs_<p>             q = dv, fwhm, fblue_int, p = logM, logsfr100,
                             logssfr100, av, logxion (halo mass, feedback,
                             dust and the ionising spectrum)

F. Checks (chk_)
  chk_dv_by_zline, chk_dv_by_grating   Delta_v split by the line and
                             grating that set z_sys, with KS p
  chk_dv_single_vs_double    doubles, the skewed-Gaussian Delta_v against the
                             red-peak Delta_v
  chk_dv_vs_snr, chk_fwhm_vs_snr   do the brightest lines drive the trends?
  chk_fblue_int_vs_data      the model and data f_blue
  (det_vs_nondet) chk_flux_vs_z, chk_snr_vs_z, the selection in z, and
                  det_vs_nondet_cdfs.png, cumulative distributions of z,
                  M_UV, M*, SFR100, sSFR100, A_V and xi_ion for the two
                  sets with Anderson-Darling and KS p

Statistics
----------
Each scatter panel gets Spearman rho and its p in the corner. stats.csv has,
per panel, N, Spearman rho and p, the 16th to 84th percentile of rho over
--n-stat-mc (1000) redraws of the points within their errors (asymmetric
where the errors are), Kendall tau and p, and for panels not split by z and
with neither axis z, the partial Spearman rho controlling for z_sys with its
p. Where a fit is drawn its slope and intercept with bootstrap errors are
added. Lines MUSE does not resolve enter the statistics at their FWHM upper
limit and are left out of the fits. With N = 37 a correlation needs |rho|
above about 0.33 for p < 0.05, about 0.47 in a panel of 18.

Line fits minimise sum (y - a x - b)^2 / (sig_y^2 + a^2 sig_x^2), the
effective-variance form that allows errors on both axes, with --n-boot
bootstrap resamples for the errors.

FWHM, the LSF and f_blue
------------------------
FWHM is the intrinsic width (--fwhm-kind int), --fwhm-kind obs the observed
one (names then gain obs_). Unresolved lines get an upper limit of
--resolve-frac x FWHM_LSF and are drawn open with an arrow. Verhamme+18 used
widths not corrected for the instrument, so obs is the closer comparison for
the sys_ figures. f_blue figures are made for each of --fblue-kinds (int
and data). The data version is smeared by the LSF, which by itself moves
flux of lines near systemic to the blue side, so f_blue_data falls with
Delta_v even with no physics. Use int for the physics and data as a check.

Usage
-----
python plot_lya_science.py                          # detections
python plot_lya_science.py --sets detections non_detections
python plot_lya_science.py --bins z zbins2 model    # only these splits
python plot_lya_science.py --fwhm-kind obs
python plot_lya_science.py --muv-source fit         # interim power-law M_UV
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy import stats as sps
from scipy.optimize import minimize

try:
    import cmasher as cmr
    _HAVE_CMASHER = True
except ImportError:
    _HAVE_CMASHER = False

import p2_common as pc

LYA_CATALOG = getattr(pc, "LYA_CATALOG", f"{pc.MUSE_CAT_DIR}/lya_catalog.csv")
SETS = {"detections": 1, "non_detections": 0}

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

# ---------------------------------------------------------------------------
# Quantities. Every column the plots use is renamed to a short key in
# load_sample, with <key>_lo and <key>_hi errors (equal when symmetric).
# key -> (axis label, histogram bin width)
# ---------------------------------------------------------------------------
Q = {
    "z": (r"$z_{\rm sys}$", 0.2),
    "dv": (r"$\Delta v_{\rm Ly\alpha}$ [km s$^{-1}$]", 100.0),
    "fwhm": (r"FWHM$_{\rm Ly\alpha,\,int}$ [km s$^{-1}$]", 50.0),
    "muv": (r"$M_{\rm UV}$", 0.25),
    "logM": (r"$\log_{10}(M_\star/{\rm M}_\odot)$", 0.25),
    "logsfr100": (r"$\log_{10}({\rm SFR}_{100}/{\rm M}_\odot\,{\rm yr}^{-1})$", 0.25),
    "logssfr100": (r"$\log_{10}({\rm sSFR}_{100}/{\rm yr}^{-1})$", 0.25),
    "av": (r"$A_V$ [mag]", 0.1),
    "aha": (r"$A_{\rm H\alpha}$ [mag]", 0.1),
    "logxion": (r"$\log_{10}(\xi_{\rm ion}/{\rm Hz\,erg^{-1}})$", 0.1),
    "snr": (r"$\Sigma_{\rm Ly\alpha}$", 2.0),
    "flux": (r"$F_{\rm Ly\alpha}$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$]", 50.0),
    "fblue_int": (r"$f_{\rm blue}$ (intrinsic model)", 0.05),
    "fblue_conv": (r"$f_{\rm blue}$ (model with LSF)", 0.05),
    "fblue_data": (r"$f_{\rm blue}$ (data)", 0.05),
    "sep": (r"$\Delta v_{\rm sep}$ [km s$^{-1}$]", 100.0),
    "halfsep": (r"$\Delta v_{\rm sep}/2$ [km s$^{-1}$]", 50.0),
    "br": (r"$F_{\rm blue}/F_{\rm red}$", 0.1),
    "dv_blue": (r"$\Delta v_{\rm blue}$ [km s$^{-1}$]", 100.0),
    "dv_single": (r"$\Delta v$ single skewed Gaussian [km s$^{-1}$]", 100.0),
    "ha_fwhm": (r"FWHM$_{\rm H\alpha,\,int}$ [km s$^{-1}$]", 50.0),
    "vresid": (r"$\Delta v - \Delta v_{\rm V18}$ [km s$^{-1}$]", 50.0),
    "vmid": (r"$(\Delta v_{\rm red}+\Delta v_{\rm blue})/2$ [km s$^{-1}$]", 50.0),
}
MUV_SOURCES = {
    # --muv-source -> (value, lower error, upper error, mean error)
    "sed": ("sed_muv", "sed_muv_err_lo", "sed_muv_err_hi", "sed_muv_err"),
    "fit": ("muv_fit", "muv_fit_err", "muv_fit_err", "muv_fit_err"),
}
SED_MAP = {"logM": "sed_logM", "logsfr100": "sed_logsfr100", "logssfr100": "sed_logssfr100",
           "av": "sed_av", "aha": "sed_aha", "logxion": "sed_logxion"}
GAL_PROPS = ["logM", "logsfr100", "logssfr100", "av", "logxion"]
BIN_DEFS = {   # split -> (key, kind)
    "zbins": ("z", "zedges"), "zbins2": ("z", "zsplit"), "muvbins": ("muv", "median"),
    "massbins": ("logM", "median"), "ssfrbins": ("logssfr100", "median"),
    "avbins": ("av", "median"), "model": ("model", "category"),
}
SYM = {"z": "z", "muv": r"M_{\rm UV}", "logM": r"\log M_\star", "logssfr100": r"\log {\rm sSFR}",
       "av": "A_V"}


FWHM_IS_INT = True


def set_fwhm_kind(kind):
    global FWHM_IS_INT
    FWHM_IS_INT = kind == "int"
    Q["fwhm"] = ((r"FWHM$_{\rm Ly\alpha,\,int}$ [km s$^{-1}$]" if kind == "int"
                  else r"FWHM$_{\rm Ly\alpha,\,obs}$ [km s$^{-1}$]"), 50.0)


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
    return {1: MNRAS_COL_WIDTH_IN, 2: 5.0}.get(n, MNRAS_PAGE_WIDTH_IN)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _num(cat, col):
    return pd.to_numeric(cat[col], errors="coerce") if col in cat.columns else \
        pd.Series(np.nan, index=cat.index)


def put(df, key, val, lo=None, hi=None):
    df[key] = val.values if hasattr(val, "values") else val
    lo = 0.0 if lo is None else lo
    hi = lo if hi is None else hi
    df[f"{key}_lo"] = lo.values if hasattr(lo, "values") else lo
    df[f"{key}_hi"] = hi.values if hasattr(hi, "values") else hi


def load_sample(catalog_csv, fwhm_kind="int", resolve_frac=0.5, muv_max_err=0.5,
                muv_source="sed"):
    """Every source of the catalogue, with the short keys of Q and their
    _lo and _hi errors, plus det, in_analysis, model, resolved, dbic,
    zline and grating."""
    cat = pd.read_csv(catalog_csv)
    cat["ID"] = cat["ID"].astype(int)
    need = ["ID", "z_sys", "delta_v", "delta_v_err", "lya_det", "lya_status", "in_analysis",
            "pair_status", f"fwhm_{fwhm_kind}_kms", f"fwhm_{fwhm_kind}_kms_err",
            "fwhm_lsf_kms", "fwhm_resolved"]
    missing = [c for c in need if c not in cat.columns]
    if missing:
        raise SystemExit(f"{catalog_csv} has no {missing}. Rerun build_lya_catalog.py")
    vcol, locol, hicol, ecol = MUV_SOURCES[muv_source]
    for c in dict.fromkeys((vcol, locol, hicol, ecol, "f_blue_int", "sed_logM")):
        if c not in cat.columns:
            print(f"  WARNING {c} not in {catalog_csv}. Rerun build_lya_catalog.py "
                  "(after build_sed_catalog.py and measure_lya_shape.py)")

    df = pd.DataFrame({"ID": cat["ID"]})
    df["det"] = _num(cat, "lya_det")
    df["in_analysis"] = _num(cat, "in_analysis").fillna(0).astype(int)
    df["pair_status"] = cat["pair_status"]
    df["model"] = cat["lya_model"].fillna("") if "lya_model" in cat.columns else ""
    df["dbic"] = _num(cat, "dbic_double")
    df["zline"] = cat["z_sys_line"] if "z_sys_line" in cat.columns else np.nan
    df["grating"] = cat["z_sys_grating"] if "z_sys_grating" in cat.columns else np.nan

    put(df, "z", _num(cat, "z_sys"), _num(cat, "z_sys_err"))
    put(df, "dv", _num(cat, "delta_v"), _num(cat, "delta_v_err"))
    fw, fe = _num(cat, f"fwhm_{fwhm_kind}_kms"), _num(cat, f"fwhm_{fwhm_kind}_kms_err")
    resolved = _num(cat, "fwhm_resolved") == 1
    no_fit = ~np.isfinite(_num(cat, "fwhm_lsf_kms"))
    resolved[no_fit] = True
    if fwhm_kind == "int":
        lim = resolve_frac * _num(cat, "fwhm_lsf_kms")
        fw = fw.where(resolved, lim)
        fe = fe.where(resolved, 0.0)
    put(df, "fwhm", fw, fe)
    df["resolved"] = resolved.values

    m, mlo, mhi, me = (_num(cat, c) for c in (vcol, locol, hicol, ecol))
    bad = ~(me < muv_max_err)
    put(df, "muv", m.mask(bad), mlo.mask(bad), mhi.mask(bad))
    for key, col in SED_MAP.items():
        put(df, key, _num(cat, col), _num(cat, f"{col}_err_lo"), _num(cat, f"{col}_err_hi"))
    put(df, "snr", _num(cat, "lya_snr"))
    put(df, "flux", _num(cat, "lya_flux"), _num(cat, "lya_flux_err"))
    for k in ("int", "conv", "data"):
        put(df, f"fblue_{k}", _num(cat, f"f_blue_{k}"), _num(cat, f"f_blue_{k}_err_lo"),
            _num(cat, f"f_blue_{k}_err_hi"))
    put(df, "sep", _num(cat, "peak_sep"), _num(cat, "peak_sep_err"))
    put(df, "halfsep", 0.5 * _num(cat, "peak_sep"), 0.5 * _num(cat, "peak_sep_err"))
    put(df, "br", _num(cat, "blue_red_ratio"), _num(cat, "blue_red_ratio_err"))
    put(df, "dv_blue", _num(cat, "dv_blue"), _num(cat, "dv_blue_err"))
    put(df, "dv_single", _num(cat, "delta_v_single"), _num(cat, "delta_v_single_err"))
    hf, he = _num(cat, "ha_fwhm_int_kms"), _num(cat, "ha_fwhm_int_kms_err")
    put(df, "ha_fwhm", hf, he)
    # Verhamme et al. 2018 Eq. 2 residual and the midpoint of the two peaks
    put(df, "vresid", df["dv"] - (VERH_SLOPE * df["fwhm"] + VERH_INT),
        np.hypot(df["dv_lo"], VERH_SLOPE * df["fwhm_lo"]))
    put(df, "vmid", 0.5 * (df["dv"] + df["dv_blue"]), 0.5 * np.hypot(df["dv_lo"], df["dv_blue_lo"]))
    return df


def finite(d, keys):
    m = np.ones(len(d), bool)
    for k in keys:
        m &= np.isfinite(d[k].to_numpy(dtype=float))
    return d[m]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def perturb_many(d, key, n, rng):
    v = d[key].to_numpy(float)
    lo = np.nan_to_num(d[f"{key}_lo"].to_numpy(float))
    hi = np.nan_to_num(d[f"{key}_hi"].to_numpy(float))
    u = rng.standard_normal((n, v.size))
    return v + np.where(u < 0, u * lo, u * hi)


def spearman_mc(d, x, y, n, rng):
    """Spearman rho of n redraws of the points within their errors, at once
    (ranks of each row, then the row-wise Pearson correlation)."""
    def ranks(a):
        return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
    rx = ranks(perturb_many(d, x, n, rng))
    ry = ranks(perturb_many(d, y, n, rng))
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / den


def partial_spearman(x, y, z):
    rxy = sps.spearmanr(x, y)[0]
    rxz = sps.spearmanr(x, z)[0]
    ryz = sps.spearmanr(y, z)[0]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if not den > 0:
        return np.nan, np.nan
    r = (rxy - rxz * ryz) / den
    n = len(x)
    if n < 5 or abs(r) >= 1:
        return r, np.nan
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return float(r), float(2 * sps.t.sf(abs(t), n - 3))


def correlation(d, x, y, n_mc, rng, control_z=True):
    """Spearman, its MC range, Kendall, and the partial rho controlling z."""
    d = finite(d, [x, y])
    out = dict(N=len(d))
    if len(d) < 4:
        return out
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    out["rho"], out["p_rho"] = (float(v) for v in sps.spearmanr(xv, yv))
    out["tau"], out["p_tau"] = (float(v) for v in sps.kendalltau(xv, yv))
    if n_mc > 0:
        rr = spearman_mc(d, x, y, n_mc, rng)
        out["rho_p16"], out["rho_p84"] = (float(v) for v in np.nanpercentile(rr, [16, 84]))
    if control_z and "z" not in (x, y) and len(d) >= 6:
        out["rho_partial_z"], out["p_partial_z"] = partial_spearman(xv, yv, d["z"].to_numpy(float))
    return out


def stats_text(st):
    if st.get("N", 0) < 4 or not np.isfinite(st.get("rho", np.nan)):
        return f"N = {st.get('N', 0)}"
    p = st["p_rho"]
    ptxt = f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.0e}"
    return rf"$\rho$ = {st['rho']:+.2f}, {ptxt}"


def fit_line(x, y, xe, ye, n_boot, rng):
    """y = a x + b with errors on both axes (effective variance), bootstrap
    errors. Returns (a, b, a_err, b_err) or None."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, xe, ye = x[ok], y[ok], np.nan_to_num(xe[ok]), np.nan_to_num(ye[ok])
    if x.size < 4:
        return None
    ye = np.where(ye > 0, ye, np.nanmedian(ye[ye > 0]) if np.any(ye > 0) else 1.0)

    def one(xx, yy, xxe, yye):
        a0, b0 = np.polyfit(xx, yy, 1)

        def nll(p):
            a, b = p
            v = yye ** 2 + (a * xxe) ** 2
            return np.sum((yy - a * xx - b) ** 2 / v + np.log(v))
        r = minimize(nll, [a0, b0], method="Nelder-Mead")
        return r.x if r.success else np.array([a0, b0])
    a, b = one(x, y, xe, ye)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, x.size, x.size)
        if np.ptp(x[i]) > 0:
            boots.append(one(x[i], y[i], xe[i], ye[i]))
    boots = np.array(boots) if boots else np.full((1, 2), np.nan)
    return float(a), float(b), float(np.nanstd(boots[:, 0])), float(np.nanstd(boots[:, 1]))


def wilson(k, n, z=1.0):
    """Wilson score interval (z = 1, 68%) for k of n."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class Ctx:
    """What every figure of a set needs: options, output folder, the PDF of
    all figures, the stats rows and a random generator."""

    def __init__(self, args, outdir, rule, name, pdf):
        self.args, self.outdir, self.rule, self.name, self.pdf = args, outdir, rule, name, pdf
        self.rows = []
        self.rng = np.random.default_rng(args.seed)
        self.title = f"{rule} {name.replace('_', '-')}"
        self.tag = ("" if args.fwhm_kind == "int" else "obs_") + \
                   ("" if args.muv_source == "sed" else "muvfit_")
        self.n_saved = 0
        self.ref = None    # the whole set, so median splits use the same edges everywhere

    def path(self, section, plot):
        return os.path.join(self.outdir, f"{section}_{self.tag}{plot}.png")

    def save(self, fig, section, plot):
        path = self.path(section, plot)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if self.pdf is not None:
            self.pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        self.n_saved += 1
        print(f"  saved  {path}")

    def stat(self, plot, panel, x, y, st, **extra):
        row = dict(plot=plot, panel=panel, x=x, y=y)
        row.update(st)
        row.update(extra)
        self.rows.append(row)


def err2(d, key):
    return np.nan_to_num(np.vstack([d[f"{key}_lo"].to_numpy(float),
                                    d[f"{key}_hi"].to_numpy(float)]))


def draw_points(ax, d, x, y, norm, cmap, ckey="z", s=18):
    """Grey error bars behind, points coloured by ckey on top. Unresolved
    lines (when x or y is fwhm) are open, and for the intrinsic FWHM carry an
    upper-limit arrow."""
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    xe, ye = err2(d, x), err2(d, y)
    cv = d[ckey].to_numpy(float)
    fw_axis = "x" if x == "fwhm" else "y" if y == "fwhm" else None
    res = d["resolved"].to_numpy(bool) if fw_axis else np.ones(len(d), bool)
    ax.errorbar(xv[res], yv[res], xerr=xe[:, res], yerr=ye[:, res], fmt="none",
                ecolor="0.6", elinewidth=0.6, capsize=0, zorder=1)
    cfin = np.isfinite(cv)
    sc = ax.scatter(xv[res & cfin], yv[res & cfin], c=cv[res & cfin], cmap=cmap, norm=norm,
                    s=s, edgecolor="black", linewidths=0.4, zorder=3)
    if np.any(res & ~cfin):     # no colour value (e.g. no M_UV): grey
        ax.scatter(xv[res & ~cfin], yv[res & ~cfin], color="0.75", s=s, edgecolor="black",
                   linewidths=0.4, zorder=3)
    u = ~res
    if u.any():
        fv = xv if fw_axis == "x" else yv
        fin = fv[np.isfinite(fv)]
        arrow = 0.08 * np.nanmax(np.abs(fin)) if fin.size else 20.0
        kw = dict(fmt="none", ecolor="0.6", color="0.6", elinewidth=0.6, capsize=2, zorder=1)
        if FWHM_IS_INT:
            if fw_axis == "x":
                ax.errorbar(xv[u], yv[u], xerr=np.full(u.sum(), arrow), xuplims=True,
                            yerr=ye[:, u], **kw)
            else:
                ax.errorbar(xv[u], yv[u], yerr=np.full(u.sum(), arrow), uplims=True,
                            xerr=xe[:, u], **kw)
        ec = np.array([cmap(norm(c)) if np.isfinite(c) else (0.6, 0.6, 0.6, 1.0) for c in cv[u]])
        ax.scatter(xv[u], yv[u], facecolors="white", edgecolors=ec, s=s, linewidths=1.0,
                   zorder=3)
    if np.any(np.isfinite(cv)):
        return sc
    return None


def bin_masks(values, edges, sym):
    edges = sorted(edges)
    fmt = (lambda e: f"{e:g}") if sym == "z" else (lambda e: f"{e:.2f}")
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


def split(d, how, args, ref):
    """(masks, labels) for a split name of BIN_DEFS, edges from ref (the whole
    set, so every figure of a set uses the same edges)."""
    key, kind = BIN_DEFS[how]
    if kind == "category":
        return [d["model"].to_numpy() == "single", d["model"].to_numpy() == "double"], \
               ["single", "double"]
    v = d[key].to_numpy(float)
    if kind == "zedges":
        edges = list(args.z_edges)
    elif kind == "zsplit":
        edges = [args.z_split]
    else:
        r = ref[key].to_numpy(float)
        r = r[np.isfinite(r)]
        if r.size < 4:
            return None, None
        edges = [float(np.median(r))]
    return bin_masks(v, edges, SYM.get(key, key))


def plain(lab):
    return (lab.replace("$", "").replace(r"\leq", "<=").replace(r"\geq", ">=")
               .replace(r"M_{\rm UV}", "M_UV").replace(r"\log M_\star", "logM")
               .replace(r"\log {\rm sSFR}", "log sSFR"))


def colour_setup(d, ckey):
    c = d[ckey].to_numpy(float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return Normalize(0, 1)
    return Normalize(vmin=c.min(), vmax=c.max() if c.max() > c.min() else c.min() + 1)


def scatter_fig(ctx, df, section, plot, x, y, bins=(), extras=None, fit=None,
                ckey="z", sample=None, xlim=None, ylim=None, one_to_one=False):
    """One figure of y against x (unbinned, coloured by ckey), then one per
    split in bins. extras(ax, d_panel, d_all) draws literature on each panel.
    fit is None, 'lin' (y = a x + b) or 'logy' (log10 y = a x + b), drawn
    and recorded in stats.csv."""
    d0 = df if sample is None else df[sample(df)]
    d0 = finite(d0, [x, y])
    if len(d0) < 3:
        print(f"  [skip] {section}_{plot}: {len(d0)} sources with {x} and {y}")
        return
    cmap = get_cmap()
    norm = colour_setup(d0, ckey)
    args = ctx.args
    variants = [("", None)] + [(b, b) for b in bins if b in args.bins
                               and BIN_DEFS[b][0] not in (x, y)
                               and not (b == "model" and sample is not None)]
    for suffix, how in variants:
        if how is None:
            masks, labels = [np.ones(len(d0), bool)], [""]
        else:
            masks, labels = split(d0, how, args, ctx.ref if ctx.ref is not None else d0)
            if masks is None:
                continue
        n = len(masks)
        h = 3.0 if n == 1 else 2.8
        fig, axes = plt.subplots(1, n, figsize=(panel_width(n), h), sharex=True, sharey=True,
                                 squeeze=False)
        axes = axes[0]
        sc = None
        for ax, m, lab in zip(axes, masks, labels):
            sub = d0[m]
            if one_to_one:
                lim = np.nanmax(np.abs(np.concatenate([d0[x], d0[y]]))) * 1.1
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="1:1", zorder=0)
            if extras is not None:
                extras(ax, sub, d0)
            if len(sub):
                s_ = draw_points(ax, sub, x, y, norm, cmap, ckey)
                sc = s_ if s_ is not None else sc
            st = correlation(sub, x, y, args.n_stat_mc, ctx.rng,
                             control_z=(how is None or BIN_DEFS[how][0] != "z"))
            fx = {}
            if fit and len(sub) >= 4:
                fd = sub[sub["resolved"]] if "fwhm" in (x, y) else sub
                xv, yv = fd[x].to_numpy(float), fd[y].to_numpy(float)
                xe = 0.5 * (fd[f"{x}_lo"] + fd[f"{x}_hi"]).to_numpy(float)
                ye = 0.5 * (fd[f"{y}_lo"] + fd[f"{y}_hi"]).to_numpy(float)
                if fit == "logy":
                    ok = yv > 0
                    xv, xe, ye, yv = xv[ok], xe[ok], ye[ok] / (yv[ok] * np.log(10)), np.log10(yv[ok])
                r = fit_line(xv, yv, xe, ye, args.n_boot, ctx.rng)
                if r is not None:
                    a, b, ae, be = r
                    fx = dict(fit=fit, fit_slope=a, fit_slope_err=ae, fit_int=b, fit_int_err=be)
                    gx = np.linspace(np.nanmin(d0[x]), np.nanmax(d0[x]), 100)
                    gy = a * gx + b if fit == "lin" else 10 ** (a * gx + b)
                    ax.plot(gx, gy, color="0.3", lw=1.0, ls="-.", zorder=2,
                            label=f"fit, slope {a:.2f} $\\pm$ {ae:.2f}")
            ctx.stat(f"{section}_{plot}{'_' + suffix if suffix else ''}", plain(lab) or "all",
                     x, y, st, **fx)
            ax.text(0.97, 0.03, stats_text(st), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.7))
            ttl = f"{lab}  (N = {len(sub)})" if lab else f"{ctx.title}  (N = {len(sub)})"
            ax.set_title(ttl, fontsize=8 if n > 1 else 9)
            ax.set_xlabel(Q[x][0])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, loc="upper left", fontsize=5.5)
        axes[0].set_ylabel(Q[y][0])
        if xlim:
            axes[0].set_xlim(*xlim)
        if ylim:
            axes[0].set_ylim(*ylim)
        if x == "muv":
            lo, hi = axes[0].get_xlim()
            axes[0].set_xlim(max(lo, hi), min(lo, hi))   # bright on the right
        if n > 1:
            fig.subplots_adjust(wspace=0.08)
            fig.suptitle(ctx.title, fontsize=9, y=1.03)
        if sc is not None:
            cb = fig.colorbar(sc, ax=list(axes), pad=0.02 if n == 1 else 0.01,
                              fraction=0.05 if n == 1 else 0.03)
            cb.set_label(Q[ckey][0])
        ctx.save(fig, section, plot + (f"_{suffix}" if suffix else ""))


def fixed_bins(values, width):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0, width])
    lo = np.floor(v.min() / width) * width
    hi = np.ceil(v.max() / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + 0.5 * width, width)


def hist_fig(ctx, df, key, bins_by_z=False):
    v = df[key].to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  [skip] hist_{key}: no values")
        return
    edges = fixed_bins(v, Q[key][1])
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.7))
    ax.hist(v, bins=edges, color=get_hist_colour(), edgecolor="black", lw=0.6)
    ax.axvline(np.median(v), color="k", ls="--", lw=0.8)
    ax.set_xlabel(Q[key][0])
    ax.set_ylabel("N")
    ax.set_title(f"{ctx.title}  (N = {v.size}, median {np.median(v):.3g})", fontsize=8)
    ctx.save(fig, "hist", key)
    if not bins_by_z:
        return
    for tag, edges_z in (("zbins", list(ctx.args.z_edges)), ("zbins2", [ctx.args.z_split])):
        if tag not in ctx.args.bins:
            continue
        d = finite(df, [key, "z"])
        masks, labels = bin_masks(d["z"].to_numpy(float), edges_z, "z")
        n = len(masks)
        fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.4), sharex=True, sharey=True)
        for ax, m, lab in zip(axes, masks, labels):
            vv = d.loc[m, key].to_numpy(float)
            ax.hist(vv, bins=edges, color=get_hist_colour(), edgecolor="black", lw=0.6)
            ax.set_title(f"{lab}  (N = {len(vv)})", fontsize=8)
            ax.set_xlabel(Q[key][0])
        axes[0].set_ylabel("N")
        fig.suptitle(ctx.title, fontsize=9, y=1.03)
        fig.subplots_adjust(wspace=0.08)
        ctx.save(fig, "hist", f"{key}_{tag}")
        for m, lab in zip(masks, labels):
            a = d.loc[m, key]
            ctx.stat(f"hist_{key}_{tag}", plain(lab), key, "", dict(N=len(a)),
                     median=float(a.median()) if len(a) else np.nan)


HIST_KEYS = ["fwhm", "dv", "z", "muv", "logM", "logsfr100", "logssfr100", "av", "aha",
             "logxion", "snr", "flux", "fblue_int", "fblue_data", "sep", "br", "dv_blue",
             "ha_fwhm", "vresid"]


def hist_overview(ctx, df):
    keys = [k for k in HIST_KEYS if np.isfinite(df[k].to_numpy(float)).any()]
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(MNRAS_PAGE_WIDTH_IN, 1.6 * nr))
    for ax, k in zip(axes.flat, keys):
        v = df[k].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=fixed_bins(v, Q[k][1]), color=get_hist_colour(), edgecolor="black",
                lw=0.4)
        ax.set_xlabel(Q[k][0], fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_title(f"N = {v.size}", fontsize=6.5)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle(ctx.title, fontsize=9)
    fig.tight_layout()
    ctx.save(fig, "hist", "overview")


# ---------------------------------------------------------------------------
# Literature
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

MASON18_SIGMA_DEX = 0.24
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
    muv = np.asarray(muv, dtype=float)
    x = muv + 20.0 + 0.26 * z
    gamma = np.where(muv >= -20.0 - 0.26 * z, -0.3, -0.7)
    out = 10.0 ** (0.32 * gamma * x + 2.34)
    return float(out) if out.ndim == 0 else out


def mason19_fwhm_median(muv, z):
    return (34.0 + mason18_dv(muv, z)) / 0.9


def density_factor(z):
    return ((1.0 + z) / 3.0) ** (2.0 / 3.0)


def lognormal_pdf_log(x, mu_log10, sigma_log10):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = x > 0
    lx = np.log10(x[ok])
    out[ok] = (np.exp(-0.5 * ((lx - mu_log10) / sigma_log10) ** 2)
               / (sigma_log10 * np.sqrt(2 * np.pi)))
    return out


def fit_lognormal_band(values, grid, n_boot, rng):
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


def draw_fwhm_models(ax, fwhm, z_med, muv, log_edges, xmax, n_boot, rng, density):
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
    ax.plot(grid, med * scale, color=COL_THIS, ls="--", lw=1.2, label="This work")
    info["this_median"] = 10 ** mu
    ax.plot(grid, lognormal_pdf_log(grid, PL25_MU, PL25_SIGMA) * scale,
            color=COL_PL, ls=":", lw=1.3, label="Prieto-Lyon+25")
    m18 = mason18_dv(muv, z_med)
    ax.plot(grid, lognormal_pdf_log(grid, np.log10(m18), MASON18_SIGMA_DEX) * scale,
            color=COL_MASON18, ls="-.", lw=1.2, label=r"Mason+18 (FWHM = $\Delta v$)")
    info["mason18_median"] = m18
    m19 = mason19_fwhm_median(muv, z_med)
    ax.plot(grid, lognormal_pdf_log(grid, np.log10(m19), MASON19_SIGMA_DEX) * scale,
            color=COL_MASON, ls="-.", lw=1.2, label="Mason+19 (V18 scaled)")
    info["mason19_median"] = m19
    if density:
        d_med = m19 * density_factor(z_med)
        ax.plot(grid, lognormal_pdf_log(grid, np.log10(d_med), MASON19_SIGMA_DEX) * scale,
                color=COL_MASON_DENS, ls="-.", lw=1.0, label=r"Mason+19 $\times\,[(1+z)/3]^{2/3}$")
        info["mason_density_median"] = d_med
    return info


def median_or(values, fallback):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else fallback


def fwhm_model_figs(ctx, df):
    """gal_fwhm_models, as before, all and split by z and M_UV."""
    a = ctx.args
    d = finite(df, ["fwhm", "z"])
    if len(d) < 3:
        return
    fwhm = d["fwhm"].to_numpy(float)
    xmax = a.fwhm_model_max
    lv = np.log10(fwhm[fwhm > 0])
    lb = a.fwhm_log_bin
    log_edges = np.arange(np.floor(lv.min() / lb) * lb, max(np.log10(xmax), lv.max()) + lb, lb)
    muv_all = a.muv if a.muv is not None else median_or(d["muv"], -20.0)
    rows = []
    z_med = float(np.nanmedian(d["z"]))
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.8))
    info = draw_fwhm_models(ax, fwhm, z_med, muv_all, log_edges, xmax, a.n_boot, ctx.rng,
                            a.density_scaling)
    ax.set_xlabel(Q["fwhm"][0])
    ax.set_ylabel("N")
    ax.set_xlim(0, xmax)
    ax.set_title(f"{ctx.title}  (N = {info['n']})", fontsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=6,
              title=rf"Mason at $M_{{\rm UV}}={muv_all:.2f}$, $z={z_med:.2f}$", title_fontsize=6)
    ctx.save(fig, "gal", "fwhm_models")
    rows.append(("all", z_med, muv_all, info))
    for how in ("zbins", "zbins2", "muvbins"):
        if how not in a.bins:
            continue
        masks, labels = split(d, how, a, ctx.ref if ctx.ref is not None else d)
        if masks is None:
            continue
        n = len(masks)
        fig, axes = plt.subplots(1, n, figsize=(panel_width(n), 2.6), sharex=True, sharey=True)
        for ax, m, lab in zip(axes, masks, labels):
            sub = d[m]
            zm = float(np.nanmedian(sub["z"])) if len(sub) else np.nan
            mu = a.muv if a.muv is not None else median_or(sub["muv"], muv_all)
            inf = (draw_fwhm_models(ax, sub["fwhm"].to_numpy(float), zm, mu, log_edges, xmax,
                                    a.n_boot, ctx.rng, a.density_scaling)
                   if len(sub) and np.isfinite(zm) else {"n": len(sub)})
            ax.set_title(f"{lab}\n(N = {inf['n']})", fontsize=8)
            ax.set_xlabel(Q["fwhm"][0])
            rows.append((f"[{how}] {plain(lab)}", zm, mu, inf))
        axes[0].set_ylabel("N")
        axes[0].set_xlim(0, xmax)
        for ax in axes[::-1]:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, loc="upper right", fontsize=5.5)
                break
        fig.suptitle(ctx.title, fontsize=9, y=1.10)
        fig.subplots_adjust(wspace=0.08)
        ctx.save(fig, "gal", f"fwhm_models_{how}")
    print(f"  [MODELS] median FWHM, Prieto-Lyon+25 {10**PL25_MU:.0f} km/s")
    for lab, zm, mu, info in rows:
        if info.get("n", 0) < 3:
            continue
        print(f"    {lab:<34s} N={info['n']:<3d} z_med={zm:.2f} M_UV={mu:.2f}  this work="
              f"{info['this_median']:.0f}  Mason+18={info['mason18_median']:.0f}  "
              f"Mason+19={info['mason19_median']:.0f} km/s")


def lit_verhamme(ax, d, d_all):
    lim = 1.05 * float(np.nanmax(np.concatenate([d_all["fwhm"] + d_all["fwhm_hi"],
                                                 d_all["dv"] + d_all["dv_hi"]])))
    f = np.linspace(0, lim, 200)
    v = VERH_SLOPE * f + VERH_INT
    sig = np.sqrt((VERH_SLOPE_ERR * f) ** 2 + VERH_INT_ERR ** 2 + VERH_SCATTER ** 2)
    ax.fill_between(f, v - sig, v + sig, color=COL_VERH, alpha=0.15, lw=0)
    ax.plot(f, v, color=COL_VERH, ls="--", lw=1.0, label="Verhamme+18")
    ax.plot(f, f, color="black", ls="--", lw=0.8, label="1:1")


def lit_mason_dv(ax, d, d_all):
    """Mason+18 at integer z across the set, or at the panel's median z with
    its scatter when the panel is a z bin."""
    m = d_all["muv"].to_numpy(float)
    grid = np.linspace(np.nanmin(m) - 0.3, np.nanmax(m) + 0.3, 200)
    cmap, norm = get_cmap(), colour_setup(d_all, "z")
    if len(d) and np.ptp(d["z"].to_numpy(float)) < 0.8 * np.ptp(d_all["z"].to_numpy(float)):
        zm = float(np.median(d["z"]))
        med = mason18_dv(grid, zm)
        ax.fill_between(grid, med * 10 ** -MASON18_SIGMA_DEX, med * 10 ** MASON18_SIGMA_DEX,
                        color=cmap(norm(zm)), alpha=0.15, lw=0, zorder=0)
        ax.plot(grid, med, color=cmap(norm(zm)), lw=1.1, label=rf"Mason+18, $z={zm:.2f}$")
        return
    for zz in np.arange(np.round(d_all["z"].min()), np.round(d_all["z"].max()) + 1):
        ax.plot(grid, mason18_dv(grid, zz), color=cmap(norm(zz)), lw=1.0,
                label=rf"Mason+18, $z={zz:g}$")


def lit_mason_fwhm(ax, d, d_all):
    m = d_all["muv"].to_numpy(float)
    grid = np.linspace(np.nanmin(m) - 0.3, np.nanmax(m) + 0.3, 200)
    cmap, norm = get_cmap(), colour_setup(d_all, "z")
    zm = float(np.median(d["z"])) if len(d) else float(np.median(d_all["z"]))
    ax.plot(grid, mason19_fwhm_median(grid, zm), color=cmap(norm(zm)), lw=1.1,
            label=rf"Mason+19, $z={zm:.2f}$")
    ax.axhline(10 ** PL25_MU, color=COL_PL, ls=":", lw=1.2, label="Prieto-Lyon+25")


def lit_density(ax, d, d_all):
    """(1+z)^(2/3) scaling (Prieto-Lyon et al. 2025) through the median."""
    z = d_all["z"].to_numpy(float)
    f = d_all["fwhm"].to_numpy(float)
    zg = np.linspace(np.nanmin(z), np.nanmax(z), 100)
    zm, fm = np.nanmedian(z), np.nanmedian(f)
    ax.plot(zg, fm * ((1 + zg) / (1 + zm)) ** (2 / 3), color=COL_PL, ls=":", lw=1.2,
            label=r"$(1+z)^{2/3}$ through the median")


def lit_zero_h(ax, d, d_all):
    ax.axhline(0, color="k", ls="--", lw=0.8)


# ---------------------------------------------------------------------------
# Special figures
# ---------------------------------------------------------------------------

def verhamme_resid_fig(ctx, df):
    d = finite(df, ["vresid"])
    if len(d) < 3:
        return
    v = d["vresid"].to_numpy(float)
    med = float(np.median(v))
    sig = float(1.4826 * np.median(np.abs(v - med)))
    boot = [np.median(ctx.rng.choice(v, v.size)) for _ in range(ctx.args.n_boot)]
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.7))
    ax.hist(v, bins=fixed_bins(v, Q["vresid"][1]), color=get_hist_colour(), edgecolor="black",
            lw=0.6)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvline(med, color=COL_VERH, lw=1.2,
               label=f"median {med:+.0f} $\\pm$ {np.std(boot):.0f}, $\\sigma$ {sig:.0f} km/s")
    ax.axvspan(-VERH_SCATTER, VERH_SCATTER, color=COL_VERH, alpha=0.1, lw=0,
               label="V18 scatter 72 km/s")
    ax.set_xlabel(Q["vresid"][0])
    ax.set_ylabel("N")
    ax.set_title(f"{ctx.title}  (N = {len(v)})", fontsize=9)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ctx.save(fig, "sys", "verhamme_resid")
    ctx.stat("sys_verhamme_resid", "all", "vresid", "", dict(N=len(v)), median=med,
             median_err=float(np.std(boot)), robust_sigma=sig)
    print(f"  Verhamme Eq. 2 residual  N={len(v)}  median {med:+.0f} +/- {np.std(boot):.0f}"
          f"  robust sigma {sig:.0f} km/s")
    for x in ("z", "muv"):
        scatter_fig(ctx, df, "sys", f"verhamme_resid_vs_{x}", x, "vresid",
                    bins=("model",) if x == "z" else ("zbins2",), extras=lit_zero_h)


def recovery_doubles_fig(ctx, df):
    d = finite(df[df["model"] == "double"], ["vresid", "vmid"])
    if len(d) < 3:
        print(f"  [skip] sys_recovery_doubles: {len(d)} doubles")
        return
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.7))
    lo = min(d["vresid"].min(), d["vmid"].min())
    hi = max(d["vresid"].max(), d["vmid"].max())
    edges = fixed_bins(np.array([lo, hi]), 50.0)
    for k, col, lab in (("vresid", COL_VERH, "FWHM relation (V18 Eq. 2)"),
                        ("vmid", "crimson", "midpoint of the peaks")):
        v = d[k].to_numpy(float)
        med = float(np.median(v))
        sig = float(1.4826 * np.median(np.abs(v - med)))
        ax.hist(v, bins=edges, histtype="step", color=col, lw=1.4,
                label=f"{lab}\nmedian {med:+.0f}, $\\sigma$ {sig:.0f} km/s")
        ctx.stat("sys_recovery_doubles", lab, k, "", dict(N=len(v)), median=med, robust_sigma=sig)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"systemic offset of the prediction [km s$^{-1}$]")
    ax.set_ylabel("N")
    ax.set_title(f"{ctx.title} doubles  (N = {len(d)})", fontsize=9)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ctx.save(fig, "sys", "recovery_doubles")


def double_fraction_fig(ctx, df):
    """Fraction of doubles in z bins, at the default rule (dBIC > 2 with the
    other tests of step 14b) and keeping only doubles with dBIC > 6 and > 10.
    One panel per z split."""
    a = ctx.args
    d = finite(df, ["z"])
    d = d[d["model"].isin(["single", "double"])]
    hows = [h for h in ("zbins", "zbins2") if h in a.bins]
    if len(d) < 4 or not hows:
        return
    fig, axes = plt.subplots(1, len(hows), figsize=(panel_width(len(hows)), 2.8), sharey=True,
                             squeeze=False)
    for ax, how in zip(axes[0], hows):
        masks, labels = split(d, how, a, d)
        for thr, col, off in ((None, "black", -0.05), (6.0, "0.45", 0.0), (10.0, "0.7", 0.05)):
            is_d = (d["model"] == "double") & ((d["dbic"] > thr) if thr is not None else True)
            xs, ps, los, his = [], [], [], []
            for m, lab in zip(masks, labels):
                k, n = int(is_d[m].sum()), int(m.sum())
                p, l_, h_ = wilson(k, n)
                xs.append(float(np.median(d.loc[m, "z"])) if n else np.nan)
                ps.append(p)
                los.append(p - l_)
                his.append(h_ - p)
                ctx.stat(f"igm_double_frac_vs_z_{how}", plain(lab), "z", "double", dict(N=n),
                         n_double=k, frac=p, frac_lo=l_, frac_hi=h_,
                         dbic_min=thr if thr is not None else 2.0)
            ax.errorbar(np.array(xs) + off, ps, yerr=[los, his], fmt="o", color=col, ms=4,
                        capsize=2, label="rule (dBIC > 2)" if thr is None else f"dBIC > {thr:g}")
        for m, lab, x_ in zip(masks, labels, xs):
            ax.text(x_, 0.97, f"N={int(m.sum())}", fontsize=6, ha="center", va="top")
        for e in (a.z_edges if how == "zbins" else [a.z_split]):
            ax.axvline(e, color="0.8", lw=0.6, zorder=0)
        ax.set_xlabel(Q["z"][0])
        ax.set_title(how, fontsize=8)
    axes[0][0].set_ylabel("double-peak fraction (lower limit)")
    axes[0][0].set_ylim(0, 1)
    axes[0][-1].legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.suptitle(f"{ctx.title}  (N = {len(d)})", fontsize=9, y=1.02)
    fig.subplots_adjust(wspace=0.08)
    ctx.save(fig, "igm", "double_frac_vs_z")


def category_fig(ctx, df, key, plot, label):
    d = finite(df, ["dv"])
    d = d[d[key].notna() & (d[key].astype(str) != "") & (d[key].astype(str) != "nan")]
    cats = sorted(d[key].astype(str).unique())
    if len(cats) < 2:
        print(f"  [skip] chk_{plot}: fewer than two {label} groups")
        return
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.8))
    rng = ctx.rng
    for i, c in enumerate(cats):
        v = d.loc[d[key].astype(str) == c, "dv"].to_numpy(float)
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=14, color=get_hist_colour(),
                   edgecolor="black", linewidths=0.4, zorder=3)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=1.5, zorder=4)
        ctx.stat(f"chk_{plot}", c, key, "dv", dict(N=int(v.size)), median=float(np.median(v)))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n(N = {int((d[key].astype(str) == c).sum())})" for c in cats])
    ax.set_xlabel(label)
    ax.set_ylabel(Q["dv"][0])
    txt = ""
    if len(cats) >= 2:
        groups = [d.loc[d[key].astype(str) == c, "dv"].to_numpy(float) for c in cats]
        groups = [g for g in groups if g.size >= 3]
        if len(groups) == 2:
            p = sps.ks_2samp(*groups).pvalue
            txt = f"KS p = {p:.2f}"
            ctx.stat(f"chk_{plot}", "KS", key, "dv", dict(N=len(d)), ks_p=float(p))
        elif len(groups) > 2:
            p = sps.kruskal(*groups).pvalue
            txt = f"Kruskal-Wallis p = {p:.2f}"
            ctx.stat(f"chk_{plot}", "Kruskal", key, "dv", dict(N=len(d)), kw_p=float(p))
    ax.set_title(f"{ctx.title}  {txt}", fontsize=8)
    ctx.save(fig, "chk", plot)


def det_vs_nondet(args, df_all, outdir, rule, pdf):
    """Selection and cumulative property distributions, detections against
    non-detections."""
    ctx = Ctx(args, outdir, rule, "det_vs_nondet", pdf)
    d = df_all[df_all["det"].isin([0, 1])]
    if d["det"].nunique() < 2:
        print("  [skip] det_vs_nondet: need both sets")
        return ctx
    for y in ("flux", "snr"):
        dd = finite(d, ["z", y])
        fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, 2.8))
        for det, mk, lab in ((1, "o", "detections"), (0, "v", "non-detections")):
            s = dd[dd["det"] == det]
            ax.scatter(s["z"], s[y], marker=mk, s=16, facecolors=get_hist_colour() if det else
                       "white", edgecolors="black", linewidths=0.5, label=f"{lab} ({len(s)})")
        if y == "snr":
            ax.axhline(pc.LYA_DET_SNR, color="k", ls="--", lw=0.8)
        ax.set_xlabel(Q["z"][0])
        ax.set_ylabel(Q[y][0])
        ax.set_yscale("symlog" if y == "flux" else "linear")
        ax.legend(frameon=False, fontsize=6)
        ax.set_title(ctx.title, fontsize=9)
        ctx.save(fig, "chk", f"{y}_vs_z")
    keys = ["z", "muv", "logM", "logsfr100", "logssfr100", "av", "logxion"]
    fig, axes = plt.subplots(2, 4, figsize=(MNRAS_PAGE_WIDTH_IN, 3.6))
    for ax, k in zip(axes.flat, keys):
        a_ = d.loc[d["det"] == 1, k].dropna().to_numpy(float)
        b_ = d.loc[d["det"] == 0, k].dropna().to_numpy(float)
        for v, col, lab in ((a_, "crimson", "det"), (b_, "0.3", "non-det")):
            if v.size:
                ax.step(np.sort(v), np.arange(1, v.size + 1) / v.size, where="post", color=col,
                        lw=1.2, label=f"{lab} ({v.size})")
        txt = ""
        if a_.size >= 3 and b_.size >= 3:
            ks = sps.ks_2samp(a_, b_).pvalue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ad = float(sps.anderson_ksamp([a_, b_]).significance_level)
            except Exception:
                ad = np.nan
            # scipy caps the AD p-value to 0.001 to 0.25
            adt = ("> 0.25" if ad >= 0.25 else "< 0.001" if ad <= 0.001 else f"= {ad:.3f}")
            txt = f"KS p = {ks:.2f}\nAD p {adt}"
            ctx.stat("chk_det_vs_nondet_cdfs", k, k, "", dict(N=int(a_.size + b_.size)),
                     n_det=int(a_.size), n_nondet=int(b_.size), ks_p=float(ks), ad_p=float(ad),
                     median_det=float(np.median(a_)), median_nondet=float(np.median(b_)))
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=6)
        ax.set_xlabel(Q[k][0], fontsize=7)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(frameon=False, fontsize=5.5, loc="lower right")
    axes.flat[-1].axis("off")
    fig.suptitle(f"{rule} detections against non-detections", fontsize=9)
    fig.tight_layout()
    ctx.save(fig, "chk", "det_vs_nondet_cdfs")
    return ctx


# ---------------------------------------------------------------------------
# One set
# ---------------------------------------------------------------------------

def is_double(d):
    return d["model"] == "double"


def run_set(ctx, df):
    a = ctx.args
    n_muv = int(df["muv"].notna().sum())
    print(f"\n[{ctx.name}] {len(df)} sources, {int(df['dv'].notna().sum())} with a Lya fit, "
          f"{n_muv} with a usable M_UV, {int(is_double(df).sum())} doubles, "
          f"{int(df['fblue_int'].notna().sum())} with f_blue")
    unres = df.loc[~df["resolved"] & df["fwhm"].notna(), "ID"].tolist()
    if unres:
        print(f"  unresolved (FWHM upper limit): {unres}")
    df = df[df["dv"].notna() & df["fwhm"].notna()].copy()
    if len(df) < 3:
        print("  [skip] fewer than three sources with a fit, no plots")
        return
    ctx.ref = df
    B_ALL = ("zbins", "zbins2", "muvbins", "massbins", "ssfrbins", "avbins", "model")
    B_Z = ("zbins", "zbins2", "model")

    # histograms
    for k in HIST_KEYS:
        hist_fig(ctx, df, k, bins_by_z=k in ("fwhm", "dv", "fblue_int"))
    hist_overview(ctx, df)

    # A. systemic
    scatter_fig(ctx, df, "sys", "dv_vs_fwhm", "fwhm", "dv", bins=B_ALL)
    scatter_fig(ctx, df, "sys", "dv_vs_fwhm_verhamme", "fwhm", "dv", bins=B_ALL,
                extras=lit_verhamme, fit="lin")
    verhamme_resid_fig(ctx, df)
    scatter_fig(ctx, df, "sys", "dvred_vs_halfsep", "halfsep", "dv", sample=is_double,
                one_to_one=True, fit="lin")
    recovery_doubles_fig(ctx, df)

    # B. radiative transfer
    for k in a.fblue_kinds:
        scatter_fig(ctx, df, "rt", f"fblue_{k}_vs_dv", "dv", f"fblue_{k}", bins=B_ALL)
    scatter_fig(ctx, df, "rt", "br_vs_sep", "sep", "br", sample=is_double)
    scatter_fig(ctx, df, "rt", "br_vs_dvred", "dv", "br", sample=is_double)
    scatter_fig(ctx, df, "rt", "sep_vs_z", "z", "sep", sample=is_double, ckey="muv"
                if df["muv"].notna().any() else "z")
    scatter_fig(ctx, df, "rt", "fwhm_vs_ha_fwhm", "ha_fwhm", "fwhm", bins=B_Z,
                one_to_one=True)
    scatter_fig(ctx, df, "rt", "dv_vs_ha_fwhm", "ha_fwhm", "dv", bins=B_Z)

    # C. IGM and redshift
    ck = "muv" if df["muv"].notna().any() else "snr"
    scatter_fig(ctx, df, "igm", "dv_vs_z", "z", "dv", ckey=ck,
                bins=("muvbins", "massbins", "model"))
    for k in a.fblue_kinds:
        scatter_fig(ctx, df, "igm", f"fblue_{k}_vs_z", "z", f"fblue_{k}", ckey=ck,
                    bins=("muvbins", "massbins", "model"))
    double_fraction_fig(ctx, df)
    d_z = df.assign(log1pz=np.log10(1 + df["z"]))
    scatter_fig(ctx, df, "igm", "fwhm_vs_z", "z", "fwhm", ckey=ck,
                bins=("muvbins", "massbins", "model"), extras=lit_density)
    # the power-law index of FWHM against (1+z), for the (1+z)^(2/3) test
    fd = finite(d_z[d_z["resolved"]], ["fwhm", "log1pz"])
    if len(fd) >= 4:
        r = fit_line(fd["log1pz"].to_numpy(float), np.log10(fd["fwhm"].to_numpy(float)),
                     np.zeros(len(fd)),
                     (0.5 * (fd["fwhm_lo"] + fd["fwhm_hi"]) / (fd["fwhm"] * np.log(10))
                      ).to_numpy(float), a.n_boot, ctx.rng)
        if r is not None:
            print(f"  FWHM ~ (1+z)^n, n = {r[0]:.2f} +/- {r[2]:.2f} (density scaling 2/3)")
            ctx.stat("igm_fwhm_vs_z", "power law", "log(1+z)", "log fwhm", dict(N=len(fd)),
                     fit="loglog", fit_slope=r[0], fit_slope_err=r[2], fit_int=r[1],
                     fit_int_err=r[3])

    # D. galaxy properties
    if n_muv:
        scatter_fig(ctx, df, "gal", "dv_vs_muv", "muv", "dv",
                    bins=("zbins", "zbins2", "massbins", "model"), extras=lit_mason_dv,
                    fit="logy")
        scatter_fig(ctx, df, "gal", "fwhm_vs_muv", "muv", "fwhm",
                    bins=("zbins", "zbins2", "massbins", "model"), extras=lit_mason_fwhm)
        scatter_fig(ctx, df, "gal", "fblue_int_vs_muv", "muv", "fblue_int",
                    bins=("zbins2", "model"))
    else:
        print("  [skip] M_UV figures, no usable M_UV")
    fwhm_model_figs(ctx, df)
    for p in GAL_PROPS:
        if not df[p].notna().any():
            print(f"  [skip] gal_*_vs_{p}: no {p}")
            continue
        for q in ("dv", "fwhm", "fblue_int"):
            scatter_fig(ctx, df, "gal", f"{q}_vs_{p}", p, q,
                        bins=("zbins", "zbins2", "muvbins", "model"))

    # F. checks
    category_fig(ctx, df, "zline", "dv_by_zline", "z_sys line")
    category_fig(ctx, df, "grating", "dv_by_grating", "z_sys grating")
    scatter_fig(ctx, df, "chk", "dv_single_vs_double", "dv_single", "dv", sample=is_double,
                one_to_one=True)
    scatter_fig(ctx, df, "chk", "dv_vs_snr", "snr", "dv", bins=("zbins2",))
    scatter_fig(ctx, df, "chk", "fwhm_vs_snr", "snr", "fwhm", bins=("zbins2",))
    scatter_fig(ctx, df, "chk", "fblue_int_vs_data", "fblue_data", "fblue_int", one_to_one=True)


def write_stats(ctx):
    if not ctx.rows:
        return
    path = os.path.join(ctx.outdir, "stats.csv")
    st = pd.DataFrame(ctx.rows)
    lead = [c for c in ("plot", "panel", "x", "y", "N", "rho", "p_rho", "rho_p16", "rho_p84",
                        "tau", "p_tau", "rho_partial_z", "p_partial_z", "fit", "fit_slope",
                        "fit_slope_err", "fit_int", "fit_int_err") if c in st.columns]
    st = st[lead + [c for c in st.columns if c not in lead]]
    st.to_csv(path, index=False)
    print(f"Wrote  {path}")
    sig = st[(st.get("p_rho", pd.Series(dtype=float)) < 0.05) & (st["panel"] == "all")] \
        if "p_rho" in st.columns else st.iloc[0:0]
    if len(sig):
        print(f"  unbinned panels with Spearman p < 0.05 ({len(sig)}):")
        for _, r in sig.sort_values("p_rho").iterrows():
            part = (f"  partial(z) {r['rho_partial_z']:+.2f} p={r['p_partial_z']:.3f}"
                    if np.isfinite(r.get("rho_partial_z", np.nan)) else "")
            print(f"    {r['plot']:<34s} N={int(r['N']):<3d} rho {r['rho']:+.2f} "
                  f"p={r['p_rho']:.3f}{part}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", choices=list(SETS), default=["detections"],
                   help="which sets to plot (default detections)")
    p.add_argument("--catalog", default=LYA_CATALOG)
    p.add_argument("--outdir", default=f"{pc.P2_ROOT}/plots/lya_science")
    p.add_argument("--bins", nargs="+", choices=list(BIN_DEFS), default=list(BIN_DEFS),
                   help="which binned versions to make (default all)")
    p.add_argument("--fwhm-kind", choices=["int", "obs"], default="int")
    p.add_argument("--fblue-kinds", nargs="+", choices=["int", "conv", "data"],
                   default=["int", "data"])
    p.add_argument("--resolve-frac", type=float, default=0.5)
    p.add_argument("--muv-source", choices=list(MUV_SOURCES), default="sed")
    p.add_argument("--muv-max-err", type=float, default=0.5)
    p.add_argument("--z-edges", type=float, nargs=2, default=[3.5, 4.5])
    p.add_argument("--z-split", type=float, default=3.5)
    p.add_argument("--muv", type=float, default=None,
                   help="fix the M_UV of the Mason curves in the FWHM model figures")
    p.add_argument("--fwhm-model-max", type=float, default=800.0)
    p.add_argument("--fwhm-log-bin", type=float, default=0.1)
    p.add_argument("--n-boot", type=int, default=500,
                   help="bootstrap resamples for fits, medians and the FWHM model bands")
    p.add_argument("--n-stat-mc", type=int, default=1000,
                   help="redraws within the errors for the range of Spearman rho")
    p.add_argument("--density-scaling", action="store_true")
    p.add_argument("--no-pdf", action="store_true", help="skip all_figures.pdf")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_fwhm_kind(args.fwhm_kind)
    print("plot_lya_science.py")
    print(f"  reading  {os.path.abspath(args.catalog)}")
    rule = pc.det_rule(pd.read_csv(args.catalog))
    root = os.path.abspath(args.outdir)
    mcol = MUV_SOURCES[args.muv_source][0]
    print(f"  detection rule {rule}, in_analysis == 1 only")
    print(f"  FWHM {args.fwhm_kind}, M_UV from {mcol} where its error < {args.muv_max_err:g}, "
          f"f_blue kinds {args.fblue_kinds}")
    print(f"  splits {args.bins}")

    df = load_sample(args.catalog, args.fwhm_kind, args.resolve_frac, args.muv_max_err,
                     args.muv_source)
    untested = df.loc[df["det"].isna(), "ID"].tolist()
    if untested:
        print(f"  not tested, in neither set: {untested}")
    for st in ("close_pair", "interloper"):
        ids = df.loc[(df["in_analysis"] == 0) & (df["pair_status"] == st), "ID"].tolist()
        if ids:
            print(f"  left out ({st}): {ids}")
    df = df[df["in_analysis"] == 1]

    written = []
    for name in args.sets:
        outdir = os.path.join(root, f"{rule}_{name}")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n  writing  {outdir}/")
        pdf = None if args.no_pdf else PdfPages(os.path.join(outdir, "all_figures.pdf"))
        ctx = Ctx(args, outdir, rule, name, pdf)
        run_set(ctx, df[df["det"] == SETS[name]].copy())
        write_stats(ctx)
        if pdf is not None:
            pdf.close()
            print(f"Wrote  {os.path.join(outdir, 'all_figures.pdf')}")
        written.append((outdir, ctx.n_saved))

    outdir = os.path.join(root, f"{rule}_det_vs_nondet")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n  writing  {outdir}/")
    pdf = None if args.no_pdf else PdfPages(os.path.join(outdir, "all_figures.pdf"))
    ctx = det_vs_nondet(args, df, outdir, rule, pdf)
    write_stats(ctx)
    if pdf is not None:
        pdf.close()
    written.append((outdir, ctx.n_saved))

    print("\n[DONE]")
    for d, n in written:
        print(f"  {n:>4d} figures in {d}/")


if __name__ == "__main__":
    main()