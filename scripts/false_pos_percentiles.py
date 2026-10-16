#!/usr/bin/env python
"""
false_pos_percentiles.py

Detection thresholds for the Lya S/N from the false positions (Patrick et
al., Paper 1, Sect. 3.3.3).

The peak S/N of every false position (optimize_lya_positions_f_grating.py)
from every source used for false positives is pooled. Its 90th, 95th, 98th
and 99.5th percentiles (--percentiles) and the percentile that each fixed
S/N in --snr-values (default 5) sits at are written out, for
build_lya_catalog.py to use.

The detection threshold drawn and used to split the real sources in the
plots is S/N = 5 by default (--det-snr, pc.LYA_DET_SNR), or a percentile if
--det-percentile is given. Per source, the same threshold measured on that
source's own false positions (its percentile at the fixed S/N, or its own
percentile value) is plotted against the Lya wavelength. The sample spans
4750 to 8500 AA and the sky is much worse in the red, so this shows whether
one threshold is equally strict at every redshift.

Sources left out of the false positives (no band coverage, e.g. the AO gap)
get status "no_coverage".

Inputs
------
  MUSE_catalogs/lya_false_pos_sources.csv           sources and their Lya wavelength
  MUSE_subcubes/<ID>/<ID>_false_optimal_snr.csv     peak_snr per false position
  MUSE_catalogs/lya_fits.csv                        lya_snr of the real sources

Outputs
-------
  MUSE_catalogs/lya_false_pos_thresholds.csv
      n_sources, n_positions, median, snr5_percentile (one per --snr-values),
      p90, p95, p98, p99_5
  MUSE_catalogs/lya_false_pos_per_source.csv     the same for each source
  MUSE_catalogs/lya_detections.csv               ID, status, lya_snr
  plots/lya_false_pos_hist.png
      false positions (filled, bins of 0.25) and real sources (outlines,
      bins of 1, split at the detection threshold), with that threshold
      (dashed, labelled with its percentile) and the 99.5th percentile
      (dotted, Paper 1) marked, as Paper 1 Fig. 5. --snr-line adds a line at
      another S/N. The S/N axis is linear to 20 (--xbreak) and logarithmic
      above it, shaded and with break marks
  plots/lya_false_pos_vs_wavelength.png
      each source's own threshold against its Lya wavelength

Usage
-----
python false_pos_percentiles.py                       # S/N = 5 drawn as the threshold
python false_pos_percentiles.py --det-percentile 95   # the 95th instead
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

import p2_common as pc

try:
    import cmasher as cmr
    _HAVE_CMASHER = True
except ImportError:
    _HAVE_CMASHER = False

P2 = pc.P2_ROOT
CAT = pc.MUSE_CAT_DIR

# MNRAS single column, as the Paper 1 figure
MNRAS_COL_WIDTH_IN = 3.46
MNRAS_FIG_HEIGHT_IN = 2.70
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})
THEME = "torch"


def make_palette(name, n=8):
    """Palette from a cmasher colourmap, as the Paper 1 plot (viridis if
    cmasher is missing)."""
    if _HAVE_CMASHER:
        hexes = cmr.take_cmap_colors(f"cmr.{name}", n, cmap_range=(0.10, 0.90),
                                     return_fmt="hex")
    else:
        cmap = plt.get_cmap("viridis")
        hexes = [mcolors.to_hex(cmap(x)) for x in np.linspace(0.10, 0.90, n)]
    pal = {f"c{i}": h for i, h in enumerate(hexes)}
    pal["near_black"] = mcolors.to_hex((0.10, 0.10, 0.12, 1.0))
    pal["neutral_grey"] = "#aab4c8"
    return pal


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=pc.FALSE_POS_SOURCES)
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits.csv")
    p.add_argument("--offsets-csv", default=f"{CAT}/lya_optimal_offsets.csv",
                   help="real-source offsets, to compare blocked offsets")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 98, 99.5],
                   help="thresholds written for build_lya_catalog.py")
    p.add_argument("--det-snr", type=float, default=pc.LYA_DET_SNR,
                   help="fixed S/N drawn as the detection line and used to split the "
                        "real sources (default 5)")
    p.add_argument("--det-percentile", type=float, default=None,
                   help="use this percentile as the detection line instead of --det-snr")
    p.add_argument("--snr-values", type=float, nargs="+", default=[5.0],
                   help="fixed S/N values whose percentile is written out (default 5)")
    p.add_argument("--snr-line", type=float, default=None,
                   help="also draw a line at this S/N, labelled with its percentile")
    p.add_argument("--out-thresholds", default=f"{CAT}/lya_false_pos_thresholds.csv")
    p.add_argument("--out-per-source", default=f"{CAT}/lya_false_pos_per_source.csv")
    p.add_argument("--out-detections", default=f"{CAT}/lya_detections.csv")
    p.add_argument("--plot-hist", default=f"{P2}/plots/lya_false_pos_hist.png")
    p.add_argument("--plot-wave", default=f"{P2}/plots/lya_false_pos_vs_wavelength.png")
    p.add_argument("--bin-width", type=float, default=0.25,
                   help="false-position histogram bin width in S/N (default 0.25)")
    p.add_argument("--xbreak", type=float, default=20.0,
                   help="S/N above which the histogram axis is compressed (log), default 20")
    p.add_argument("--compress-frac", type=float, default=0.25,
                   help="width of the compressed part as a fraction of the linear part")
    p.add_argument("--high-bins", type=int, default=5,
                   help="number of bins above --xbreak, equal in log S/N (default 5)")
    p.add_argument("--real-bin-width", type=float, default=1.0,
                   help="real-source histogram bin width (default 1, wider because "
                        "the real sources are few and spread to high S/N)")
    return p.parse_args()


def pname(pc_):
    """Column suffix for a percentile, 98 -> 98, 99.5 -> 99_5."""
    return f"{pc_:g}".replace(".", "_")


def compressed_scale(xbreak, xmax, frac):
    """Forward and inverse of an axis that is linear up to xbreak and
    logarithmic above it, with the log part taking frac x xbreak of the axis."""
    if xmax <= xbreak * 1.001:
        return None
    k = frac * xbreak / np.log(xmax / xbreak)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= xbreak, x, xbreak + k * np.log(np.maximum(x, xbreak) / xbreak))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= xbreak, y, xbreak * np.exp((np.maximum(y, xbreak) - xbreak) / k))
    return fwd, inv


def plot_hist(path, pooled, real_above, real_below, dl, thr, snr_line, bin_width,
              real_bin_width, xbreak, frac, n_high_bins):
    """dl: the detection line, a dict with the threshold 'cut' and the TeX
    labels 'line', 'above' and 'below'."""
    cut = dl["cut"]
    pal = make_palette(THEME)
    allv = np.concatenate([pooled, real_above, real_below])
    lo = min(0.0, np.floor(pooled.min()))
    vmax = float(np.nanmax(allv))
    scale_fns = compressed_scale(xbreak, vmax * 1.05, frac)
    top = xbreak if scale_fns else np.ceil(vmax / bin_width) * bin_width + bin_width

    # Linear bins up to the break, then n_high_bins equal in log S/N above it
    high = (np.geomspace(xbreak, vmax * 1.05, n_high_bins + 1)[1:] if scale_fns
            else np.array([]))
    edges = np.concatenate([np.arange(lo, top + 0.5 * bin_width, bin_width), high])
    redges = cut + real_bin_width * np.arange(np.floor((lo - cut) / real_bin_width),
                                              np.floor((top - cut) / real_bin_width) + 1)
    redges = np.unique(np.concatenate([redges[redges <= top], [top], high]))

    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, MNRAS_FIG_HEIGHT_IN), dpi=300)
    if scale_fns:
        ax.set_xscale("function", functions=scale_fns)
    c, _ = np.histogram(pooled, bins=edges)
    w = np.ones(pooled.size) / (c.max() if c.max() > 0 else 1.0)
    ax.hist(pooled, bins=edges, weights=w, histtype="stepfilled",
            color=pal["c1"], alpha=0.25, lw=0, label="False positives")
    ax.hist(pooled, bins=edges, weights=w, histtype="step", color=pal["c1"], alpha=0.6, lw=0.9)
    real = np.concatenate([real_above, real_below])
    if real.size:
        # real-source bins aligned on the S/N cut so no bin straddles it
        c, _ = np.histogram(real, bins=redges)
        scale = 1.0 / c.max() if c.max() > 0 else 1.0
        for vals, col, lab in ((real_below, "tab:purple", dl["below"]),
                               (real_above, "tab:red", dl["above"])):
            if vals.size:
                ax.hist(vals, bins=redges, weights=np.full(vals.size, scale), histtype="step",
                        color=col, lw=1.1, label=lab)
    ax.axvline(cut, color=pal["near_black"], lw=1.6, ls="--", label=dl["line"])
    if 99.5 in thr and not np.isclose(cut, thr[99.5]):
        ax.axvline(thr[99.5], color=pal["neutral_grey"], lw=1.2, ls=":",
                   label=rf"$\Sigma^{{99.5}} = {thr[99.5]:.2f}$ (Paper 1)")
    if snr_line is not None:
        sl_pct = 100.0 * float(np.mean(pooled < snr_line))
        ax.axvline(snr_line, color=pal["c5"], lw=1.0, ls="-.",
                   label=rf"$\Sigma = {snr_line:g}$ ({sl_pct:.2f}th percentile)")

    xr = edges[-1]
    ax.set_xlim(edges[0], xr)
    if scale_fns:
        # shade the compressed part and put break marks on both spines
        ax.axvspan(xbreak, xr, color="0.93", lw=0, zorder=0)
        y_b = [-0.025, 1.0 - 0.025]
        for yb in y_b:
            for dx in (-0.012, 0.0):
                xb_ax = (scale_fns[0](xbreak) - scale_fns[0](edges[0])) / \
                        (scale_fns[0](xr) - scale_fns[0](edges[0])) + dx
                ax.plot([xb_ax - 0.008, xb_ax + 0.008], [yb, yb + 0.05], transform=ax.transAxes,
                        color="k", lw=0.8, clip_on=False, zorder=5)
        # ticks in the compressed part, kept far enough apart to read
        fwd = scale_fns[0]
        zone = float(fwd(xr) - fwd(xbreak))
        hi_ticks, last = [], float(fwd(xbreak))
        for t in [30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000]:
            if t <= xr and float(fwd(t)) - last >= 0.45 * zone:
                hi_ticks.append(t)
                last = float(fwd(t))
        lo_ticks = list(np.arange(0, xbreak + 0.1, 5))
        ax.set_xticks(lo_ticks + hi_ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"Peak $\Sigma_{\rm Ly\alpha}$", fontsize=10)
    ax.set_ylabel("Normalised counts", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wave(path, per, dl):
    """Each source's own threshold against its Lya wavelength. dl['col'] is
    the per-source column, dl['pooled'] the pooled value."""
    pal = make_palette(THEME)
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN * 1.4, MNRAS_FIG_HEIGHT_IN), dpi=300)
    ax.scatter(per["lya_obs"], per[dl["col"]], s=10, color=pal["c2"], label="each source")
    ax.axhline(dl["pooled"], color=pal["near_black"], ls="--", lw=1,
               label=f"pooled = {dl['pooled']:.2f}")
    ax.set_xlabel(r"Ly$\alpha$ wavelength (air) [$\AA$]")
    ax.set_ylabel(dl["ylabel"])
    sec = ax.secondary_xaxis("top", functions=(lambda w: w / pc.LYA_REST - 1,
                                               lambda z: (1 + z) * pc.LYA_REST))
    sec.set_xlabel("z", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    pcts = sorted(set(a.percentiles) | ({a.det_percentile} if a.det_percentile else set()))
    svals = sorted(set(a.snr_values) | ({a.det_snr} if a.det_percentile is None else set()))
    print("false_pos_percentiles.py")
    man_path = os.path.abspath(a.manifest)
    print(f"Reading manifest    {man_path}")
    man = pd.read_csv(man_path)
    man["ID"] = man["ID"].astype(int)

    pooled, per, blocked = [], [], []
    for _, m in man[man["used"] == 1].iterrows():
        sid = int(m["ID"])
        path = pc.muse_path(sid, "false_snr", a.subcube_dir)
        if not os.path.exists(path):
            print(f"  WARNING no false-position S/N for {sid} ({path})")
            continue
        fd = pd.read_csv(path)
        if "n_blocked" in fd:
            blocked.append(fd["n_blocked"].to_numpy(float))
        vals = fd["peak_snr"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        pooled.append(vals)
        rec = dict(ID=sid, lya_obs=m["lya_obs"], n=len(vals),
                   median=round(float(np.median(vals)), 3))
        rec.update({f"p{pname(p_)}": round(float(np.percentile(vals, p_)), 3) for p_ in pcts})
        rec.update({f"snr{pname(v)}_percentile": round(100.0 * float(np.mean(vals < v)), 3)
                    for v in svals})
        per.append(rec)
    if not pooled:
        raise SystemExit("No false-position S/N files found, run "
                         "optimize_lya_positions_f_grating.py first")
    pooled = np.concatenate(pooled)
    per = pd.DataFrame(per)
    thr = {p_: float(np.percentile(pooled, p_)) for p_ in pcts}
    spct = {v: 100.0 * float(np.mean(pooled < v)) for v in svals}

    print(f"\n{len(per)} sources, {pooled.size} false positions")
    print(f"  median {np.median(pooled):.2f}  (Paper 1: 2.7)")
    for p_ in pcts:
        print(f"  {p_:>5g}th percentile  {thr[p_]:.2f}")
    for v in svals:
        print(f"  S/N = {v:g} is the {spct[v]:.2f}th percentile")
    if a.det_percentile is None:
        cut = a.det_snr
        dl = dict(cut=cut, col=f"snr{pname(cut)}_percentile", pooled=spct[cut],
                  ylabel=rf"Percentile of $\Sigma = {cut:g}$",
                  line=rf"$\Sigma = {cut:g}$ ({spct[cut]:.2f}th percentile)",
                  above=rf"Ly$\alpha$ $\Sigma \geq {cut:g}$",
                  below=rf"Ly$\alpha$ $\Sigma < {cut:g}$")
        print(f"  each source's percentile at S/N = {cut:g}: {per[dl['col']].min():.2f} to "
              f"{per[dl['col']].max():.2f}, median {per[dl['col']].median():.2f}")
    else:
        p_ = a.det_percentile
        cut = thr[p_]
        dl = dict(cut=cut, col=f"p{pname(p_)}", pooled=cut,
                  ylabel=rf"False-position $\Sigma^{{{p_:g}}}$",
                  line=rf"$\Sigma^{{{p_:g}}} = {cut:.2f}$",
                  above=rf"Ly$\alpha$ $\Sigma > \Sigma^{{{p_:g}}}$",
                  below=rf"Ly$\alpha$ $\Sigma \leq \Sigma^{{{p_:g}}}$")
        print(f"  each source's {p_:g}th percentile: {per[dl['col']].min():.2f} to "
              f"{per[dl['col']].max():.2f}, median {per[dl['col']].median():.2f}")

    # Search freedom: offsets blocked by Isaac objects, false vs real
    if blocked and os.path.exists(a.offsets_csv):
        fb = np.concatenate(blocked)
        rb = pd.read_csv(a.offsets_csv)["n_blocked"].to_numpy(float)
        print("\nOffsets blocked by Isaac objects (of 81), false positions vs real sources")
        print(f"  false  mean {fb.mean():.1f}, median {np.median(fb):.0f}, "
              f"any blocked {np.mean(fb > 0):.0%}")
        print(f"  real   mean {rb.mean():.1f}, median {np.median(rb):.0f}, "
              f"any blocked {np.mean(rb > 0):.0%}")

    # Real sources
    fits_path = os.path.abspath(a.fits_csv)
    print(f"\nReading lya_snr     {fits_path}")
    fitsdf = pd.read_csv(fits_path)[["ID", "lya_snr"]]
    fitsdf["ID"] = fitsdf["ID"].astype(int)
    det = man[["ID", "used", "reason"]].merge(fitsdf, on="ID", how="left")
    det["status"] = np.where(det["used"] == 1, "ok",
                             np.where(det["reason"] == "band coverage", "no_coverage",
                                      det["reason"].fillna("missing")))
    ok = det["status"] == "ok"
    det = det[["ID", "status", "lya_snr"]].sort_values("ID")
    for v in svals:
        print(f"  S/N >= {v:<6g}     {int((ok & (det['lya_snr'].round(2) >= v)).sum())} of "
              f"{int(ok.sum())} tested sources")
    for p_ in pcts:
        print(f"  above the {p_:>5g}th  {int((ok & (det['lya_snr'] > thr[p_])).sum())} of "
              f"{int(ok.sum())} tested sources")
    nc = det.loc[det["status"] != "ok", "ID"].tolist()
    if nc:
        print(f"  not usable       {nc}")

    # Write
    row = dict(n_sources=len(per), n_positions=pooled.size,
               median=round(float(np.median(pooled)), 3))
    row.update({f"snr{pname(v)}_percentile": round(spct[v], 3) for v in svals})
    row.update({f"p{pname(p_)}": round(thr[p_], 3) for p_ in pcts})
    outs = [(a.out_thresholds, pd.DataFrame([row])), (a.out_per_source, per),
            (a.out_detections, det)]
    for path, d in outs:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d.to_csv(path, index=False)
    for path in (a.plot_hist, a.plot_wave):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if a.det_percentile is None:
        is_above = det["lya_snr"].round(2) >= dl["cut"]   # as build_lya_catalog.py
    else:
        is_above = det["lya_snr"] > dl["cut"]
    above = det.loc[ok & is_above, "lya_snr"].to_numpy(float)
    below = det.loc[ok & ~is_above, "lya_snr"].to_numpy(float)
    plot_hist(a.plot_hist, pooled, above, below, dl, thr, a.snr_line, a.bin_width,
              a.real_bin_width, a.xbreak, a.compress_frac, a.high_bins)
    plot_wave(a.plot_wave, per, dl)

    print("\nWrote")
    for path in [o[0] for o in outs] + [a.plot_hist, a.plot_wave]:
        print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()