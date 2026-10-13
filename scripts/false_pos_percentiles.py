#!/usr/bin/env python
"""
false_pos_percentiles.py

Turn the false-position S/N into detection thresholds for the Lya S/N, and
flag every source as a detection or not at each threshold (Patrick et al.,
Paper 1, Sect. 3.3.3).

The peak S/N of every false position (optimize_lya_positions_f_grating.py)
from every source used for false positives is pooled, and the 90th, 95th,
98th and 99.5th percentiles (--percentiles) are the thresholds. Paper 1 used
the 98th for detections and the 99.5th for strong detections. A source is
detected at a level when its lya_snr exceeds that threshold.

The same percentiles are also measured for each source's own false positions
(the local thresholds), and plotted against the Lya wavelength. The sample
spans 4750 to 8500 AA and the sky is much worse in the red, so this shows
whether one pooled threshold is fair for every redshift. det_local_98 flags
detections against each source's own 98th percentile for comparison.

Sources left out of the false positives (no band coverage, e.g. the AO gap)
get status "no_coverage" and no flags, since they could not have been
detected.

Inputs
------
  MUSE_catalogs/lya_false_pos_sources.csv           sources and their Lya wavelength
  MUSE_subcubes/<ID>/<ID>_false_optimal_snr.csv     peak_snr per false position
  MUSE_catalogs/lya_fits.csv                        lya_snr of the real sources

Outputs
-------
  MUSE_catalogs/lya_false_pos_thresholds.csv     pooled n, median and percentiles
  MUSE_catalogs/lya_false_pos_per_source.csv     the same for each source
  MUSE_catalogs/lya_detections.csv
      ID, status, lya_snr, det_90, det_95, det_98, det_99_5 (pooled),
      local_p98, det_local_98
  plots/lya_false_pos_hist.png
      false positives (filled) and real sources (outlines, detections at the
      98th percentile in red), as Paper 1 Fig. 5
  plots/lya_false_pos_vs_wavelength.png
      each source's 98th and 95th percentile against its Lya wavelength, with
      the pooled values

Usage
-----
python false_pos_percentiles.py
python false_pos_percentiles.py --percentiles 90 95 98 99.5
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
    p.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 98, 99.5])
    p.add_argument("--out-thresholds", default=f"{CAT}/lya_false_pos_thresholds.csv")
    p.add_argument("--out-per-source", default=f"{CAT}/lya_false_pos_per_source.csv")
    p.add_argument("--out-detections", default=f"{CAT}/lya_detections.csv")
    p.add_argument("--plot-hist", default=f"{P2}/plots/lya_false_pos_hist.png")
    p.add_argument("--plot-wave", default=f"{P2}/plots/lya_false_pos_vs_wavelength.png")
    p.add_argument("--xmax", type=float, default=20.0, help="histogram S/N limit")
    p.add_argument("--bins", type=int, default=80)
    return p.parse_args()


def pname(pc_):
    """Column suffix for a percentile, 98 -> 98, 99.5 -> 99_5."""
    return f"{pc_:g}".replace(".", "_")


def plot_hist(path, pooled, real_det, real_non, thr, pcts, xmax, bins):
    pal = make_palette(THEME)
    edges = np.linspace(min(0.0, pooled.min()), xmax, bins + 1)

    def peak_norm(vals):
        c, _ = np.histogram(vals, bins=edges)
        return np.ones(len(vals)) / (c.max() if c.max() > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN, MNRAS_FIG_HEIGHT_IN), dpi=300)
    ax.hist(pooled, bins=edges, weights=peak_norm(pooled), histtype="stepfilled",
            color=pal["c1"], alpha=0.25, lw=0, label="False positives")
    ax.hist(pooled, bins=edges, weights=peak_norm(pooled), histtype="step",
            color=pal["c1"], alpha=0.6, lw=0.9)
    real = np.concatenate([real_det, real_non])
    if real.size:
        # Real sources normalised to the peak of their combined histogram,
        # values above xmax drawn in the last bin.
        c, _ = np.histogram(np.clip(real, None, xmax - 1e-6), bins=edges)
        scale = 1.0 / c.max() if c.max() > 0 else 1.0
        for vals, col, lab in ((real_non, "tab:purple", r"Ly$\alpha$ non-detect."),
                               (real_det, "tab:red", r"Ly$\alpha$ detect.")):
            if vals.size:
                ax.hist(np.clip(vals, None, xmax - 1e-6), bins=edges,
                        weights=np.full(vals.size, scale), histtype="step", color=col,
                        lw=1.1, label=lab)
    # Paper 1 style for the 98th (dashed, black) and 99.5th (dotted, grey)
    styles = {90: (":", pal["c5"], 1.0), 95: ("-.", pal["c3"], 1.0),
              98: ("--", pal["near_black"], 1.6), 99.5: (":", pal["neutral_grey"], 1.2)}
    for p_ in pcts:
        ls, col, lw = styles.get(p_, ("--", pal["c6"], 1.0))
        ax.axvline(thr[p_], color=col, lw=lw, ls=ls, label=rf"$\Sigma^{{{p_:g}}}={thr[p_]:.2f}$")
    n_over = int((real > xmax).sum())
    if n_over:
        ax.text(0.98, 0.55, f"{n_over} sources\nabove {xmax:g}", transform=ax.transAxes,
                ha="right", fontsize=6)
    ax.set_xlabel(r"Peak $\Sigma_{\rm Ly\alpha}$", fontsize=10)
    ax.set_ylabel("Normalised counts", fontsize=10)
    ax.set_xlim(edges[0], xmax)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wave(path, per, thr):
    pal = make_palette(THEME)
    fig, ax = plt.subplots(figsize=(MNRAS_COL_WIDTH_IN * 1.4, MNRAS_FIG_HEIGHT_IN), dpi=300)
    ax.scatter(per["lya_obs"], per["p98"], s=10, color=pal["c2"], label="source 98th")
    if "p95" in per:
        ax.scatter(per["lya_obs"], per["p95"], s=8, marker="s", facecolor="none",
                   edgecolor=pal["c5"], lw=0.7, label="source 95th")
    ax.axhline(thr[98], color=pal["near_black"], ls="--", lw=1,
               label=rf"pooled 98th = {thr[98]:.2f}")
    if 95 in thr:
        ax.axhline(thr[95], color=pal["neutral_grey"], ls="-.", lw=1,
                   label=rf"pooled 95th = {thr[95]:.2f}")
    ax.set_xlabel(r"Ly$\alpha$ wavelength (air) [$\AA$]")
    ax.set_ylabel(r"False-positive $\Sigma$ percentile")
    sec = ax.secondary_xaxis("top", functions=(lambda w: w / pc.LYA_REST - 1,
                                               lambda z: (1 + z) * pc.LYA_REST))
    sec.set_xlabel("z", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    pcts = sorted(a.percentiles)
    if 98 not in pcts:
        pcts = sorted(pcts + [98])
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
        per.append(rec)
    if not pooled:
        raise SystemExit("No false-position S/N files found, run "
                         "optimize_lya_positions_f_grating.py first")
    pooled = np.concatenate(pooled)
    per = pd.DataFrame(per)
    thr = {p_: float(np.percentile(pooled, p_)) for p_ in pcts}

    print(f"\n{len(per)} sources, {pooled.size} false positions")
    print(f"  median {np.median(pooled):.2f}  (Paper 1: 2.7)")
    for p_ in pcts:
        print(f"  {p_:>5g}th percentile  {thr[p_]:.2f}")
    print(f"  source 98th percentiles range {per['p98'].min():.2f} to {per['p98'].max():.2f}, "
          f"median {per['p98'].median():.2f}")

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
    det = det.merge(per[["ID", "p98"]].rename(columns={"p98": "local_p98"}), on="ID", how="left")
    det["status"] = np.where(det["used"] == 1, "ok",
                             np.where(det["reason"] == "band coverage", "no_coverage",
                                      det["reason"].fillna("missing")))
    ok = det["status"] == "ok"
    for p_ in pcts:
        col = f"det_{pname(p_)}"
        det[col] = pd.array([pd.NA] * len(det), dtype="Int64")
        det.loc[ok, col] = (det.loc[ok, "lya_snr"] > thr[p_]).astype(int)
    det["det_local_98"] = pd.array([pd.NA] * len(det), dtype="Int64")
    det.loc[ok, "det_local_98"] = (det.loc[ok, "lya_snr"] > det.loc[ok, "local_p98"]).astype(int)
    det = det.drop(columns=["used", "reason"])
    lead = ["ID", "status", "lya_snr"] + [f"det_{pname(p_)}" for p_ in pcts] \
        + ["local_p98", "det_local_98"]
    det = det[lead].sort_values("ID")

    print("\nDetections (pooled thresholds)")
    for p_ in pcts:
        print(f"  above the {p_:>5g}th  {int(det[f'det_{pname(p_)}'].sum())} of {int(ok.sum())}")
    print(f"  above own 98th   {int(det['det_local_98'].sum())} of {int(ok.sum())}")
    nc = det.loc[det["status"] != "ok", "ID"].tolist()
    if nc:
        print(f"  not usable       {nc}")

    # Write
    row = dict(n_sources=len(per), n_positions=pooled.size,
               median=round(float(np.median(pooled)), 3))
    row.update({f"p{pname(p_)}": round(thr[p_], 3) for p_ in pcts})
    outs = [(a.out_thresholds, pd.DataFrame([row])), (a.out_per_source, per),
            (a.out_detections, det)]
    for path, d in outs:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d.to_csv(path, index=False)
    for path in (a.plot_hist, a.plot_wave):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    real_det = det.loc[ok & (det["det_98"] == 1), "lya_snr"].to_numpy(float)
    real_non = det.loc[ok & (det["det_98"] == 0), "lya_snr"].to_numpy(float)
    plot_hist(a.plot_hist, pooled, real_det, real_non, thr, pcts, a.xmax, a.bins)
    plot_wave(a.plot_wave, per, thr)

    print("\nWrote")
    for path in [o[0] for o in outs] + [a.plot_hist, a.plot_wave]:
        print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()