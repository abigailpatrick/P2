#!/usr/bin/env python
"""
false_pos_percentiles.py

Pool the peak-S/N values from every per-source false-positive optimiser output
(source_{ID}_false_optimal_snr.csv, written by optimize_lya_positions_f_grating.py)
and report the overall percentile thresholds across ALL false positions combined.

By default it reports the 98th and 99.5th percentiles, which set the detection
thresholds for the real sources.
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

try:
    import cmasher as cmr
    _HAVE_CMASHER = True
except ImportError:
    _HAVE_CMASHER = False


# ---------------------------------------------------------------------------
# Matplotlib style - MNRAS single-column (matched to the P1 figure)
# ---------------------------------------------------------------------------
MNRAS_COL_WIDTH_IN = 3.46   # 88 mm
MNRAS_FIG_HEIGHT_IN = 2.70

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

THEME = "torch"


def make_palette(name, n=8):
    """Build a colour palette from a cmasher colourmap, matching the P1 plot.
    Falls back to matplotlib viridis samples if cmasher is unavailable."""
    if _HAVE_CMASHER:
        hexes = cmr.take_cmap_colors(
            f"cmr.{name}", n, cmap_range=(0.10, 0.90), return_fmt="hex")
    else:
        cmap = plt.get_cmap("viridis")
        hexes = [mcolors.to_hex(cmap(x)) for x in np.linspace(0.10, 0.90, n)]
    palette = {f"c{i}": h for i, h in enumerate(hexes)}
    palette["near_black"] = mcolors.to_hex((0.10, 0.10, 0.12, 1.0))
    palette["neutral_grey"] = "#aab4c8"
    return palette


def parse_args():
    p = argparse.ArgumentParser(
        description="Pool all false-positive peak S/N and report overall percentiles."
    )
    p.add_argument("--dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions",
                   help="Directory of source_{ID}_false_optimal_snr.csv files")
    p.add_argument("--snr-col", default="peak_snr",
                   help="Column holding the per-position peak S/N")
    p.add_argument("--percentiles", type=float, nargs="+",
                   default=[98.0, 99.5],
                   help="Percentiles to report (default 98 99.5)")
    p.add_argument("--out-csv", default=None,
                   help="Optional path to write a one-row summary CSV")
    p.add_argument("--plot",
                   default="/ceph/cephfs/apatrick/P2/plots/false_pos_summary.png",
                   help="Output path for the summary histogram PNG. "
                        "Set to '' to skip plotting.")
    p.add_argument("--bins", type=int, default=300,
                   help="Number of histogram bins")
    p.add_argument("--xmax", type=float, default=20.0,
                   help="Upper x-axis limit for the histogram")
    return p.parse_args()


def main():
    args = parse_args()

    fdir = os.path.abspath(args.dir)
    print("[PATHS]")
    print(f"  Input directory : {fdir}")
    if args.out_csv:
        print(f"  Summary CSV     : {os.path.abspath(args.out_csv)}")
    print("")

    files = sorted(glob.glob(os.path.join(fdir, "source_*_false_optimal_snr.csv")))
    if not files:
        raise SystemExit(f"No source_*_false_optimal_snr.csv files found in {fdir}")

    all_snr = []
    per_source = []
    n_files = 0

    for path in files:
        m = re.search(r"source_(\d+)_false_optimal_snr", os.path.basename(path))
        sid = int(m.group(1)) if m else -1

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(path)}: could not read ({e})")
            continue

        if args.snr_col not in df.columns:
            print(f"[SKIP] {os.path.basename(path)}: no '{args.snr_col}' column "
                  f"(has {list(df.columns)})")
            continue

        vals = pd.to_numeric(df[args.snr_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"[SKIP] {os.path.basename(path)}: no finite S/N values")
            continue

        all_snr.append(vals)
        per_source.append((sid, vals.size))
        n_files += 1

    if not all_snr:
        raise SystemExit("No usable S/N values found in any file.")

    pooled = np.concatenate(all_snr)

    print(f"[POOL] {n_files} source files, {pooled.size} false positions total")
    print(f"       min={pooled.min():.2f}  median={np.median(pooled):.2f}  "
          f"max={pooled.max():.2f}  mean={pooled.mean():.2f}")
    print("")

    print("[OVERALL PERCENTILES] (all false positions combined)")
    results = {}
    for pc in args.percentiles:
        thr = float(np.percentile(pooled, pc))
        results[pc] = thr
        print(f"  {pc:>6.1f}th percentile S/N = {thr:.3f}")
    print("")

    # ------------------------------------------------------------------
    # Summary histogram (cmasher-styled, MNRAS single-column)
    # ------------------------------------------------------------------
    if args.plot:
        plot_path = os.path.abspath(args.plot)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        pal = make_palette(THEME)
        c_false = pal["c1"]

        # Always mark the 98 and 99.5 thresholds on the figure
        p98 = float(np.percentile(pooled, 98.0))
        p995 = float(np.percentile(pooled, 99.5))

        snr_min = max(0.0, float(pooled.min()))
        bins = np.linspace(snr_min, args.xmax, args.bins + 1)

        # Peak-normalise so the tallest bar is 1, as in the P1 figure
        counts, _ = np.histogram(pooled, bins=bins)
        peak = counts.max() if counts.max() > 0 else 1.0
        weights = np.ones_like(pooled, dtype=float) / peak

        fig, ax = plt.subplots(
            figsize=(MNRAS_COL_WIDTH_IN, MNRAS_FIG_HEIGHT_IN), dpi=300)

        ax.hist(pooled, bins=bins, weights=weights,
                histtype="stepfilled", linewidth=0, color=c_false,
                alpha=0.25, zorder=2, label="False positives")
        ax.hist(pooled, bins=bins, weights=weights,
                histtype="step", linewidth=0.9, color=c_false,
                alpha=0.60, zorder=3)

        ax.axvline(p98, color=pal["near_black"], linewidth=1.6, linestyle="--",
                   zorder=6,
                   label=rf"$\Sigma^{{98}}={p98:.2f}$")
        ax.axvline(p995, color=pal["neutral_grey"], linewidth=1.0, linestyle=":",
                   zorder=6,
                   label=rf"$\Sigma^{{99.5}}={p995:.2f}$")

        ax.set_xlabel(r"Peak $\Sigma$", fontsize=10)
        ax.set_ylabel("Normalised counts", fontsize=10)
        ax.set_xlim(snr_min, args.xmax)
        ax.set_ylim(0.0, 1.12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))

        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  handlelength=1.6, handletextpad=0.4, labelspacing=0.35)

        fig.tight_layout(pad=0.4)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved summary histogram to {plot_path}")
        print("")

    if args.out_csv:
        out_path = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        row = {
            "n_source_files": n_files,
            "n_false_positions": pooled.size,
            "min": pooled.min(),
            "median": float(np.median(pooled)),
            "max": pooled.max(),
            "mean": pooled.mean(),
        }
        for pc, thr in results.items():
            row[f"pct_{pc}"] = thr
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()


"""
report overall 98 and 99.5 percentiles across all false positions
python false_pos_percentiles.py \
  --dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions

also write a summary CSV
python false_pos_percentiles.py \
  --dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions \
  --out-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/false_pos_snr_percentiles.csv
"""