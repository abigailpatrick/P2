#!/usr/bin/env python
"""
make_lya_fit_pdfs.py

Contact sheets of the Lya fits for checking by eye what the detections and
non-detections look like at each false-positive threshold.

For each threshold (default the pooled 99.5th, 98th and 95th percentiles from
false_pos_percentiles.py) two PDFs are written, one of the sources whose
lya_snr is above it and one of those below. Each page holds four of the
<ID>_lya_fit.png figures (spectrum with the fit and detection window,
pseudo-NB), sorted by lya_snr from highest to lowest, with a caption giving
the S/N, z_sys, Delta_v and the fit's quality flag. The first page lists the
threshold and every source in the PDF.

Sources without band coverage (status not "ok" in lya_detections.csv, e.g.
65863 in the AO gap) are in neither PDF and are listed on the summary pages.

Inputs
------
  MUSE_catalogs/lya_detections.csv              status, lya_snr, det_<p> flags
  MUSE_catalogs/lya_false_pos_thresholds.csv    threshold values
  MUSE_catalogs/lya_fits.csv                    z_sys, delta_v_kms, quality_reason
  MUSE_subcubes/<ID>/<ID>_lya_fit.png

Outputs, in plots/lya_fit_pdfs/
-------
  lya_fits_above_99_5.pdf   lya_fits_below_99_5.pdf
  lya_fits_above_98.pdf     lya_fits_below_98.pdf
  lya_fits_above_95.pdf     lya_fits_below_95.pdf

Usage
-----
python make_lya_fit_pdfs.py
python make_lya_fit_pdfs.py --levels 99.5 98 95 90
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import p2_common as pc

CAT = pc.MUSE_CAT_DIR
PER_PAGE = 4
A4 = (8.27, 11.69)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", default=f"{CAT}/lya_detections.csv")
    p.add_argument("--thresholds", default=f"{CAT}/lya_false_pos_thresholds.csv")
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits.csv")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outdir", default=f"{pc.P2_ROOT}/plots/lya_fit_pdfs")
    p.add_argument("--levels", type=float, nargs="+", default=[99.5, 98, 95])
    p.add_argument("--width-px", type=int, default=2000,
                   help="figures are shrunk to this width to keep the PDFs small")
    return p.parse_args()


def pname(p_):
    return f"{p_:g}".replace(".", "_")


def load_png(path, width):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return np.asarray(im)


def caption(r):
    parts = [f"ID {int(r['ID'])}", rf"$\Sigma$ = {r['lya_snr']:.2f}"]
    if pd.notna(r.get("z_sys")):
        parts.append(f"z_sys = {r['z_sys']:.4f}")
    if pd.notna(r.get("delta_v_kms")):
        parts.append(rf"$\Delta v$ = {r['delta_v_kms']:.0f} km/s")
    if isinstance(r.get("quality_reason"), str) and r["quality_reason"] != "ok":
        parts.append(f"flag: {r['quality_reason']}")
    return "    ".join(parts)


def summary_page(pdf, title, thr, rows, unusable):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold", va="top")
    fig.text(0.08, 0.92, rf"threshold $\Sigma$ = {thr:.2f}, {len(rows)} sources, "
             "sorted by S/N", fontsize=10, va="top")
    lines = [f"{int(r['ID']):>6d}   {r['lya_snr']:6.2f}" for _, r in rows.iterrows()]
    ncol = 3
    per_col = int(np.ceil(len(lines) / ncol)) if lines else 0
    for c in range(ncol):
        chunk = lines[c * per_col:(c + 1) * per_col]
        fig.text(0.08 + c * 0.3, 0.88, "    ID      S/N\n" + "\n".join(chunk),
                 family="monospace", fontsize=8, va="top")
    if unusable:
        fig.text(0.08, 0.06, "Not in either PDF (no band coverage): "
                 + ", ".join(map(str, unusable)), fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, title, thr, rows, unusable, args):
    missing = []
    with PdfPages(path) as pdf:
        summary_page(pdf, title, thr, rows, unusable)
        for start in range(0, len(rows), PER_PAGE):
            fig, axes = plt.subplots(PER_PAGE, 1, figsize=A4)
            for ax, (_, r) in zip(axes, rows.iloc[start:start + PER_PAGE].iterrows()):
                png = pc.muse_path(int(r["ID"]), "fit_png", args.subcube_dir)
                if os.path.exists(png):
                    ax.imshow(load_png(png, args.width_px))
                else:
                    missing.append(int(r["ID"]))
                    ax.text(0.5, 0.5, f"no figure at {png}", ha="center", fontsize=7,
                            transform=ax.transAxes)
                ax.set_title(caption(r), fontsize=8, loc="left")
                ax.set_axis_off()
            for ax in axes[len(rows.iloc[start:start + PER_PAGE]):]:
                ax.set_axis_off()
            fig.tight_layout(h_pad=0.6)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return missing


def main():
    a = parse_args()
    print("make_lya_fit_pdfs.py")
    for path in (a.detections, a.thresholds, a.fits_csv):
        print(f"Reading  {os.path.abspath(path)}")
    det = pd.read_csv(a.detections)
    thr = pd.read_csv(a.thresholds).iloc[0]
    fits = pd.read_csv(a.fits_csv)[["ID", "z_sys", "delta_v_kms", "quality_reason"]]
    det = det.merge(fits, on="ID", how="left")
    ok = det[det["status"] == "ok"].sort_values("lya_snr", ascending=False)
    unusable = det.loc[det["status"] != "ok", "ID"].astype(int).tolist()
    os.makedirs(a.outdir, exist_ok=True)
    print("")

    written = []
    for lev in a.levels:
        col = f"p{pname(lev)}"
        if col not in thr:
            print(f"  WARNING no {col} in the thresholds file, skipped. Rerun "
                  f"false_pos_percentiles.py with --percentiles including {lev:g}")
            continue
        t = float(thr[col])
        above = ok[ok["lya_snr"] > t]
        below = ok[ok["lya_snr"] <= t]
        for rows, side in ((above, "above"), (below, "below")):
            path = os.path.join(os.path.abspath(a.outdir), f"lya_fits_{side}_{pname(lev)}.pdf")
            title = f"Lya fits {side} the {lev:g}th percentile"
            missing = write_pdf(path, title, t, rows, unusable, a)
            print(f"  {lev:>5g}th ({t:.2f})  {side:<5}  {len(rows):>3d} sources  {path}")
            if missing:
                print(f"         WARNING no fit figure for {missing}")
            written.append(path)
    print(f"\n{len(written)} PDFs written to {os.path.abspath(a.outdir)}")


if __name__ == "__main__":
    main()