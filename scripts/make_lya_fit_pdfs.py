#!/usr/bin/env python
"""
make_lya_fit_pdfs.py

Contact sheets of the Lya fits for checking the detections and
non-detections by eye. By default (--figure best) each source shows the fit
the catalogue prefers (lya_model): the skewed-Gaussian figure
<ID>_lya_fit.png for single, or the double-peak figure
<ID>_lya_double_fit.png for double. --figure single or double shows that
figure for every source instead.

The sets follow lya_catalog.csv (build_lya_catalog.py), so they follow
whichever detection rule the catalogue was built with. The rule tag (det_rule,
e.g. p95 or p90_dverr100) goes into every PDF name.
  detections       in_analysis == 1 and lya_det == 1
  non_detections   in_analysis == 1 and lya_det == 0
  close_pairs      pair_status == close_pair, detected or not
Interlopers (the lower-redshift member of close-pair groups 4 and 5) are in
no PDF. Sources not tested (lya_status not ok, e.g. 65863) are in no PDF and
are listed on the first page of each.

Each page holds four figures sorted by lya_snr from highest to lowest, each
captioned with the S/N, z_sys, the preferred model with its Delta_BIC,
Delta_v and its error (from the preferred fit, so the red peak for a double),
the blue peak for a double, the detection state (and the reason for a
non-detection) and any fit quality flag. Close-pair
captions also give the group number. The first page lists the detection
rule and every source in the PDF.

Inputs
------
  MUSE_catalogs/lya_catalog.csv                 lya_det, det_reason, det_rule, ...
  MUSE_catalogs/lya_detection_cuts_<tag>.csv    the rule, quoted on the first page
  MUSE_catalogs/lya_fits.csv               quality_reason
  MUSE_subcubes/<ID>/<ID>_lya_fit.png
  MUSE_subcubes/<ID>/<ID>_lya_double_fit.png    for double sources, or --figure double

Outputs, in plots/lya_fit_pdfs/
-------
  lya_fits_<tag>_detections.pdf
  lya_fits_<tag>_non_detections.pdf
  lya_fits_<tag>_close_pairs.pdf
--figure single and --figure double write the same three as
lya_single_fits_<tag>_<set>.pdf and lya_double_fits_<tag>_<set>.pdf.

Usage
-----
python make_lya_fit_pdfs.py                                        # preferred fit per source
python make_lya_fit_pdfs.py --figure double                        # every source's double fit
python make_lya_fit_pdfs.py --figure single                        # every source's skewed Gaussian
python make_lya_fit_pdfs.py --catalog MUSE_catalogs/lya_catalog_p90.csv
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
TITLES = {"detections": "Lya fits, detections ({})",
          "non_detections": "Lya fits, non-detections ({})",
          "close_pairs": "Lya fits, close pairs, not in the analysis ({})"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.LYA_CATALOG)
    p.add_argument("--cuts", default=None,
                   help="default lya_detection_cuts_<tag>.csv next to the catalogue")
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits.csv")
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outdir", default=f"{pc.P2_ROOT}/plots/lya_fit_pdfs")
    p.add_argument("--figure", choices=["best", "single", "double"], default="best",
                   help="best (default): the preferred fit of each source (lya_model). "
                        "single: every source's skewed-Gaussian figure (<ID>_lya_fit.png). "
                        "double: every source's double-peak figure (<ID>_lya_double_fit.png)")
    p.add_argument("--width-px", type=int, default=2000,
                   help="figures are shrunk to this width to keep the PDFs small")
    return p.parse_args()


def load_png(path, width):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return np.asarray(im)


def rule_text(cuts_path, tex=True):
    """One line describing the detection rule, from lya_detection_cuts_<tag>.csv.
    tex=False gives plain text for the terminal."""
    if not os.path.exists(cuts_path):
        return f"Detection rule: see {cuts_path}"
    c = pd.read_csv(cuts_path).iloc[0]
    thr = float(c["snr_threshold"])
    extra = str(c.get("extra_cut", "none"))
    fixed = str(c.get("det_mode", "percentile")) == "snr"
    pct = c.get("snr_threshold_percentile", np.nan)
    if fixed:
        p_txt = f" ({float(pct):.2f}th percentile)" if pd.notna(pct) else ""
    else:
        p_ = float(c["det_percentile"])
    if tex:
        txt = (rf"Detection ({c['det_rule']}): $\Sigma \geq {thr:g}${p_txt}" if fixed else
               rf"Detection ({c['det_rule']}): $\Sigma > \Sigma^{{{p_:g}}} = {thr:.2f}$")
        if extra == "dv_err":
            txt += rf" and $\sigma_{{\Delta v}} \leq {float(c['dv_err_max']):g}$ km/s"
        elif extra == "centre_err":
            txt += rf" and Ly$\alpha$ centre error $\leq {float(c['centre_err_max']):g}$ km/s"
        return txt
    txt = (f"Detection ({c['det_rule']}): S/N >= {thr:g}{p_txt}" if fixed else
           f"Detection ({c['det_rule']}): S/N > {thr:.2f} ({p_:g}th percentile)")
    if extra == "dv_err":
        txt += f" and Delta_v error <= {float(c['dv_err_max']):g} km/s"
    elif extra == "centre_err":
        txt += f" and Lya centre error <= {float(c['centre_err_max']):g} km/s"
    return txt


def cuts_for(cat, catalog_path, given=None):
    """(tag, path of the matching lya_detection_cuts_<tag>.csv)."""
    tag = pc.det_rule(cat)
    path = given or pc.detection_cuts_path(tag, os.path.dirname(os.path.abspath(catalog_path)))
    return tag, path


def caption(r, with_group=False):
    parts = [f"ID {int(r['ID'])}"]
    if with_group:
        parts.append(f"group {int(r['close_pair_group'])}")
    parts.append(rf"$\Sigma$ = {r['lya_snr']:.2f}")
    if pd.notna(r.get("z_sys")):
        parts.append(f"z_sys = {r['z_sys']:.4f}")
    model = r.get("lya_model")
    if isinstance(model, str) and model:
        b = (rf" $\Delta$BIC {r['dbic_double']:.1f}" if pd.notna(r.get("dbic_double")) else "")
        parts.append(f"{model}{b}")
    if pd.notna(r.get("delta_v")):
        e = (f" $\\pm$ {r['delta_v_err']:.0f}" if pd.notna(r.get("delta_v_err")) else "")
        if pd.notna(r.get("lya_centre_err")):
            e += f" (centre $\\pm$ {r['lya_centre_err']:.0f})"
        lab = r"$\Delta v_{\rm red}$" if model == "double" else r"$\Delta v$"
        parts.append(rf"{lab} = {r['delta_v']:.0f}{e} km/s")
    if model == "double" and pd.notna(r.get("dv_blue")):
        parts.append(rf"blue {r['dv_blue']:.0f} km/s, B/R {r['blue_red_ratio']:.2f}")
    parts.append(pc.det_label(r))
    if isinstance(r.get("quality_reason"), str) and r["quality_reason"] != "ok":
        parts.append(f"fit flag: {r['quality_reason']}")
    return "    ".join(parts)


def summary_page(pdf, title, rule, rows, untested, with_reason):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold", va="top")
    fig.text(0.08, 0.92, rule, fontsize=9, va="top")
    fig.text(0.08, 0.895, f"{len(rows)} sources, sorted by S/N", fontsize=9, va="top")
    lines = []
    for _, r in rows.iterrows():
        extra = f"  {r['det_reason']}" if with_reason and r.get("det_reason") else ""
        lines.append(f"{int(r['ID']):>6d}  {r['lya_snr']:6.2f}{extra}")
    ncol = 3
    per_col = int(np.ceil(len(lines) / ncol)) if lines else 0
    for c in range(ncol):
        chunk = lines[c * per_col:(c + 1) * per_col]
        fig.text(0.08 + c * 0.3, 0.86, "    ID     S/N\n" + "\n".join(chunk),
                 family="monospace", fontsize=7.5, va="top")
    if untested:
        fig.text(0.08, 0.06, "Not tested for Lya, in no PDF: " + ", ".join(map(str, untested)),
                 fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, title, rule, rows, untested, args, with_group, with_reason):
    missing = []
    with PdfPages(path) as pdf:
        summary_page(pdf, title, rule, rows, untested, with_reason)
        for start in range(0, len(rows), PER_PAGE):
            chunk = rows.iloc[start:start + PER_PAGE]
            fig, axes = plt.subplots(PER_PAGE, 1, figsize=A4)
            for ax, (_, r) in zip(axes, chunk.iterrows()):
                if args.figure == "best":
                    key = "double_fit_png" if r.get("lya_model") == "double" else "fit_png"
                else:
                    key = "fit_png" if args.figure == "single" else "double_fit_png"
                png = pc.muse_path(int(r["ID"]), key, args.subcube_dir)
                if os.path.exists(png):
                    ax.imshow(load_png(png, args.width_px))
                else:
                    missing.append(int(r["ID"]))
                    ax.text(0.5, 0.5, f"no figure at {png}", ha="center", fontsize=7,
                            transform=ax.transAxes)
                ax.set_title(caption(r, with_group), fontsize=7.5, loc="left")
                ax.set_axis_off()
            for ax in axes[len(chunk):]:
                ax.set_axis_off()
            fig.tight_layout(h_pad=0.6)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return missing


def main():
    a = parse_args()
    print("make_lya_fit_pdfs.py")
    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    tag, cuts = cuts_for(cat, a.catalog, a.cuts)
    for path in (a.catalog, cuts, a.fits_csv):
        print(f"  reading  {os.path.abspath(path)}")
    if os.path.exists(a.fits_csv) and "quality_reason" in pd.read_csv(a.fits_csv, nrows=0).columns:
        q = pd.read_csv(a.fits_csv)[["ID", "quality_reason"]]
        q["ID"] = q["ID"].astype(int)
        cat = cat.merge(q, on="ID", how="left")
    cat["det_reason"] = cat["det_reason"].fillna("")
    groups, untested, interlopers = pc.lya_pdf_groups(cat)
    rule = rule_text(cuts)
    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    stem = {"best": "lya_fits", "single": "lya_single_fits", "double": "lya_double_fits"}[a.figure]
    print(f"  writing  {outdir}/{stem}_{tag}_<detections|non_detections|close_pairs>.pdf")
    print(f"  {rule_text(cuts, tex=False)}\n")

    for name, rows in groups.items():
        path = os.path.join(outdir, f"{stem}_{tag}_{name}.pdf")
        title = TITLES[name].format(tag)
        if a.figure != "best":
            title = title.replace("Lya fits", "Lya double-peak fits" if a.figure == "double"
                                  else "Lya skewed-Gaussian fits")
        missing = write_pdf(path, title, rule, rows, untested, a,
                            with_group=name == "close_pairs",
                            with_reason=name != "detections")
        print(f"  {name:<15} {len(rows):>3d} sources  {path}")
        if missing:
            print(f"                  WARNING no fit figure for {missing}")
    if interlopers:
        print(f"\n  interlopers, in no PDF: {interlopers}")
    if untested:
        print(f"  not tested, in no PDF:  {untested}")


if __name__ == "__main__":
    main()