#!/usr/bin/env python3
"""
Split the Lya sample by velocity-offset precision and build a contact-sheet PDF
of the fit figures for each group.

Group 1 : delta_v_err_kms is finite and <= threshold (default 100 km/s), the
          sources whose velocity offset is precise enough to be scientifically
          useful.
Group 2 : everything else, delta_v_err_kms above the threshold OR non-finite
          (the missing-z_sys_err fallbacks and the runaway systematic errors).

Why 100 km/s: Lya velocity offsets in the literature span a few hundred km/s,
with the science in the ~100 to 600 km/s range. An error much above 100 km/s
means a source cannot distinguish a small offset from a large one, so it cannot
contribute to a Delta_v scaling relation. It is also roughly where the measured
delta_v_err distribution thins out, so the cut follows the data rather than an
arbitrary round number. Adjust with --threshold and regenerate.

For each group this writes a multi-page PDF tiling the existing
{ID}_lya_fit.png figures, 6 per page, each captioned with the source ID, the
OIII systemic-redshift grade (z_sys_quality, merged from the catalogue) and the
Lya S/N. A CSV with the group assignment per source is also written.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def load_grade_map(catalog_path, id_col, grade_col):
    """Return {ID: grade} from the catalogue, or empty if the column is absent."""
    cat = pd.read_csv(catalog_path)
    if id_col not in cat.columns:
        raise KeyError(f"Catalogue missing '{id_col}'. Available: {list(cat.columns)}")
    if grade_col not in cat.columns:
        print(f"[WARN] catalogue has no '{grade_col}' column, grade shown as '?'.")
        return {}
    cat[id_col] = cat[id_col].astype("Int64")
    return cat.set_index(id_col)[grade_col].to_dict()


def assign_group(dv_err, threshold):
    """Group 1 if dv_err finite and <= threshold, else group 2."""
    if np.isfinite(dv_err) and dv_err <= threshold:
        return 1
    return 2


def build_pdf(group_df, figdir, outpdf, grade_map, per_page, title):
    """Tile each source's lya_fit PNG into a multi-page PDF with captions."""
    ncols, nrows = 3, 2
    per_page = ncols * nrows

    ids = list(group_df["ID"])
    missing = []

    with PdfPages(outpdf) as pdf:
        for start in range(0, len(ids), per_page):
            page_ids = ids[start:start + per_page]

            fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 8.5))
            axes = np.atleast_1d(axes).ravel()

            for ax in axes:
                ax.axis("off")

            for ax, sid in zip(axes, page_ids):
                png = os.path.join(figdir, f"{sid}_lya_fit.png")
                row = group_df[group_df["ID"] == sid].iloc[0]

                grade = grade_map.get(sid, "?")
                if pd.isna(grade):
                    grade = "?"
                snr = row.get("lya_snr", np.nan)
                dv = row.get("delta_v_kms", np.nan)
                dv_err = row.get("delta_v_err_kms", np.nan)

                snr_str = f"{snr:.1f}" if np.isfinite(snr) else "nan"
                dv_str = f"{dv:.0f}" if np.isfinite(dv) else "nan"
                dverr_str = f"{dv_err:.0f}" if np.isfinite(dv_err) else "nan"

                caption = (f"ID {sid}   OIII grade {grade}   "
                           f"Lya S/N {snr_str}   "
                           f"$\\Delta v$={dv_str}$\\pm${dverr_str} km/s")

                if os.path.exists(png):
                    ax.imshow(mpimg.imread(png))
                else:
                    missing.append(sid)
                    ax.text(0.5, 0.5, f"ID {sid}\n(no fit PNG)",
                            ha="center", va="center", fontsize=12,
                            transform=ax.transAxes)

                ax.set_title(caption, fontsize=10)
                ax.axis("off")

            fig.suptitle(title, fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return missing


def main():
    p = argparse.ArgumentParser(
        description="Group Lya sources by delta_v_err and tile fit PNGs to PDF.")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="CSV with ID, delta_v_kms, delta_v_err_kms, lya_snr.")
    p.add_argument("--catalog",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv",
                   help="Catalogue providing the OIII grade (z_sys_quality).")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures",
                   help="Directory holding the {ID}_lya_fit.png figures.")
    p.add_argument("--outdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs",
                   help="Where the group CSV and the two PDFs are written.")

    p.add_argument("--id-col", default="ID")
    p.add_argument("--grade-col", default="z_sys_quality",
                   help="Catalogue column with the OIII a/b/c/d grade.")
    p.add_argument("--threshold", type=float, default=100.0,
                   help="delta_v_err_kms cut in km/s (default 100).")
    args = p.parse_args()

    props = pd.read_csv(args.properties_csv)
    if "delta_v_err_kms" not in props.columns:
        raise KeyError(
            f"'delta_v_err_kms' not in {args.properties_csv}. Run the MC error "
            f"script first. Available: {list(props.columns)}")
    props["ID"] = props["ID"].astype(int)

    grade_map = load_grade_map(args.catalog, args.id_col, args.grade_col)

    props["dv_err_group"] = props["delta_v_err_kms"].apply(
        lambda e: assign_group(e, args.threshold))

    g1 = props[props["dv_err_group"] == 1].sort_values("delta_v_err_kms")
    g2 = props[props["dv_err_group"] == 2].sort_values(
        "delta_v_err_kms", na_position="last")

    os.makedirs(args.outdir, exist_ok=True)

    out_csv = os.path.join(args.outdir, "lya_properties_grouped.csv")
    props.to_csv(out_csv, index=False)

    pdf1 = os.path.join(args.outdir, "lya_fits_group1_dv_err_le{:g}.pdf".format(args.threshold))
    pdf2 = os.path.join(args.outdir, "lya_fits_group2_dv_err_gt{:g}.pdf".format(args.threshold))

    print("[CONFIG]")
    print(f"  Properties CSV : {os.path.abspath(args.properties_csv)}")
    print(f"  Catalogue      : {os.path.abspath(args.catalog)}")
    print(f"  Figure dir     : {os.path.abspath(args.figdir)}")
    print(f"  Threshold      : {args.threshold:g} km/s")
    print("")
    print(f"[INFO] Group 1 (dv_err <= {args.threshold:g}) : {len(g1)} sources")
    print(f"[INFO] Group 2 (dv_err >  {args.threshold:g} or non-finite) : {len(g2)} sources")
    print("")

    m1 = build_pdf(g1, args.figdir, pdf1, grade_map, 6,
                   f"Group 1  |  dv_err <= {args.threshold:g} km/s  |  {len(g1)} sources")
    m2 = build_pdf(g2, args.figdir, pdf2, grade_map, 6,
                   f"Group 2  |  dv_err > {args.threshold:g} km/s or non-finite  |  {len(g2)} sources")

    print(f"[DONE] Group assignment CSV : {os.path.abspath(out_csv)}")
    print(f"[DONE] Group 1 PDF          : {os.path.abspath(pdf1)}")
    print(f"[DONE] Group 2 PDF          : {os.path.abspath(pdf2)}")
    if m1 or m2:
        allm = sorted(set(m1) | set(m2))
        print(f"[WARN] {len(allm)} sources had no fit PNG: {allm}")


if __name__ == "__main__":
    main()


"""
python group_by_dv_err.py \
  --properties-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --figdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures \
  --outdir /ceph/cephfs/apatrick/P2/MUSE_catalogs \
  --threshold 100
"""