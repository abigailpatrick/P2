#!/usr/bin/env python3
"""
Tier the Lya sample into gold / silver / bronze / bad and build a contact-sheet
PDF of the fit figures for each tier, with per-tier summary histograms.

Five quality criteria are evaluated per source:
  1. OIII grade (z_sys_quality) == 'a'
  2. FWHM > 100 km/s
  3. delta_v_err_kms < 120 km/s
  4. Lya S/N > 2
  5. NOT flagged skew_pinned or dv_out_of_band (the two count as one combined
     criterion, so a source flagged with either fails this one criterion once)

Tiers by cumulative number of failed criteria:
  gold   : 0 failures
  silver : <= 1 failure
  bronze : <= 3 failures
  bad    : everything else (>= 4 failures, or fit failed)

Duplicate sources (the same object under two IDs, sharing RA/Dec) are collapsed
to a single row, keeping the one with the lowest delta_v_err_kms.

For each tier the PDF tiles the {ID}_lya_fit.png figures 6 per page, each
captioned with ID, OIII grade, Lya S/N, delta_v +/- error, and a note if the
source is skew_pinned and/or dv_out_of_band. The final page of each tier holds
summary histograms of systemic redshift, velocity offset and FWHM.
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


# Criterion thresholds (kept as module constants so the docstring, the code and
# any future CLI exposure stay in one place)
GRADE_GOLD = "a"
FWHM_MIN = 100.0        # km/s
DV_ERR_MAX = 120.0      # km/s
SNR_MIN = 2.0


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


def deduplicate(props, ra_round=5, dec_round=5):
    """Collapse duplicate sources (same RA/Dec) to the lowest-delta_v_err row.

    Sources sharing RA and Dec to ra_round/dec_round decimals are treated as the
    same object. Within each group the row with the smallest finite
    delta_v_err_kms is kept (NaN errors sort last, so a finite twin wins).
    """
    if "ra" not in props.columns or "dec" not in props.columns:
        print("[WARN] no ra/dec columns, cannot deduplicate, keeping all rows.")
        return props, []

    df = props.copy()
    df["_ra_key"] = df["ra"].round(ra_round)
    df["_dec_key"] = df["dec"].round(dec_round)
    df["_err_sort"] = df["delta_v_err_kms"].where(
        np.isfinite(df["delta_v_err_kms"]), np.inf)

    dropped = []
    keep_idx = []
    for _, grp in df.groupby(["_ra_key", "_dec_key"], sort=False):
        if len(grp) == 1:
            keep_idx.append(grp.index[0])
            continue
        grp_sorted = grp.sort_values("_err_sort")
        keep = grp_sorted.index[0]
        keep_idx.append(keep)
        losers = [int(df.loc[i, "ID"]) for i in grp_sorted.index[1:]]
        dropped.append((int(df.loc[keep, "ID"]), losers))

    out = df.loc[keep_idx].drop(columns=["_ra_key", "_dec_key", "_err_sort"])
    return out, dropped


def has_flag(reason, token):
    """True if a quality_reason string contains the given token."""
    if not isinstance(reason, str):
        return False
    return token in reason.split(";")


def evaluate_source(row):
    """Return (n_fail, criteria_dict, pinned, oob) for one source.

    criteria_dict maps each criterion name to True (passed) or False (failed).
    """
    fit_ok = bool(row.get("fit_success", False))
    grade = row.get("_grade", "?")
    fwhm = pd.to_numeric(row.get("fwhm_kms"), errors="coerce")
    dv_err = pd.to_numeric(row.get("delta_v_err_kms"), errors="coerce")
    snr = pd.to_numeric(row.get("lya_snr"), errors="coerce")
    reason = row.get("quality_reason", "")

    pinned = has_flag(reason, "skew_pinned")
    oob = has_flag(reason, "dv_out_of_band")

    crit = {
        "grade_a": (str(grade) == GRADE_GOLD),
        "fwhm>100": bool(np.isfinite(fwhm) and fwhm > FWHM_MIN),
        "dv_err<120": bool(np.isfinite(dv_err) and dv_err < DV_ERR_MAX),
        "snr>2": bool(np.isfinite(snr) and snr > SNR_MIN),
        "not_pinned_or_oob": (not pinned and not oob),
    }

    # A failed fit fails everything
    if not fit_ok:
        crit = {k: False for k in crit}

    n_fail = sum(1 for v in crit.values() if not v)
    return n_fail, crit, pinned, oob


def tier_from_fails(n_fail, fit_ok):
    """Map failure count to a tier label."""
    if not fit_ok:
        return "bad"
    if n_fail == 0:
        return "gold"
    if n_fail <= 1:
        return "silver"
    if n_fail <= 3:
        return "bronze"
    return "bad"


def caption_for(row):
    """Build the per-panel caption, including flag notes."""
    sid = int(row["ID"])
    grade = row.get("_grade", "?")
    if pd.isna(grade):
        grade = "?"
    snr = pd.to_numeric(row.get("lya_snr"), errors="coerce")
    dv = pd.to_numeric(row.get("delta_v_kms"), errors="coerce")
    dv_err = pd.to_numeric(row.get("delta_v_err_kms"), errors="coerce")

    snr_str = f"{snr:.1f}" if np.isfinite(snr) else "nan"
    dv_str = f"{dv:.0f}" if np.isfinite(dv) else "nan"
    dverr_str = f"{dv_err:.0f}" if np.isfinite(dv_err) else "nan"

    reason = row.get("quality_reason", "")
    notes = []
    if has_flag(reason, "skew_pinned"):
        notes.append("skew pinned")
    if has_flag(reason, "dv_out_of_band"):
        notes.append("dv OOB")
    note_str = ("   [" + ", ".join(notes) + "]") if notes else ""

    return (f"ID {sid}   OIII {grade}   S/N {snr_str}   "
            f"$\\Delta v$={dv_str}$\\pm${dverr_str} km/s{note_str}")


def make_summary_page(pdf, tier_df, tier_name):
    """Add a page of z_sys, delta_v and FWHM histograms for the tier."""
    z = pd.to_numeric(tier_df.get("z_sys"), errors="coerce").dropna()
    dv = pd.to_numeric(tier_df.get("delta_v_kms"), errors="coerce").dropna()
    fwhm = pd.to_numeric(tier_df.get("fwhm_kms"), errors="coerce").dropna()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))

    axes[0].hist(z, bins="auto", color="steelblue", edgecolor="black")
    axes[0].set_xlabel(r"$z_{\rm sys}$")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Systemic redshift")

    axes[1].hist(dv, bins="auto", color="seagreen", edgecolor="black")
    axes[1].axvline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\Delta v$ [km/s]")
    axes[1].set_title("Velocity offset")

    axes[2].hist(fwhm, bins="auto", color="indianred", edgecolor="black")
    axes[2].set_xlabel("FWHM [km/s]")
    axes[2].set_title("Line width")

    fig.suptitle(f"{tier_name.capitalize()} tier summary  |  {len(tier_df)} sources",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def build_pdf(tier_df, figdir, outpdf, tier_name):
    """Tile fit PNGs 6 per page, then append a summary-histogram page."""
    ncols, nrows = 3, 2
    per_page = ncols * nrows
    ids = list(tier_df["ID"])
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
                row = tier_df[tier_df["ID"] == sid].iloc[0]
                if os.path.exists(png):
                    ax.imshow(mpimg.imread(png))
                else:
                    missing.append(sid)
                    ax.text(0.5, 0.5, f"ID {sid}\n(no fit PNG)",
                            ha="center", va="center", fontsize=12,
                            transform=ax.transAxes)
                ax.set_title(caption_for(row), fontsize=10)
                ax.axis("off")

            fig.suptitle(f"{tier_name.capitalize()} tier  |  {len(ids)} sources",
                         fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        if len(tier_df):
            make_summary_page(pdf, tier_df, tier_name)

    return missing


def main():
    p = argparse.ArgumentParser(
        description="Tier Lya sample (gold/silver/bronze/bad) and build PDFs.")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv")
    p.add_argument("--catalog",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures")
    p.add_argument("--outdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--grade-col", default="z_sys_quality")
    args = p.parse_args()

    props = pd.read_csv(args.properties_csv)
    for col in ("ID", "delta_v_err_kms", "fwhm_kms", "lya_snr",
                "delta_v_kms", "z_sys", "quality_reason", "fit_success"):
        if col not in props.columns:
            raise KeyError(
                f"'{col}' not in {args.properties_csv}. Available: {list(props.columns)}")
    props["ID"] = props["ID"].astype(int)

    grade_map = load_grade_map(args.catalog, args.id_col, args.grade_col)
    props["_grade"] = props["ID"].map(grade_map)

    # Deduplicate before tiering
    props, dropped = deduplicate(props)
    if dropped:
        print("[INFO] Duplicate sources collapsed (kept, dropped):")
        for kept, losers in dropped:
            print(f"   kept {kept}, dropped {losers}")
        print("")

    # Evaluate every source
    tiers, nfails, notes_pinned, notes_oob = [], [], [], []
    crit_cols = {k: [] for k in
                 ["grade_a", "fwhm>100", "dv_err<120", "snr>2", "not_pinned_or_oob"]}
    for _, row in props.iterrows():
        n_fail, crit, pinned, oob = evaluate_source(row)
        tier = tier_from_fails(n_fail, bool(row.get("fit_success", False)))
        tiers.append(tier)
        nfails.append(n_fail)
        notes_pinned.append(pinned)
        notes_oob.append(oob)
        for k in crit_cols:
            crit_cols[k].append(crit[k])

    props["n_criteria_failed"] = nfails
    props["tier"] = tiers
    props["skew_pinned"] = notes_pinned
    props["dv_out_of_band"] = notes_oob
    for k, v in crit_cols.items():
        props["pass_" + k] = v

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "lya_sample_tiered.csv")
    props.to_csv(out_csv, index=False)

    print("[CONFIG]")
    print(f"  Properties CSV : {os.path.abspath(args.properties_csv)}")
    print(f"  Catalogue      : {os.path.abspath(args.catalog)}")
    print(f"  Figure dir     : {os.path.abspath(args.figdir)}")
    print(f"  Criteria       : grade==a, FWHM>{FWHM_MIN:g}, "
          f"dv_err<{DV_ERR_MAX:g}, S/N>{SNR_MIN:g}, not(pinned|OOB)")
    print("")

    order = ["gold", "silver", "bronze", "bad"]
    for tier in order:
        tdf = props[props["tier"] == tier].sort_values(
            "delta_v_err_kms", na_position="last")
        print(f"[INFO] {tier:6s}: {len(tdf)} sources")
        if len(tdf) == 0:
            continue
        outpdf = os.path.join(args.outdir, f"lya_fits_{tier}.pdf")
        miss = build_pdf(tdf, args.figdir, outpdf, tier)
        print(f"        PDF -> {os.path.abspath(outpdf)}")
        if miss:
            print(f"        [WARN] no fit PNG for: {sorted(set(miss))}")

    print("")
    print(f"[DONE] Tier assignment CSV : {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()


"""
python group_lya_sample.py \
  --properties-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --figdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures \
  --outdir /ceph/cephfs/apatrick/P2/MUSE_catalogs
"""