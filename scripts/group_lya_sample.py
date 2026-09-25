#!/usr/bin/env python3
"""
Tier the Lya sample into gold / silver / bronze / stone / bad, using the
best-available-systemic-line Delta_v (from find_delta_v_from_best_zsys_line.py)
rather than the OIII-only version. Six pass/fail criteria per source:

  1. zsys_is_OIII        z_sys_line == 'OIII'
  2. dv_positive         delta_v_kms > 0
  3. dv_err<120          delta_v_err_kms < 120 km/s
  4. fwhm>100            Lya FWHM > 100 km/s
  5. snr>3               Lya S/N > 3
  6. not_pinned_or_oob   NOT flagged skew_pinned or dv_out_of_band (the two
                         count as one combined criterion)

Tiers by number of failed criteria (a failed Lya fit fails all six and is
automatically 'bad', regardless of count):
  gold   : 0 failures
  silver : 1 failure
  bronze : 2 failures
  stone  : 3 or 4 failures
  bad    : 5 or 6 failures, or fit_success is False

Two PDFs per tier
------------------
  lya_fits_<tier>.pdf         Contact sheet, 6 sources per page (3x2), one
                               {ID}_lya_fit.png per panel, same layout as
                               before, caption updated to the new criteria.
  lya_fits_<tier>_pairs.pdf   3 sources per page, 2 rows: the Lya fit on top,
                               and directly below it whichever systemic-line
                               fit was actually used for that source (the
                               OIII fit if z_sys_line is OIII, the Ha fit if
                               it's Ha, and so on). A source with no usable
                               systemic line shows a placeholder in the
                               bottom panel instead.

One CSV per tier
-----------------
  lya_group_<tier>.csv   ID, the six underlying values, a pass/fail column
                          per criterion, n_criteria_failed, tier, and an
                          empty 'manual' column for you to fill in by eye.

Duplicate sources are collapsed to a single row before tiering. The
'duplicate' column in grating_sources_with_zsys.csv is 0 for a source with
no pair; both members of a pair share the same nonzero group number. Within
each pair the member with the lower delta_v_err_kms is kept.

Inputs
------
  delta_v_from_best_zsys_line.csv   z_sys, z_sys_err, z_sys_line, delta_v_kms,
                                     delta_v_err_kms (the recomputed values)
  lya_properties_mc.csv             fwhm_kms, quality_reason, fit_success,
                                     ra, dec (everything the Delta_v csv
                                     doesn't carry)
  systemic_redshifts_by_JELS_ID.csv only used to look up which grating each
                                     source's chosen line was fitted in, so
                                     the pairs PDF can find the right PNG

Usage
-----
python group_lya_sample.py
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


# Criterion thresholds
DV_ERR_MAX = 120.0      # km/s
FWHM_MIN = 100.0        # km/s
SNR_MIN = 3.0

# Lines in the priority order used by find_delta_v_from_best_zsys_line.py,
# needed to look up each line's own fit-figure directory.
LINES = ["OIII", "Ha", "Hbeta", "NII", "OII"]
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"

# systemic_redshifts_by_JELS_ID.csv records the short grating code
# (G235M, G235H, G395M, G395H), but the fit-figure filenames use the full
# grating_filter string, e.g. 15479_G395H_F290LP_OIII_fit.png.
FULL_GRATING = {
    "G235M": "G235M_F170LP",
    "G235H": "G235H_F170LP",
    "G395M": "G395M_F290LP",
    "G395H": "G395H_F290LP",
}

CRITERIA = ["zsys_is_OIII", "dv_positive", "dv_err<120", "fwhm>100",
            "snr>3", "not_pinned_or_oob"]


# ============================================================
# Helpers
# ============================================================

def to_bool(val):
    """Robust bool coercion for a value that may be a real bool or a CSV
    string ('True'/'False'), unlike bool('False') which is True."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val).strip().lower() in ("true", "1")


def has_flag(reason, token):
    """True if a quality_reason string contains the given token."""
    if not isinstance(reason, str):
        return False
    return token in reason.split(";")


def load_grating_map(systemic_csv):
    """Return {(ID, line): grating} for every line in LINES."""
    sysdf = pd.read_csv(systemic_csv)
    sysdf["ID"] = sysdf["ID"].astype(int)
    out = {}
    for line in LINES:
        col = f"z_{line}_grating"
        if col not in sysdf.columns:
            continue
        for _, row in sysdf.iterrows():
            g = row[col]
            if pd.notna(g):
                out[(int(row["ID"]), line)] = g
    return out


def line_fit_png_path(line, grating_short, src_id):
    """Path to the {ID}_{grating}_{line}_fit.png figure for one line fit.

    grating_short is the short code (G395H etc) as stored in
    systemic_redshifts_by_JELS_ID.csv; it's expanded to the full
    grating_filter string used in the actual filenames.
    """
    grating = FULL_GRATING.get(grating_short, grating_short)
    fig_root = os.path.join(JWST_SPECTRA_ROOT, f"{line}_fits")
    fname = f"{src_id}_{grating}_{line}_fit.png"
    return os.path.join(fig_root, str(src_id), fname)


def load_duplicate_map(path):
    """Return {ID: duplicate_group} from a catalogue with ID and duplicate
    columns. duplicate is 0 for a source with no pair; both members of a
    pair share the same nonzero group number."""
    df = pd.read_csv(path)
    if "duplicate" not in df.columns:
        print(f"[WARN] '{path}' has no 'duplicate' column, skipping dedup.")
        return {}
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID")["duplicate"].to_dict()


def deduplicate(df, dup_map):
    """Collapse each duplicate pair (same nonzero 'duplicate' group number
    from dup_map) to the member with the lowest delta_v_err_kms. A source
    with group 0, or missing from dup_map, isn't part of any pair and
    passes through unchanged."""
    if not dup_map:
        return df, []

    out_df = df.copy()
    out_df["_dup_group"] = out_df["ID"].map(dup_map).fillna(0).astype(int)
    out_df["_err_sort"] = out_df["delta_v_err_kms"].where(
        np.isfinite(out_df["delta_v_err_kms"]), np.inf)

    dropped = []
    keep_idx = []
    for grp_val, grp in out_df.groupby("_dup_group", sort=False):
        if grp_val == 0 or len(grp) == 1:
            keep_idx.extend(grp.index.tolist())
            continue
        grp_sorted = grp.sort_values("_err_sort")
        keep = grp_sorted.index[0]
        keep_idx.append(keep)
        losers = [int(out_df.loc[i, "ID"]) for i in grp_sorted.index[1:]]
        dropped.append((int(out_df.loc[keep, "ID"]), losers))

    out = out_df.loc[keep_idx].drop(columns=["_dup_group", "_err_sort"])
    return out, dropped


# ============================================================
# Criteria and tiering
# ============================================================

def evaluate_source(row):
    """Return (n_fail, criteria_dict, pinned, oob) for one source."""
    fit_ok = to_bool(row.get("fit_success", False))

    z_sys_line = row.get("z_sys_line")
    dv = pd.to_numeric(row.get("delta_v_kms"), errors="coerce")
    dv_err = pd.to_numeric(row.get("delta_v_err_kms"), errors="coerce")
    fwhm = pd.to_numeric(row.get("fwhm_kms"), errors="coerce")
    snr = pd.to_numeric(row.get("lya_snr"), errors="coerce")
    reason = row.get("quality_reason", "")

    pinned = has_flag(reason, "skew_pinned")
    oob = has_flag(reason, "dv_out_of_band")

    crit = {
        "zsys_is_OIII": (isinstance(z_sys_line, str) and z_sys_line == "OIII"),
        "dv_positive": bool(np.isfinite(dv) and dv > 0),
        "dv_err<120": bool(np.isfinite(dv_err) and dv_err < DV_ERR_MAX),
        "fwhm>100": bool(np.isfinite(fwhm) and fwhm > FWHM_MIN),
        "snr>3": bool(np.isfinite(snr) and snr > SNR_MIN),
        "not_pinned_or_oob": (not pinned and not oob),
    }

    if not fit_ok:
        crit = {k: False for k in crit}

    n_fail = sum(1 for v in crit.values() if not v)
    return n_fail, crit, pinned, oob


def tier_from_fails(n_fail, fit_ok):
    if not fit_ok:
        return "bad"
    if n_fail == 0:
        return "gold"
    if n_fail == 1:
        return "silver"
    if n_fail == 2:
        return "bronze"
    if n_fail <= 4:
        return "stone"
    return "bad"


# ============================================================
# Plotting
# ============================================================

def caption_for(row):
    """Caption for the contact-sheet PDF panels."""
    sid = int(row["ID"])
    z_sys_line = row.get("z_sys_line")
    line_str = z_sys_line if isinstance(z_sys_line, str) else "none"
    snr = pd.to_numeric(row.get("lya_snr"), errors="coerce")
    dv = pd.to_numeric(row.get("delta_v_kms"), errors="coerce")
    dv_err = pd.to_numeric(row.get("delta_v_err_kms"), errors="coerce")
    fwhm = pd.to_numeric(row.get("fwhm_kms"), errors="coerce")

    snr_str = f"{snr:.1f}" if np.isfinite(snr) else "nan"
    dv_str = f"{dv:.0f}" if np.isfinite(dv) else "nan"
    dverr_str = f"{dv_err:.0f}" if np.isfinite(dv_err) else "nan"
    fwhm_str = f"{fwhm:.0f}" if np.isfinite(fwhm) else "nan"

    reason = row.get("quality_reason", "")
    notes = []
    if has_flag(reason, "skew_pinned"):
        notes.append("skew pinned")
    if has_flag(reason, "dv_out_of_band"):
        notes.append("dv OOB")
    note_str = ("   [" + ", ".join(notes) + "]") if notes else ""

    return (f"ID {sid}   $z_{{\\rm sys}}$: {line_str}   S/N {snr_str}   "
            f"FWHM {fwhm_str}   $\\Delta v$={dv_str}$\\pm${dverr_str} km/s"
            f"{note_str}")


def make_summary_page(pdf, tier_df, tier_name):
    """Page of z_sys, delta_v and FWHM histograms for the tier."""
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


def build_contact_sheet(tier_df, figdir, outpdf, tier_name):
    """Tile {ID}_lya_fit.png 6 per page (3x2), then a summary-histogram page."""
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
                ax.set_title(caption_for(row), fontsize=9)
                ax.axis("off")

            fig.suptitle(f"{tier_name.capitalize()} tier  |  {len(ids)} sources",
                         fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        if len(tier_df):
            make_summary_page(pdf, tier_df, tier_name)

    return missing


def build_paired_sheet(tier_df, figdir, grating_map, outpdf, tier_name):
    """3 sources per page: Lya fit on top, the chosen systemic-line fit
    directly below it."""
    ncols, nrows = 3, 2
    per_page = ncols
    ids = list(tier_df["ID"])
    missing_lya, missing_line = [], []

    with PdfPages(outpdf) as pdf:
        for start in range(0, len(ids), per_page):
            page_ids = ids[start:start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 9.5))
            axes = np.atleast_2d(axes)
            for ax in axes.ravel():
                ax.axis("off")

            for col, sid in enumerate(page_ids):
                row = tier_df[tier_df["ID"] == sid].iloc[0]
                z_sys_line = row.get("z_sys_line")

                # Top: Lya fit
                ax_top = axes[0, col]
                lya_png = os.path.join(figdir, f"{sid}_lya_fit.png")
                if os.path.exists(lya_png):
                    ax_top.imshow(mpimg.imread(lya_png))
                else:
                    missing_lya.append(sid)
                    ax_top.text(0.5, 0.5, f"ID {sid}\n(no Lya fit PNG)",
                                ha="center", va="center", fontsize=11,
                                transform=ax_top.transAxes)
                ax_top.set_title(caption_for(row), fontsize=8)
                ax_top.axis("off")

                # Bottom: the systemic-line fit actually used
                ax_bot = axes[1, col]
                if isinstance(z_sys_line, str):
                    grating = grating_map.get((sid, z_sys_line))
                    line_png = (line_fit_png_path(z_sys_line, grating, sid)
                                if grating else None)
                    if line_png and os.path.exists(line_png):
                        ax_bot.imshow(mpimg.imread(line_png))
                        ax_bot.set_title(f"{z_sys_line} fit", fontsize=9)
                    else:
                        missing_line.append(sid)
                        ax_bot.text(0.5, 0.5,
                                    f"{z_sys_line} fit PNG not found",
                                    ha="center", va="center", fontsize=11,
                                    transform=ax_bot.transAxes)
                else:
                    ax_bot.text(0.5, 0.5, "no usable systemic line",
                                ha="center", va="center", fontsize=11,
                                transform=ax_bot.transAxes)
                ax_bot.axis("off")

            fig.suptitle(f"{tier_name.capitalize()} tier, paired  |  "
                         f"{len(ids)} sources", fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return missing_lya, missing_line


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Tier Lya sample (gold/silver/bronze/stone/bad) and "
                    "build the contact-sheet and paired-fit PDFs.")
    p.add_argument("--delta-v-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/delta_v_from_best_zsys_line.csv",
                   help="Recomputed z_sys/delta_v, from "
                        "find_delta_v_from_best_zsys_line.py.")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="fwhm_kms, quality_reason, fit_success, ra, dec.")
    p.add_argument("--systemic-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv",
                   help="Used only to look up the grating of the chosen "
                        "systemic line, for the paired-fit PDF.")
    p.add_argument("--grating-sources-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv",
                   help="Source of the 'duplicate' column: 0 for a source "
                        "with no pair, a shared nonzero group number for "
                        "both members of a pair.")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures",
                   help="Where {ID}_lya_fit.png live.")
    p.add_argument("--outdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs")
    args = p.parse_args()

    delta_v_path = os.path.abspath(args.delta_v_csv)
    properties_path = os.path.abspath(args.properties_csv)
    systemic_path = os.path.abspath(args.systemic_csv)
    grating_sources_path = os.path.abspath(args.grating_sources_csv)
    figdir = os.path.abspath(args.figdir)
    outdir = os.path.abspath(args.outdir)

    print("[CONFIG]")
    print(f"  delta_v csv     {delta_v_path}")
    print(f"  properties csv  {properties_path}")
    print(f"  systemic csv    {systemic_path}")
    print(f"  duplicate csv   {grating_sources_path}")
    print(f"  figure dir      {figdir}")
    print(f"  output dir      {outdir}")
    print(f"  criteria        dv_err<{DV_ERR_MAX:g}, fwhm>{FWHM_MIN:g}, "
          f"snr>{SNR_MIN:g}, zsys==OIII, dv>0, not(pinned|OOB)")
    print("")

    props = pd.read_csv(properties_path)
    for col in ("ID", "fwhm_kms", "lya_snr", "quality_reason", "fit_success",
                "ra", "dec"):
        if col not in props.columns:
            raise KeyError(
                f"'{col}' not in {properties_path}. Available: {list(props.columns)}")
    props["ID"] = props["ID"].astype(int)

    dv = pd.read_csv(delta_v_path)
    dv_cols = ["ID", "z_sys", "z_sys_err", "z_sys_snr", "z_sys_line",
               "delta_v_kms", "delta_v_err_lya_kms", "delta_v_err_sys_kms",
               "delta_v_err_kms"]
    missing_dv_cols = [c for c in dv_cols if c not in dv.columns]
    if missing_dv_cols:
        raise KeyError(
            f"'{missing_dv_cols}' not in {delta_v_path}. Available: {list(dv.columns)}")
    dv["ID"] = dv["ID"].astype(int)
    dv = dv[dv_cols]

    # dv's recomputed z_sys/delta_v columns replace any stale ones in props.
    props = props.drop(columns=[c for c in dv_cols if c != "ID" and c in props.columns])
    merged = props.merge(dv, on="ID", how="left")

    grating_map = load_grating_map(systemic_path)
    dup_map = load_duplicate_map(grating_sources_path)

    # Deduplicate before tiering
    merged, dropped = deduplicate(merged, dup_map)
    if dropped:
        print("[INFO] Duplicate sources collapsed (kept, dropped):")
        for kept, losers in dropped:
            print(f"   kept {kept}, dropped {losers}")
        print("")

    # Evaluate every source
    tiers, nfails, notes_pinned, notes_oob = [], [], [], []
    crit_cols = {k: [] for k in CRITERIA}
    for _, row in merged.iterrows():
        n_fail, crit, pinned, oob = evaluate_source(row)
        tier = tier_from_fails(n_fail, to_bool(row.get("fit_success", False)))
        tiers.append(tier)
        nfails.append(n_fail)
        notes_pinned.append(pinned)
        notes_oob.append(oob)
        for k in crit_cols:
            crit_cols[k].append(crit[k])

    merged["n_criteria_failed"] = nfails
    merged["tier"] = tiers
    merged["skew_pinned"] = notes_pinned
    merged["dv_out_of_band"] = notes_oob
    for k, v in crit_cols.items():
        merged["pass_" + k] = v

    os.makedirs(outdir, exist_ok=True)

    order = ["gold", "silver", "bronze", "stone", "bad"]
    csv_value_cols = ["ID", "z_sys_line", "delta_v_kms", "delta_v_err_kms",
                      "fwhm_kms", "lya_snr", "skew_pinned", "dv_out_of_band"]
    csv_pass_cols = ["pass_" + k for k in CRITERIA]

    for tier in order:
        tdf = merged[merged["tier"] == tier].sort_values(
            "delta_v_err_kms", na_position="last")
        print(f"[INFO] {tier:6s}: {len(tdf)} sources")
        if len(tdf) == 0:
            continue

        # Per-tier CSV
        tier_csv = tdf[csv_value_cols + csv_pass_cols + ["n_criteria_failed"]].copy()
        tier_csv["tier"] = tier
        tier_csv["manual"] = ""
        out_csv = os.path.join(outdir, f"lya_group_{tier}.csv")
        tier_csv.to_csv(out_csv, index=False)
        print(f"        csv    -> {out_csv}")

        # Contact-sheet PDF
        outpdf = os.path.join(outdir, f"lya_fits_{tier}.pdf")
        miss = build_contact_sheet(tdf, figdir, outpdf, tier)
        print(f"        pdf    -> {outpdf}")
        if miss:
            print(f"        [WARN] no Lya fit PNG for: {sorted(set(miss))}")

        # Paired-fit PDF
        outpdf_pairs = os.path.join(outdir, f"lya_fits_{tier}_pairs.pdf")
        miss_lya, miss_line = build_paired_sheet(
            tdf, figdir, grating_map, outpdf_pairs, tier)
        print(f"        pdf    -> {outpdf_pairs}")
        if miss_line:
            print(f"        [WARN] no systemic-line fit PNG for: "
                  f"{sorted(set(miss_line))}")

    print("")
    print("[DONE]")


if __name__ == "__main__":
    main()