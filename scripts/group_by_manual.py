#!/usr/bin/env python3
"""
Regroup the Lya sample by the manual a / b / c / d labels written by eye into
the 'manual' column of the tier CSVs from group_lya_sample.py.

Reads
-----
  lya_group_<tier>.csv   for tier in gold, silver, bronze, stone, bad. Each
                         row must carry a manual label a, b, c or d.
  delta_v_from_best_zsys_line.csv
                         z_sys, z_sys_line, delta_v_kms and its errors,
                         merged on ID.
  lya_properties_mc.csv  fwhm_int_kms, fwhm_obs_kms, fwhm_lsf_kms, resolved,
                         skew_at_max, lya_snr, quality_reason, fit_success,
                         merged on ID.

Only the ID, tier and manual columns are taken from the tier CSVs. Every
measured value is refreshed from the current pipeline outputs above, and the
pass/fail criteria are re-evaluated with group_lya_sample.py's own
evaluate_source. So after rerunning the LSF-corrected fit, MC and
find_delta_v_from_best_zsys_line.py, this script picks up the corrected
FWHM and Delta_v while keeping your by-eye labels. 'tier' is the tier the
source had when you labelled it. 'tier_now' is the tier the current values
would give.
  systemic_redshifts_by_JELS_ID.csv
                         grating of each source's chosen systemic line, so the
                         pairs PDF can find the right line-fit PNG.

Writes, per label, to --outdir (the same place as the tier outputs)
---------------------------------------------------------------------
  lya_group_<label>.csv        same columns as the tier CSVs (tier and manual
                               kept, so you can see where each source came
                               from)
  lya_fits_<label>.pdf         contact sheet, 6 per page, plus a summary page
  lya_fits_<label>_pairs.pdf   Lya fit on top, systemic-line fit below

The PDF builders and captions are imported from group_lya_sample.py, so the
layout matches the tier PDFs exactly. Within each label, sources are ordered by
original tier (gold first) and then by delta_v_err_kms.

Note: group_lya_sample.py now carries the manual labels over by ID when it
rewrites lya_group_<tier>.csv, but backing up the labelled CSVs first is still
sensible.

Usage
-----
python group_lya_by_manual.py
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

TIERS = ["gold", "silver", "bronze", "stone", "bad"]
LABELS = ["a", "b", "c", "d"]


def load_grouping_module(grouping_dir):
    """Import group_lya_sample.py so its PDF builders can be reused."""
    path = os.path.join(grouping_dir, "group_lya_sample.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"group_lya_sample.py not found at {path}")
    spec = importlib.util.spec_from_file_location("group_lya_sample", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tier_csvs(tier_dir):
    """Concatenate every lya_group_<tier>.csv that exists."""
    parts = []
    for tier in TIERS:
        path = os.path.join(tier_dir, f"lya_group_{tier}.csv")
        if not os.path.exists(path):
            print(f"[WARN] no tier CSV for {tier}: {path}")
            continue
        df = pd.read_csv(path)
        if "manual" not in df.columns:
            raise KeyError(f"'manual' column missing from {path}")
        df["tier"] = tier
        parts.append(df)
        print(f"  read  {path}  ({len(df)} rows)")
    if not parts:
        raise SystemExit(f"No lya_group_<tier>.csv files found in {tier_dir}")
    out = pd.concat(parts, ignore_index=True)
    out["ID"] = out["ID"].astype(int)

    dup_ids = out.loc[out["ID"].duplicated(), "ID"].tolist()
    if dup_ids:
        raise ValueError(f"IDs appear in more than one tier CSV: {dup_ids}")
    return out


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Regroup the Lya sample by the manual a/b/c/d labels.")
    p.add_argument("--tier-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs",
                   help="Directory holding the labelled lya_group_<tier>.csv files.")
    p.add_argument("--delta-v-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/delta_v_from_best_zsys_line.csv")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv")
    p.add_argument("--systemic-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv")
    p.add_argument("--figdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures",
                   help="Where {ID}_lya_fit.png live.")
    p.add_argument("--outdir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs")
    p.add_argument("--grouping-dir", default=script_dir,
                   help="Directory containing group_lya_sample.py.")
    args = p.parse_args()

    tier_dir = os.path.abspath(args.tier_dir)
    delta_v_path = os.path.abspath(args.delta_v_csv)
    properties_path = os.path.abspath(args.properties_csv)
    systemic_path = os.path.abspath(args.systemic_csv)
    figdir = os.path.abspath(args.figdir)
    outdir = os.path.abspath(args.outdir)

    print("[CONFIG]")
    print(f"  tier csv dir    {tier_dir}")
    print(f"  delta_v csv     {delta_v_path}")
    print(f"  properties csv  {properties_path}")
    print(f"  systemic csv    {systemic_path}")
    print(f"  figure dir      {figdir}")
    print(f"  output dir      {outdir}")
    print("")

    gls = load_grouping_module(os.path.abspath(args.grouping_dir))

    print("[READ]")
    df = read_tier_csvs(tier_dir)
    print("")

    # Clean labels and report anything unexpected
    df["manual"] = df["manual"].astype(str).str.strip().str.lower()
    bad = df[~df["manual"].isin(LABELS)]
    if len(bad):
        print(f"[WARN] {len(bad)} sources without a valid a/b/c/d label, left out:")
        for _, r in bad.iterrows():
            print(f"        ID {int(r['ID'])}  tier {r['tier']}  manual '{r['manual']}'")
        print("")
    df = df[df["manual"].isin(LABELS)].copy()

    # Keep only ID / tier / manual from the tier CSVs, refresh every value
    # from the current pipeline outputs (LSF-corrected FWHM, current Delta_v).
    df = df[["ID", "tier", "manual"]].copy()
    dv_cols = ["ID", "z_sys", "z_sys_line", "delta_v_kms",
               "delta_v_err_kms"]
    dv = pd.read_csv(delta_v_path)
    missing = [c for c in dv_cols if c not in dv.columns]
    if missing:
        raise KeyError(f"{missing} not in {delta_v_path}")
    dv = dv[dv_cols]
    dv["ID"] = dv["ID"].astype(int)
    prop_cols = ["ID", "fwhm_int_kms", "fwhm_obs_kms", "fwhm_lsf_kms",
                 "resolved", "skew_at_max", "lya_snr", "quality_reason",
                 "fit_success"]
    props = pd.read_csv(properties_path)
    missing = [c for c in prop_cols if c not in props.columns]
    if missing:
        raise KeyError(f"{missing} not in {properties_path}. Rerun the LSF "
                       f"version of fit_lya_properties_grating.py and "
                       f"mc_lya_errors_grating.py first.")
    props = props[prop_cols]
    props["ID"] = props["ID"].astype(int)
    df = df.merge(dv, on="ID", how="left").merge(props, on="ID", how="left")

    # Re-evaluate the criteria on the refreshed values
    pass_cols = {k: [] for k in gls.CRITERIA}
    nfail, oobs, tier_now = [], [], []
    for _, row in df.iterrows():
        n, crit, _, oob = gls.evaluate_source(row)
        nfail.append(n)
        oobs.append(oob)
        tier_now.append(gls.tier_from_fails(n, gls.to_bool(row.get("fit_success", False))))
        for k in pass_cols:
            pass_cols[k].append(crit[k])
    df["dv_out_of_band"] = oobs
    for k, v in pass_cols.items():
        df["pass_" + k] = v
    df["n_criteria_failed"] = nfail
    df["tier_now"] = tier_now

    grating_map = gls.load_grating_map(systemic_path)

    # Order within each label: original tier, then delta_v error
    df["_tier_rank"] = df["tier"].map({t: i for i, t in enumerate(TIERS)})
    df["_err_sort"] = pd.to_numeric(df["delta_v_err_kms"], errors="coerce")
    df["_err_sort"] = df["_err_sort"].where(np.isfinite(df["_err_sort"]), np.inf)

    # Output columns match the tier CSVs written by group_lya_sample.py
    tier_cols = (gls.CSV_VALUE_COLS + ["pass_" + k for k in gls.CRITERIA]
                 + ["n_criteria_failed", "tier", "tier_now", "manual", "z_sys"])

    os.makedirs(outdir, exist_ok=True)
    written = []

    for label in LABELS:
        ldf = df[df["manual"] == label].sort_values(["_tier_rank", "_err_sort"])
        breakdown = ldf["tier"].value_counts().reindex(TIERS).dropna().astype(int)
        breakdown_str = ", ".join(f"{t} {n}" for t, n in breakdown.items())
        print(f"[INFO] {label}: {len(ldf)} sources  ({breakdown_str})")
        if len(ldf) == 0:
            continue

        out_csv = os.path.join(outdir, f"lya_group_{label}.csv")
        ldf[[c for c in tier_cols if c in ldf.columns]].to_csv(out_csv, index=False)
        print(f"        csv    -> {out_csv}")
        written.append(out_csv)

        outpdf = os.path.join(outdir, f"lya_fits_{label}.pdf")
        miss = gls.build_contact_sheet(ldf, figdir, outpdf, label)
        print(f"        pdf    -> {outpdf}")
        written.append(outpdf)
        if miss:
            print(f"        [WARN] no Lya fit PNG for: {sorted(set(miss))}")

        outpdf_pairs = os.path.join(outdir, f"lya_fits_{label}_pairs.pdf")
        _, miss_line = gls.build_paired_sheet(
            ldf, figdir, grating_map, outpdf_pairs, label)
        print(f"        pdf    -> {outpdf_pairs}")
        written.append(outpdf_pairs)
        if miss_line:
            print(f"        [WARN] no systemic-line fit PNG for: "
                  f"{sorted(set(miss_line))}")

    print("")
    print("[DONE] files written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()