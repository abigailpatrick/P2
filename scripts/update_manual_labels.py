#!/usr/bin/env python3
"""
Create or extend the hand-edited label file lya_manual_labels.csv.

This file is the single home for the a / b / c / d labels given by eye. No
other script writes to it. group_lya_sample.py, group_by_manual.py and
build_master_catalog.py all read the labels from here.

Columns
-------
  ID                   JELS ID
  manual               a, b, c or d (blank until labelled)
  tier_when_labelled   the automatic tier the source had when labelled
  note                 free text, optional

First run (file does not exist)
    The labels are seeded from the 'manual' column of the existing
    lya_group_<tier>.csv files, so nothing already labelled is lost.

Later runs (file exists)
    Existing rows are never changed. Any ID in the current tier CSVs that is
    not yet in the file is appended with a blank label, ready to fill in.
    Labelled IDs that are no longer in any tier CSV are listed but kept.

Usage
-----
python update_manual_labels.py              # seed, or add new IDs
python update_manual_labels.py --dry-run    # report only
"""

import argparse
import os

import pandas as pd

import p2_common as pc

TIERS = ["gold", "silver", "bronze", "stone", "bad"]
COLS = ["ID", "manual", "tier_when_labelled", "note"]


def parse_args():
    p = argparse.ArgumentParser(description="Seed or extend lya_manual_labels.csv.")
    p.add_argument("--p2-root", default=pc.P2_ROOT)
    p.add_argument("--tier-dir", default=None,
                   help="Default <p2-root>/MUSE_catalogs")
    p.add_argument("--labels-csv", default=None,
                   help="Default <p2-root>/MUSE_catalogs/lya_manual_labels.csv")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.tier_dir = os.path.abspath(a.tier_dir or os.path.join(a.p2_root, "MUSE_catalogs"))
    a.labels_csv = os.path.abspath(a.labels_csv or os.path.join(a.tier_dir, "lya_manual_labels.csv"))
    return a


def read_tiers(tier_dir):
    parts = []
    for t in TIERS:
        path = os.path.join(tier_dir, f"lya_group_{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["tier"] = t
        if "manual" not in df.columns:
            df["manual"] = ""
        parts.append(df[["ID", "tier", "manual"]])
        print(f"  read   {path}  ({len(df)} rows)")
    if not parts:
        raise SystemExit(f"No lya_group_<tier>.csv files in {tier_dir}")
    out = pd.concat(parts, ignore_index=True)
    out["ID"] = out["ID"].astype(int)
    out["manual"] = out["manual"].fillna("").astype(str).str.strip().str.lower()
    return out


def main():
    a = parse_args()
    print("[CONFIG]")
    print(f"  tier dir     {a.tier_dir}")
    print(f"  labels csv   {a.labels_csv}")
    print("")
    tiers = read_tiers(a.tier_dir)

    if not os.path.exists(a.labels_csv):
        labels = pd.DataFrame({
            "ID": tiers["ID"],
            "manual": tiers["manual"],
            "tier_when_labelled": tiers["tier"].where(tiers["manual"] != "", ""),
            "note": "",
        })[COLS].sort_values("ID")
        n_lab = int((labels["manual"] != "").sum())
        print(f"\n[SEED] {len(labels)} IDs, {n_lab} already labelled, "
              f"{len(labels) - n_lab} blank")
    else:
        labels = pd.read_csv(a.labels_csv, dtype={"manual": str, "note": str,
                                                  "tier_when_labelled": str})
        labels["ID"] = labels["ID"].astype(int)
        for c in COLS:
            if c not in labels.columns:
                labels[c] = ""
        new = tiers[~tiers["ID"].isin(labels["ID"])]
        gone = labels[~labels["ID"].isin(tiers["ID"])]
        print(f"\n[UPDATE] {len(labels)} IDs already in the file")
        print(f"  new IDs added with a blank label  {len(new)}  {new['ID'].tolist()}")
        print(f"  IDs no longer in any tier CSV     {len(gone)}  {gone['ID'].tolist()} (kept)")
        add = pd.DataFrame({"ID": new["ID"], "manual": "", "tier_when_labelled": "", "note": ""})
        labels = pd.concat([labels[COLS], add], ignore_index=True).sort_values("ID")

    blank = labels.loc[labels["manual"].fillna("") == "", "ID"].tolist()
    if blank:
        print(f"  still to label by eye             {len(blank)}  {blank}")
    if a.dry_run:
        print("\n[DRY RUN] nothing written")
        return
    os.makedirs(os.path.dirname(a.labels_csv), exist_ok=True)
    labels.to_csv(a.labels_csv, index=False)
    print(f"\nwritten  {a.labels_csv}")


if __name__ == "__main__":
    main()
