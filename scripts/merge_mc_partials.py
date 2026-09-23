#!/usr/bin/env python3
"""
Merge the per-source MC partial CSVs (written by mc_lya_errors_grating.py in
--row-index mode, one per SLURM array task) into the final properties table.

Each partial holds the MC error columns for one source. This concatenates them,
joins them onto the science properties CSV by ID, and writes the merged
lya_properties_mc.csv, exactly matching what a single-node MC run produces.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Merge MC partial CSVs.")
    p.add_argument("--partials-dir", required=True,
                   help="Directory holding the per-source partial CSVs.")
    p.add_argument("--partials-glob", default="mc_row*.csv",
                   help="Glob for the partial files (default mc_row*.csv).")
    p.add_argument("--properties-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv",
                   help="Science properties CSV to join the MC columns onto.")
    p.add_argument("--out-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="Final merged output.")
    args = p.parse_args()

    pattern = os.path.join(args.partials_dir, args.partials_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No partials matched {pattern}")

    print(f"[INFO] merging {len(files)} partial files from {args.partials_dir}")

    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"[WARN] empty partial, skipping: {f}")
            continue
        if len(d):
            parts.append(d)

    if not parts:
        raise SystemExit("All partials were empty, nothing to merge.")

    mc = pd.concat(parts, ignore_index=True)

    # Guard against a source appearing twice (e.g. a task re-run); keep the last
    before = len(mc)
    mc = mc.drop_duplicates(subset="ID", keep="last")
    if len(mc) != before:
        print(f"[INFO] dropped {before - len(mc)} duplicate partial rows")

    props = pd.read_csv(args.properties_csv)
    props["ID"] = props["ID"].astype(int)
    mc["ID"] = mc["ID"].astype(int)

    merged = props.merge(mc, on="ID", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    n_expected = int(props["fit_success"].astype(str).str.lower()
                     .isin(["true", "1"]).sum())
    n_have = int(np.isfinite(
        pd.to_numeric(merged.get("delta_v_err_kms"), errors="coerce")).sum())
    print(f"[INFO] fitted sources expecting MC errors : {n_expected}")
    print(f"[INFO] sources with a finite delta_v_err  : {n_have}")
    if n_have < n_expected:
        missing = merged.loc[
            props["fit_success"].astype(str).str.lower().isin(["true", "1"])
            & ~np.isfinite(pd.to_numeric(merged["delta_v_err_kms"],
                                         errors="coerce")), "ID"].tolist()
        print(f"[WARN] {n_expected - n_have} fitted sources have no MC error "
              f"(array task may have failed): {sorted(missing)}")
    print(f"[DONE] written {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()