#!/usr/bin/env python
"""
add_detection_flags.py

Add Lya detection columns to the P2 grating source catalogues and report
detection counts against the false-positive thresholds.

For both:
  /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv
  /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID_good.csv

it adds three columns:
  - peak_snr   : the real-source peak S/N from lya_sliding_snr_grating.csv,
                 matched on ID. NaN if the source has no measured value.
  - over_98    : 1 if peak_snr >= the pooled 98th-percentile false-positive S/N,
                 else 0. Sources with no peak_snr are 0.
  - over_99.5  : same against the 99.5th percentile.

The percentile thresholds are read from the summary CSV written by
false_pos_percentiles.py (--out-csv), columns pct_98.0 and pct_99.5.

It then prints:
  1. the number of sources in the FULL catalogue with over_98 == 1
  2. the number of sources that are in_muse, not AO_block, and deduplicated
     (one per duplicate pair)
  3. how many of those deduplicated sources have over_98 == 1
"""

import argparse
import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FULL_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv"
GOOD_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID_good.csv"
SNR_CSV = "/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv"
PCT_CSV = "/ceph/cephfs/apatrick/P2/MUSE_catalogs/false_pos_snr_percentiles.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Add Lya detection flags to the grating catalogues."
    )
    p.add_argument("--full-csv", default=FULL_CSV)
    p.add_argument("--good-csv", default=GOOD_CSV)
    p.add_argument("--snr-csv", default=SNR_CSV,
                   help="Real-source sliding-SNR CSV with ID and peak_snr")
    p.add_argument("--pct-csv", default=PCT_CSV,
                   help="Percentile summary CSV with pct_98.0 and pct_99.5")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--snr-col", default="peak_snr")
    return p.parse_args()


def load_thresholds(pct_path):
    if not os.path.exists(pct_path):
        raise SystemExit(
            f"Percentile summary not found: {pct_path}\n"
            f"Run false_pos_percentiles.py with --out-csv first."
        )
    pct = pd.read_csv(pct_path)
    row = pct.iloc[0]
    # Column names as written by false_pos_percentiles.py
    for c98 in ("pct_98.0", "pct_98"):
        if c98 in pct.columns:
            thr98 = float(row[c98])
            break
    else:
        raise SystemExit(f"No pct_98.0 column in {pct_path} (has {list(pct.columns)})")
    for c995 in ("pct_99.5", "pct_99p5"):
        if c995 in pct.columns:
            thr995 = float(row[c995])
            break
    else:
        raise SystemExit(f"No pct_99.5 column in {pct_path} (has {list(pct.columns)})")
    return thr98, thr995


def load_snr_map(snr_path, id_col, snr_col):
    if not os.path.exists(snr_path):
        raise SystemExit(f"Sliding-SNR CSV not found: {snr_path}")
    snr = pd.read_csv(snr_path)
    # The sliding-SNR CSV uses ID and peak_snr
    key = id_col if id_col in snr.columns else "ID"
    if key not in snr.columns or snr_col not in snr.columns:
        raise SystemExit(
            f"{snr_path} must contain '{key}' and '{snr_col}' "
            f"(has {list(snr.columns)})"
        )
    snr[key] = snr[key].astype("Int64")
    # Map ID -> peak_snr
    return dict(zip(snr[key].astype("Int64"), pd.to_numeric(snr[snr_col], errors="coerce")))


def add_flags(df, id_col, snr_map, thr98, thr995):
    ids = df[id_col].astype("Int64")
    peak = ids.map(snr_map).astype(float)

    over98 = np.where(np.isfinite(peak) & (peak >= thr98), 1, 0)
    over995 = np.where(np.isfinite(peak) & (peak >= thr995), 1, 0)

    df = df.copy()
    df["peak_snr"] = peak.values
    df["over_98"] = over98
    df["over_99.5"] = over995
    return df


def dedup_first_per_pair(df):
    """Keep all rows with duplicate == 0, plus the first row of each numbered
    duplicate group (duplicate == 1, 2, 3, ...)."""
    if "duplicate" not in df.columns:
        raise SystemExit("No 'duplicate' column present for deduplication.")
    dup = pd.to_numeric(df["duplicate"], errors="coerce").fillna(0).astype(int)
    keep = dup == 0
    # First occurrence of each non-zero group
    seen = set()
    first_mask = []
    for d in dup:
        if d == 0:
            first_mask.append(False)
        elif d not in seen:
            seen.add(d)
            first_mask.append(True)
        else:
            first_mask.append(False)
    first_mask = np.array(first_mask)
    return df[keep.values | first_mask]


def main():
    args = parse_args()

    thr98, thr995 = load_thresholds(os.path.abspath(args.pct_csv))
    snr_map = load_snr_map(os.path.abspath(args.snr_csv), args.id_col, args.snr_col)

    print("[THRESHOLDS]")
    print(f"  98.0th percentile S/N = {thr98:.3f}")
    print(f"  99.5th percentile S/N = {thr995:.3f}")
    print("")

    # Process and overwrite both catalogues
    outputs = {}
    for label, path in (("FULL", args.full_csv), ("GOOD", args.good_csv)):
        p = os.path.abspath(path)
        if not os.path.exists(p):
            raise SystemExit(f"Catalogue not found: {p}")
        df = pd.read_csv(p)
        if args.id_col not in df.columns:
            raise SystemExit(f"No '{args.id_col}' column in {p}")
        df[args.id_col] = df[args.id_col].astype("Int64")
        df = add_flags(df, args.id_col, snr_map, thr98, thr995)
        df.to_csv(p, index=False)
        outputs[label] = df
        n_snr = int(np.isfinite(df["peak_snr"]).sum())
        print(f"[WRITE] {label}: added columns to {p}")
        print(f"         {len(df)} rows, {n_snr} with a peak_snr value")
    print("")

    full = outputs["FULL"]

    # ---- Report 1: full catalogue over_98 ----
    n_full_over98 = int((full["over_98"] == 1).sum())
    print("[COUNTS]")
    print(f"  Full catalogue sources with over_98 == 1 : {n_full_over98}")

    # ---- Report 2: clean subsample (in_muse, not AO_block, deduplicated) ----
    sub = full
    missing = [c for c in ("in_muse", "AO_block", "duplicate") if c not in sub.columns]
    if missing:
        raise SystemExit(
            f"Full catalogue missing columns needed for the clean subsample: "
            f"{missing}"
        )
    in_muse = pd.to_numeric(sub["in_muse"], errors="coerce").fillna(0).astype(int) == 1
    not_ao = pd.to_numeric(sub["AO_block"], errors="coerce").fillna(0).astype(int) == 0
    sub_clean = sub[in_muse & not_ao]
    sub_clean = dedup_first_per_pair(sub_clean)

    n_clean = len(sub_clean)
    print(f"  In-MUSE, non-AO, deduplicated sources     : {n_clean}")

    # ---- Report 3: clean subsample over_98 ----
    n_clean_over98 = int((sub_clean["over_98"] == 1).sum())
    print(f"  Of those, with over_98 == 1               : {n_clean_over98}")

    # Bonus: same for 99.5 to save a rerun
    n_full_over995 = int((full["over_99.5"] == 1).sum())
    n_clean_over995 = int((sub_clean["over_99.5"] == 1).sum())
    print("")
    print("[ALSO, for 99.5]")
    print(f"  Full catalogue over_99.5                  : {n_full_over995}")
    print(f"  Clean subsample over_99.5                 : {n_clean_over995}")


if __name__ == "__main__":
    main()


"""
run after the false-positive steps and after false_pos_percentiles.py --out-csv
python add_detection_flags.py


python add_detection_flags.py
"""