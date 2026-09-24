#!/usr/bin/env python
"""Merge per-line systemic redshift results into one table, one row per
source.

For each source and each of OIII, Hbeta, Ha, NII, OII, picks the grating
measurement with the highest LiMe snr_line among gratings flagged
successful in that line's own summary CSV. A source with no successful fit
for a line is left blank for that line's columns.

Reads (written by lime_OIII_jointfit.py and lime_lines_jointfit.py)
-----------------------------------------------------------------------
  OIII_results_by_JELS_ID.csv,    OIII_summary_by_JELS_ID.csv
  Hbeta_results_by_JELS_ID.csv,   Hbeta_summary_by_JELS_ID.csv
  Ha_results_by_JELS_ID.csv,      Ha_summary_by_JELS_ID.csv
  NII_results_by_JELS_ID.csv,     NII_summary_by_JELS_ID.csv
  OII_results_by_JELS_ID.csv,     OII_summary_by_JELS_ID.csv

Writes
------
  systemic_redshifts_by_JELS_ID.csv

Per line, four columns: z_<line>, z_<line>_err, z_<line>_snr, z_<line>_grating.
The _snr and _grating columns are provenance, kept so you can see which
grating a value came from; drop them at the merge-into-master-catalogue
stage if you only want z and z_err.

Usage
-----
python merge_systemic_redshifts.py
"""

import os

import numpy as np
import pandas as pd

CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
OUT_CSV = os.path.join(CATALOG_DIR, "systemic_redshifts_by_JELS_ID.csv")

LINES = ["OIII", "Hbeta", "Ha", "NII", "OII"]
GRATINGS = ["G235M", "G235H", "G395M", "G395H"]


def paths(name):
    return (os.path.join(CATALOG_DIR, f"{name}_results_by_JELS_ID.csv"),
            os.path.join(CATALOG_DIR, f"{name}_summary_by_JELS_ID.csv"))


def best_per_source(name):
    """Return {ID: (z, z_err, snr, grating)}, best grating by S/N."""
    res_path, sum_path = paths(name)
    if not (os.path.exists(res_path) and os.path.exists(sum_path)):
        print(f"missing CSVs for {name}, skipping")
        return {}

    res = pd.read_csv(res_path).set_index("ID")
    summ = pd.read_csv(sum_path).set_index("ID")

    out = {}
    for src_id in res.index:
        best = None
        for gr in GRATINGS:
            success_col = f"{name}_{gr}_success"
            if src_id not in summ.index or success_col not in summ.columns:
                continue

            val = summ.loc[src_id, success_col]
            if pd.isna(val) or not bool(val):
                continue

            z_col = f"z_{name}_{gr}"
            err_col = f"z_{name}_{gr}_err"
            snr_col = f"z_{name}_{gr}_snr"
            if not all(c in res.columns for c in (z_col, err_col, snr_col)):
                continue

            row = res.loc[src_id]
            if pd.isna(row[z_col]) or pd.isna(row[snr_col]):
                continue

            snr = float(row[snr_col])
            if best is None or snr > best[2]:
                best = (float(row[z_col]), float(row[err_col]), snr, gr)

        if best is not None:
            out[src_id] = best
    return out


def main():
    print(f"catalogue dir  {CATALOG_DIR}")

    per_line = {}
    all_ids = set()
    for name in LINES:
        res_path, sum_path = paths(name)
        print(f"reading  {res_path}")
        print(f"reading  {sum_path}")
        best = best_per_source(name)
        per_line[name] = best
        all_ids |= set(best.keys())

    rows = []
    for src_id in sorted(all_ids):
        row = {"ID": src_id}
        for name in LINES:
            best = per_line[name].get(src_id)
            if best is not None:
                z, z_err, snr, gr = best
                row[f"z_{name}"] = z
                row[f"z_{name}_err"] = z_err
                row[f"z_{name}_snr"] = snr
                row[f"z_{name}_grating"] = gr
        rows.append(row)

    df = pd.DataFrame(rows)

    cols = ["ID"]
    for name in LINES:
        cols += [f"z_{name}", f"z_{name}_err", f"z_{name}_snr", f"z_{name}_grating"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.reindex(columns=cols).sort_values("ID")

    df.to_csv(OUT_CSV, index=False)

    print("")
    print(f"sources with >=1 line measured  {len(df)}")
    for name in LINES:
        n = int(df[f"z_{name}"].notna().sum())
        print(f"  {name:<6} {n}")
    print(f"written  {OUT_CSV}")


if __name__ == "__main__":
    main()