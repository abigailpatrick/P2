#!/usr/bin/env python3
"""
merge_mc_partials.py

Join the per-source Monte Carlo results, MUSE_subcubes/<ID>/<ID>_lya_mc.csv
from mc_lya_errors_grating.py, onto MUSE_catalogs/lya_fits.csv and write
MUSE_catalogs/lya_fits_mc.csv. Use lya_fits_mc.csv for science.

Each MC file stores the peak of its unperturbed refit (lya_peak_repeat). If
that no longer matches lya_peak_wave in lya_fits.csv, the fit has been rerun
since the MC, and the source is listed as stale so its MC can be redone.

mc_lya_errors_grating.py runs this itself at the end of a run without
--task-index. After a slurm array, run it on its own.

Usage
-----
python merge_mc_partials.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc

FITS_CSV = f"{pc.MUSE_CAT_DIR}/lya_fits.csv"


def merge(fits_csv=FITS_CSV, subcube_dir=pc.MUSE_SUBCUBE_DIR, out_csv=None):
    out_csv = out_csv or fits_csv.replace(".csv", "_mc.csv")
    print(f"Merging MC results onto {os.path.abspath(fits_csv)}")
    props = pd.read_csv(fits_csv)
    props["ID"] = props["ID"].astype(int)
    ok = props["fit_success"].astype(str).str.lower().isin(["true", "1"])

    parts, missing, stale = [], [], []
    for sid, good, peak in zip(props["ID"], ok, props["lya_peak_wave"]):
        if not good:
            continue
        path = pc.muse_path(sid, "mc", subcube_dir)
        if not os.path.exists(path):
            missing.append(sid)
            continue
        d = pd.read_csv(path)
        rep = float(d["lya_peak_repeat"].iloc[0])
        if np.isfinite(rep) and np.isfinite(peak) and abs(rep - peak) > 1e-3:
            stale.append(sid)
            continue
        parts.append(d)

    mc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"ID": []})
    merged = props.merge(mc, on="ID", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    merged.to_csv(out_csv, index=False)

    print(f"  successful fits          {int(ok.sum())}")
    print(f"  with MC errors           {len(parts)}")
    if missing:
        print(f"  no MC file yet           {missing}")
    if stale:
        print(f"  MC older than the fit, left out, rerun these {stale}")
    print(f"Wrote\n  {os.path.abspath(out_csv)}")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits-csv", default=FITS_CSV)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--out-csv", default=None,
                   help="default lya_fits_mc.csv next to the fits CSV")
    a = p.parse_args()
    merge(a.fits_csv, a.subcube_dir, a.out_csv)


if __name__ == "__main__":
    main()