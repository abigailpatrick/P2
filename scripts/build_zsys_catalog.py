#!/usr/bin/env python3
"""
Build the per-source z_sys catalogue that the MUSE steps read
(grating_sources_with_zsys.csv).

z_sys now comes from systemic_redshifts_by_JELS_ID.csv with the same line
priority used for Delta_v (p2_common.pick_zsys):

    [OIII] if its S/N > 13, then Halpha if its S/N > 13, else no z_sys

The old version took the best-S/N OIII grating only, with no threshold, so the
MUSE continuum subtraction, extraction and Lya fit ran on a different z_sys to
the one used for Delta_v whenever OIII was weak or missing. The output file
name and its first columns are unchanged, so lya_local_contsub.py,
optimize_lya_position_grating.py, ap_extract_specs_grating.py,
sliding_snr_lya_grating.py, fit_lya_properties_grating.py,
mc_lya_errors_grating.py, group_lya_sample.py and build_ha_flux_corrections.py
need no edits. They read z_sys and fall back to z_dja where it is blank.

Columns
-------
  ID, grating, z_dja, z_sys, z_sys_err, z_sys_snr, z_sys_quality,
  in_muse, edge, duplicate, AO_block, ra, dec, z_sys_line

  grating        grating of the chosen line. For a source with no usable
                 line, the grating with the highest attempted [OIII] S/N
                 (p2_common.fallback_grating), otherwise the first grating
                 observed. DJA redshifts from different gratings can differ
                 by ~1000 km/s, so this choice sets the MUSE search window.
  z_dja          DJA redshift of that grating
  z_sys_quality  a  [OIII]
                 b  Halpha
                 d  no z_sys, the MUSE steps fall back to z_dja
                 (the old 'c', low-S/N OIII, no longer exists)
  z_sys_line     which line set z_sys

Usage
-----
python build_zsys_catalog.py
python build_zsys_catalog.py --p2-root /some/other/P2 --dry-run
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc

EXTRA_COLS = ["in_muse", "edge", "duplicate", "AO_block", "ra", "dec"]
OUTPUT_COLS = (["ID", "grating", "z_dja", "z_sys", "z_sys_err", "z_sys_snr",
                "z_sys_quality"] + EXTRA_COLS + ["z_sys_line"])


def parse_args():
    p = argparse.ArgumentParser(description="Write grating_sources_with_zsys.csv "
                                            "from the best available systemic line.")
    p.add_argument("--p2-root", default=pc.P2_ROOT)
    p.add_argument("--systemic-csv", default=None,
                   help="Default <p2-root>/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv")
    p.add_argument("--grating-csv", default=None,
                   help="Default <p2-root>/jwst_catalogs/grating_sources_by_JELS_ID.csv")
    p.add_argument("--oiii-csv", default=None,
                   help="Default <p2-root>/jwst_catalogs/OIII_results_by_JELS_ID.csv. "
                        "Only used to pick the fallback grating for z_dja.")
    p.add_argument("--out-csv", default=None,
                   help="Default <p2-root>/jwst_catalogs/grating_sources_with_zsys.csv")
    p.add_argument("--oiii-snr-min", type=float, default=pc.SNR_MIN["OIII"])
    p.add_argument("--ha-snr-min", type=float, default=pc.SNR_MIN["Ha"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print the comparison with the existing file, write nothing.")
    a = p.parse_args()
    cat = os.path.join(a.p2_root, "jwst_catalogs")
    a.systemic_csv = os.path.abspath(a.systemic_csv or os.path.join(cat, "systemic_redshifts_by_JELS_ID.csv"))
    a.grating_csv = os.path.abspath(a.grating_csv or os.path.join(cat, "grating_sources_by_JELS_ID.csv"))
    a.oiii_csv = os.path.abspath(a.oiii_csv or os.path.join(cat, "OIII_results_by_JELS_ID.csv"))
    a.out_csv = os.path.abspath(a.out_csv or os.path.join(cat, "grating_sources_with_zsys.csv"))
    return a


def main():
    a = parse_args()
    print("[CONFIG]")
    print(f"  systemic csv   {a.systemic_csv}")
    print(f"  grating csv    {a.grating_csv}")
    print(f"  OIII csv       {a.oiii_csv}")
    print(f"  output csv     {a.out_csv}")
    cuts = {"OIII": a.oiii_snr_min, "Ha": a.ha_snr_min}
    print(f"  z_sys rule     {pc.rule_text(cuts)}")
    print("")

    systemic = pd.read_csv(a.systemic_csv)
    systemic["ID"] = systemic["ID"].astype(int)
    systemic = systemic.set_index("ID")
    grating = pd.read_csv(a.grating_csv)
    grating["ID"] = grating["ID"].astype(int)
    if os.path.exists(a.oiii_csv):
        oiii = pd.read_csv(a.oiii_csv)
        oiii["ID"] = oiii["ID"].astype(int)
        oiii = oiii.set_index("ID")
    else:
        print(f"[WARN] {a.oiii_csv} missing, fallback grating is the first observed")
        oiii = pd.DataFrame()

    rows = []
    for _, g_row in grating.iterrows():
        sid = int(g_row["ID"])
        if sid in systemic.index:
            z, z_err, z_snr, line, gr = pc.pick_zsys(systemic.loc[sid], cuts)
        else:
            z, z_err, z_snr, line, gr = np.nan, np.nan, np.nan, None, None
        if gr is None:
            gr = pc.fallback_grating(g_row, oiii.loc[sid] if sid in oiii.index else None)
        row = {
            "ID": sid,
            "grating": gr,
            "z_dja": g_row.get(f"z_{gr}", np.nan) if gr else np.nan,
            "z_sys": z, "z_sys_err": z_err, "z_sys_snr": z_snr,
            "z_sys_quality": pc.zsys_quality(line),
            "z_sys_line": line,
        }
        for c in EXTRA_COLS:
            row[c] = g_row.get(c, np.nan)
        rows.append(row)

    out = pd.DataFrame(rows, columns=OUTPUT_COLS)

    # Compare with the file being replaced, so a change in z_sys is visible.
    if os.path.exists(a.out_csv):
        old = pd.read_csv(a.out_csv)
        old["ID"] = old["ID"].astype(int)
        m = out.merge(old[["ID", "z_sys"]], on="ID", how="left", suffixes=("", "_old"))
        dv = pc.dz_to_kms(m["z_sys"], m["z_sys_old"])
        gained = m["z_sys"].notna() & m["z_sys_old"].isna()
        lost = m["z_sys"].isna() & m["z_sys_old"].notna()
        moved = np.isfinite(dv) & (np.abs(dv) > 1.0)
        print("[CHANGES vs existing file]")
        print(f"  z_sys gained   {int(gained.sum())}  {m.loc[gained, 'ID'].tolist()}")
        print(f"  z_sys lost     {int(lost.sum())}  {m.loc[lost, 'ID'].tolist()}")
        print(f"  z_sys moved >1 km/s  {int(moved.sum())}")
        for _, r in m[moved].iterrows():
            i = r.name
            print(f"     ID {int(r['ID'])}  {r['z_sys_old']:.5f} -> {r['z_sys']:.5f}  "
                  f"({dv[i]:+.0f} km/s, now {r['z_sys_line']})")
        print("")

    q = out["z_sys_quality"].value_counts()
    print("[SUMMARY]")
    print(f"  sources        {len(out)}")
    print(f"  quality a/b/d  {q.get('a', 0)} / {q.get('b', 0)} / {q.get('d', 0)}")
    print("  z_sys_line     " + ", ".join(f"{k} {v}" for k, v in
                                          out["z_sys_line"].value_counts().items()))
    if a.dry_run:
        print("\n[DRY RUN] nothing written")
        return
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print(f"\nwritten  {a.out_csv}")


if __name__ == "__main__":
    main()
