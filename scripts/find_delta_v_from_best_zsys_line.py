#!/usr/bin/env python3
"""
Recompute the Lya velocity offset using the best available systemic line per
source, without re-fitting or re-bootstrapping Lya.

For each source, z_sys comes from p2_common.pick_zsys, the rule shared by
the whole pipeline:

    1. [OIII] if z_OIII_snr > 13
    2. Halpha if z_Ha_snr   > 13
    3. otherwise no z_sys

systemic_redshifts_by_JELS_ID.csv (written by merge_systemic_redshifts.py)
already only carries a z_<line> value when that line's fit was flagged a
success, and it already picked the highest-S/N grating.

z_lya, z_lya_err_mc (the bootstrap error on the Lya line centre) and lya_snr
come straight from lya_properties_mc.csv. z_lya is the peak of the
LSF-convolved model by default (peak_used = 'obs', see
fit_lya_properties_grating.py --peak), and z_lya_int / z_lya_obs are carried
through so the alternative Delta_v can be compared. Neither the Lya fit nor its Monte
Carlo bootstrap depend on which systemic line is used, so nothing there is
re-run. Only the two things that do depend on z_sys are recomputed:

    delta_v_kms          = c (z_lya - z_sys) / (1 + z_sys)
    delta_v_err_sys_kms  = c * z_sys_err / (1 + z_sys)
    delta_v_err_lya_kms  = c * z_lya_err_mc / (1 + z_sys)     [reusing z_lya_err_mc]
    delta_v_err_kms      = sqrt(delta_v_err_sys_kms^2 + delta_v_err_lya_kms^2)

A source with no z_sys gets delta_v_kms, its errors and z_sys_line left
blank.

Inputs
------
  systemic_redshifts_by_JELS_ID.csv   ID, z_<line>, z_<line>_err, z_<line>_snr
                                       for line in OIII and Ha
  lya_properties_mc.csv               ID, z_lya, z_lya_err_mc, lya_snr,
                                       fit_success, ra, dec

Output
------
  delta_v_from_best_zsys_line.csv, written to /ceph/cephfs/apatrick/P2/MUSE_catalogs

Usage
-----
python find_delta_v_from_best_zsys_line.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc

C_KMS = 299792.458
LINE_PRIORITY = pc.LINE_PRIORITY


def parse_args():
    p = argparse.ArgumentParser(
        description="Recompute Delta_v using the best available systemic "
                    "line per source (OIII S/N > 13, then Ha S/N > 13).")
    p.add_argument("--systemic-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv",
                   help="Per-line systemic redshift table.")
    p.add_argument("--lya-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="Lya fit + MC bootstrap results (z_lya, z_lya_err_mc).")
    p.add_argument("--out-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/delta_v_from_best_zsys_line.csv",
                   help="Output CSV.")
    p.add_argument("--oiii-snr-min", type=float, default=pc.SNR_MIN["OIII"],
                   help="[OIII] is used only above this S/N. Default 13.")
    p.add_argument("--ha-snr-min", type=float, default=pc.SNR_MIN["Ha"],
                   help="Halpha is used only above this S/N. Default 13.")
    p.add_argument("--lya-rest", type=float, default=1215.67,
                   help="Vacuum rest wavelength of Lya, Angstrom.")
    return p.parse_args()


def pick_zsys(row, cuts):
    """Return (z_sys, z_sys_err, z_sys_snr, z_sys_line) for one source,
    using the shared rule in p2_common.pick_zsys."""
    z, z_err, z_snr, line, _ = pc.pick_zsys(row, cuts)
    return z, z_err, z_snr, line


def main():
    args = parse_args()

    systemic_path = os.path.abspath(args.systemic_csv)
    lya_path = os.path.abspath(args.lya_csv)
    out_path = os.path.abspath(args.out_csv)

    print("[CONFIG]")
    print(f"  systemic csv    {systemic_path}")
    print(f"  lya csv         {lya_path}")
    print(f"  output csv      {out_path}")
    cuts = {"OIII": args.oiii_snr_min, "Ha": args.ha_snr_min}
    print(f"  z_sys rule      {pc.rule_text(cuts)}")
    print("")

    systemic = pd.read_csv(systemic_path)
    systemic["ID"] = systemic["ID"].astype(int)
    systemic = systemic.set_index("ID")

    lya = pd.read_csv(lya_path)
    lya["ID"] = lya["ID"].astype(int)

    required = ["ID", "z_lya", "z_lya_err_mc", "lya_snr", "fit_success"]
    missing = [c for c in required if c not in lya.columns]
    if missing:
        raise KeyError(
            f"{lya_path} is missing columns {missing}. Available: "
            f"{list(lya.columns)}. Did you point --lya-csv at "
            f"lya_properties.csv instead of the _mc.csv? z_lya_err_mc only "
            f"exists after mc_lya_errors_grating.py has run.")

    rows = []
    line_counts = {line: 0 for line in LINE_PRIORITY}
    n_none = 0
    n_no_lya = 0

    for _, lrow in lya.iterrows():
        src_id = int(lrow["ID"])

        z_lya = pd.to_numeric(lrow.get("z_lya"), errors="coerce")
        z_lya_err_mc = pd.to_numeric(lrow.get("z_lya_err_mc"), errors="coerce")

        out = {
            "ID": src_id,
            "ra": lrow.get("ra"),
            "dec": lrow.get("dec"),
            "z_lya": float(z_lya) if pd.notna(z_lya) else np.nan,
            "peak_used": lrow.get("peak_used"),
            "lya_snr": lrow.get("lya_snr"),
            "fit_success": lrow.get("fit_success"),
            "z_sys": np.nan, "z_sys_err": np.nan, "z_sys_snr": np.nan,
            "z_sys_line": None,
            "delta_v_kms": np.nan,
            "delta_v_err_lya_kms": np.nan,
            "delta_v_err_sys_kms": np.nan,
            "delta_v_err_kms": np.nan,
        }

        if not np.isfinite(z_lya):
            n_no_lya += 1
            rows.append(out)
            continue

        if src_id not in systemic.index:
            n_none += 1
            rows.append(out)
            continue

        z_sys, z_sys_err, z_sys_snr, z_sys_line = pick_zsys(
            systemic.loc[src_id], cuts)

        if z_sys_line is None:
            n_none += 1
            rows.append(out)
            continue

        line_counts[z_sys_line] += 1

        conv = C_KMS / (1.0 + z_sys)
        delta_v = conv * (z_lya - z_sys)
        dv_err_sys = conv * z_sys_err if np.isfinite(z_sys_err) else np.nan
        dv_err_lya = conv * z_lya_err_mc if np.isfinite(z_lya_err_mc) else np.nan

        err_terms = [e for e in (dv_err_sys, dv_err_lya) if np.isfinite(e)]
        dv_err = float(np.sqrt(sum(e ** 2 for e in err_terms))) if err_terms else np.nan

        extra = {}
        for kind in ("obs", "int"):
            zk = pd.to_numeric(lrow.get(f"z_lya_{kind}"), errors="coerce")
            extra[f"delta_v_{kind}_kms"] = (float(conv * (zk - z_sys))
                                            if pd.notna(zk) else np.nan)

        out.update(extra)
        out.update({
            "z_sys": z_sys, "z_sys_err": z_sys_err, "z_sys_snr": z_sys_snr,
            "z_sys_line": z_sys_line,
            "delta_v_kms": float(delta_v),
            "delta_v_err_lya_kms": float(dv_err_lya) if np.isfinite(dv_err_lya) else np.nan,
            "delta_v_err_sys_kms": float(dv_err_sys) if np.isfinite(dv_err_sys) else np.nan,
            "delta_v_err_kms": dv_err,
        })
        rows.append(out)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_total = len(out_df)
    n_with_dv = int(out_df["z_sys_line"].notna().sum())

    print("=" * 60)
    print(f"sources in Lya properties file  {n_total}")
    print(f"  no successful Lya fit         {n_no_lya}")
    print(f"  no usable systemic line       {n_none}")
    print(f"  delta_v computed              {n_with_dv}")
    print("  z_sys_line breakdown:")
    for line in LINE_PRIORITY:
        print(f"    {line:<6} {line_counts[line]}")
    print(f"written  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()