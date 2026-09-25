#!/usr/bin/env python3
"""
Recompute the Lya velocity offset using the best available systemic line per
source, without re-fitting or re-bootstrapping Lya.

For each source, z_sys is chosen by trying lines in this fixed priority order,
taking the first one that is usable:

    1. OIII   (only if grade a, i.e. z_OIII_snr >= OIII_SNR_MIN)
    2. Ha
    3. Hbeta
    4. NII
    5. OII

Steps 2-5 need no further success check here: systemic_redshifts_by_JELS_ID.csv
(written by merge_systemic_redshifts.py) already only carries a z_<line> value
for a source when that line's own summary CSV flagged it a success, and it
already picked the highest-S/N grating where more than one succeeded. This
script just reads whichever z_<line> column is populated.

z_lya, z_lya_err_mc (the bootstrap error on the Lya line centre) and lya_snr
come straight from lya_properties_mc.csv. Neither the Lya fit nor its Monte
Carlo bootstrap depend on which systemic line is used, so nothing there is
re-run. Only the two things that do depend on z_sys are recomputed:

    delta_v_kms          = c (z_lya - z_sys) / (1 + z_sys)
    delta_v_err_sys_kms  = c * z_sys_err / (1 + z_sys)
    delta_v_err_lya_kms  = c * z_lya_err_mc / (1 + z_sys)     [reusing z_lya_err_mc]
    delta_v_err_kms      = sqrt(delta_v_err_sys_kms^2 + delta_v_err_lya_kms^2)

A source with no usable line in any of the five gets delta_v_kms, its errors,
and z_sys_line left blank.

Inputs
------
  systemic_redshifts_by_JELS_ID.csv   ID, z_<line>, z_<line>_err, z_<line>_snr
                                       for line in OIII, Ha, Hbeta, NII, OII
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

C_KMS = 299792.458

# Priority order. OIII is gated on SNR, the rest are gated only on presence,
# since systemic_redshifts_by_JELS_ID.csv already filters on success.
LINE_PRIORITY = ["OIII", "Ha", "Hbeta", "NII", "OII"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Recompute Delta_v using the best available systemic "
                    "line per source (OIII grade a, then Ha, Hbeta, NII, OII).")
    p.add_argument("--systemic-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv",
                   help="Per-line systemic redshift table.")
    p.add_argument("--lya-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="Lya fit + MC bootstrap results (z_lya, z_lya_err_mc).")
    p.add_argument("--out-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/delta_v_from_best_zsys_line.csv",
                   help="Output CSV.")
    p.add_argument("--oiii-snr-min", type=float, default=13.0,
                   help="OIII is only accepted (grade a) at or above this S/N. "
                        "Below it, or if OIII is missing, the script falls "
                        "through to Ha. Default 13, matching the rest of the "
                        "P2 pipeline.")
    p.add_argument("--lya-rest", type=float, default=1215.67,
                   help="Vacuum rest wavelength of Lya, Angstrom.")
    return p.parse_args()


def pick_zsys(row, oiii_snr_min):
    """Return (z_sys, z_sys_err, z_sys_snr, z_sys_line) for one source.

    Walks LINE_PRIORITY in order. OIII additionally requires z_OIII_snr to
    clear oiii_snr_min (the 'grade a' condition); every other line is
    accepted on presence alone, since the systemic CSV already only carries
    a value there for a successful fit.
    """
    for line in LINE_PRIORITY:
        z = row.get(f"z_{line}")
        if pd.isna(z):
            continue

        if line == "OIII":
            snr = row.get("z_OIII_snr")
            if pd.isna(snr) or float(snr) < oiii_snr_min:
                continue

        z_err = row.get(f"z_{line}_err")
        z_snr = row.get(f"z_{line}_snr")
        return float(z), float(z_err), float(z_snr), line

    return np.nan, np.nan, np.nan, None


def main():
    args = parse_args()

    systemic_path = os.path.abspath(args.systemic_csv)
    lya_path = os.path.abspath(args.lya_csv)
    out_path = os.path.abspath(args.out_csv)

    print("[CONFIG]")
    print(f"  systemic csv    {systemic_path}")
    print(f"  lya csv         {lya_path}")
    print(f"  output csv      {out_path}")
    print(f"  OIII grade-a    snr >= {args.oiii_snr_min:g}")
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
            systemic.loc[src_id], args.oiii_snr_min)

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