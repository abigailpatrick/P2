#!/usr/bin/env python
"""
build_lya_catalog.py

One table of the Lya results for every source with a z_sys, for later scripts
to read instead of joining the pipeline CSVs themselves.

Columns
-------
  ID                Isaac's PRIMER + MINERVA Number
  z_sys             systemic redshift (build_zsys_catalog.py)
  z_sys_err
  z_sys_grating     NIRSpec grating of the line that set z_sys
  z_lya             from the vacuum Lya peak of the skewed-Gaussian fit
  z_lya_err         MC error (z_lya_err_mc)
  lya_snr           Sigma_Lya, the peak sliding 10 AA S/N (Paper 1)
  delta_v           km/s, c (z_lya - z_sys) / (1 + z_sys)
  delta_v_err       km/s, MC Lya error and z_sys_err in quadrature
  det_99_5          1 if lya_snr is above the pooled 99.5th percentile of the
  det_98            false-position S/N, 0 if not (98th, 95th likewise). Blank
  det_95            for sources that could not be tested (see lya_status)
  lya_status        ok, or why the source has no flags (no_coverage for a Lya
                    band in the AO gap, e.g. 65863)
  close_pair_group  0 if the source's NIRSpec spectrum is its own. Otherwise
                    the number (1, 2, ...) of the group of Isaac objects that
                    share one spectrum. The groups and their numbers are the
                    ones check_close_pairs.py uses, so group n here is
                    plots/close_pairs/group_<n>_<IDs>.png
  muv_fit           photometric M_UV and its error (fit_muv_beta.py, total
  muv_fit_err       fluxes, no dust correction). Blank if the fit failed
  beta_fit          UV slope and its error, same fit
  beta_fit_err
  beta_fit_good     1 if beta_fit_err < 0.5 (Paper 1), else 0
  f1500_fit         f_nu at rest 1500 AA from the same fit, nJy
  f1500_fit_err

z_lya, delta_v and their errors are kept for every source with a successful
fit, detected or not, so a cut is always made with the flags. They are blank
where the fit failed. The flag levels follow --levels and are computed from
the thresholds file, so any percentile in it can be added (e.g. --levels
99.5 98 95 90 gives det_90 too).

Close-pair groups are connected sets of IDs in primer_minerva_close_pairs.csv
(ID and near_ID share at least one spectrum), numbered from 1 in order of
their smallest ID. A group is numbered even if none of its members has a
z_sys, so the numbers stay the same as in check_close_pairs.py. Only members
with a z_sys appear here.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv   ID, z_sys, z_sys_err, z_sys_grating
                                                  (close_pair, for a consistency check)
  MUSE_catalogs/lya_fits_mc.csv                   z_lya, z_lya_err_mc, lya_snr,
                                                  delta_v_kms, delta_v_err_kms
  MUSE_catalogs/lya_detections.csv                status
  MUSE_catalogs/lya_false_pos_thresholds.csv      p99_5, p98, p95, ...
  jwst_catalogs/primer_minerva_close_pairs.csv    ID, near_ID
  jwst_catalogs/muv_beta_fits.csv                 muv_fit, beta_fit, ... (optional,
                                                  the UV columns are blank without it)

Output
------
  MUSE_catalogs/lya_catalog.csv   (overwritten)

Usage
-----
python build_lya_catalog.py
python build_lya_catalog.py --levels 99.5 98 95 90
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc
from check_close_pairs import groups_from_pairs

CAT = pc.MUSE_CAT_DIR
LYA_CATALOG = getattr(pc, "LYA_CATALOG", f"{CAT}/lya_catalog.csv")
CLOSE_PAIRS = f"{pc.P2_ROOT}/jwst_catalogs/primer_minerva_close_pairs.csv"
MUV_FITS = f"{pc.P2_ROOT}/jwst_catalogs/muv_beta_fits.csv"
UV_COLS = ["muv_fit", "muv_fit_err", "beta_fit", "beta_fit_err", "beta_fit_good",
           "f1500_fit", "f1500_fit_err"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits_mc.csv")
    p.add_argument("--detections", default=f"{CAT}/lya_detections.csv")
    p.add_argument("--thresholds", default=f"{CAT}/lya_false_pos_thresholds.csv")
    p.add_argument("--pairs", default=CLOSE_PAIRS)
    p.add_argument("--muv-csv", default=MUV_FITS)
    p.add_argument("--levels", type=float, nargs="+", default=[99.5, 98, 95])
    p.add_argument("--out", default=LYA_CATALOG)
    return p.parse_args()


def pname(p_):
    return f"{p_:g}".replace(".", "_")


def close_pair_groups(pairs_csv):
    """{ID: group number} for every ID in a close-pair group, and the groups."""
    pairs = pd.read_csv(pairs_csv)
    groups = groups_from_pairs(pairs)
    lookup = {i: gi for gi, ids in enumerate(groups, start=1) for i in ids}
    return lookup, groups


def main():
    a = parse_args()
    print("build_lya_catalog.py")
    for path in (a.zsys, a.fits_csv, a.detections, a.thresholds, a.pairs):
        print(f"  reading  {os.path.abspath(path)}")
    have_uv = os.path.exists(a.muv_csv)
    print(f"  reading  {os.path.abspath(a.muv_csv)}"
          f"{'' if have_uv else '  (not found, UV columns left blank)'}")

    zs_all = pd.read_csv(a.zsys)
    zs = zs_all[["ID", "z_sys", "z_sys_err", "z_sys_grating"]].copy()
    fits = pd.read_csv(a.fits_csv)
    det = pd.read_csv(a.detections)[["ID", "status"]]
    thr = pd.read_csv(a.thresholds).iloc[0]
    for d in (zs_all, zs, fits, det):
        d["ID"] = d["ID"].astype(int)

    need = ["z_lya", "z_lya_err_mc", "lya_snr", "delta_v_kms", "delta_v_err_kms"]
    missing = [c for c in need if c not in fits.columns]
    if missing:
        raise SystemExit(f"{a.fits_csv} has no {missing}. Run the MC merge "
                         "(merge_mc_partials.py) first")
    fits = fits[["ID"] + need].rename(columns={"z_lya_err_mc": "z_lya_err",
                                               "delta_v_kms": "delta_v",
                                               "delta_v_err_kms": "delta_v_err"})

    out = zs.merge(fits, on="ID", how="left").merge(det, on="ID", how="left")
    out = out.rename(columns={"status": "lya_status"})
    out["lya_status"] = out["lya_status"].fillna("missing")
    ok = out["lya_status"] == "ok"

    flags = []
    print("")
    for lev in a.levels:
        col = f"p{pname(lev)}"
        if col not in thr:
            print(f"  WARNING no {col} in the thresholds file, det_{pname(lev)} left out. "
                  f"Rerun false_pos_percentiles.py with --percentiles including {lev:g}")
            continue
        t = float(thr[col])
        flag = f"det_{pname(lev)}"
        out[flag] = pd.array([pd.NA] * len(out), dtype="Int64")
        out.loc[ok, flag] = (out.loc[ok, "lya_snr"] > t).astype(int)
        flags.append(flag)
        print(f"  {flag:<9} lya_snr > {t:.2f}  {int(out[flag].sum()):>3d} of {int(ok.sum())}")

    # close-pair groups
    lookup, groups = close_pair_groups(a.pairs)
    out["close_pair_group"] = out["ID"].map(lookup).fillna(0).astype(int)
    print(f"\n  close-pair groups  {len(groups)} in the pairs file")
    for gi, ids in enumerate(groups, start=1):
        here = [i for i in ids if i in set(out["ID"])]
        print(f"    group {gi}  {' '.join(map(str, ids)):<22}  with z_sys: "
              f"{' '.join(map(str, here)) if here else 'none'}")
    if "close_pair" in zs_all.columns:
        cp = zs_all.set_index("ID")["close_pair"].reindex(out["ID"]).fillna(0).astype(int)
        odd = out.loc[(cp.values == 1) != (out["close_pair_group"].values > 0), "ID"].tolist()
        if odd:
            print(f"  WARNING close_pair in the z_sys CSV disagrees with the groups for {odd}. "
                  "Was primer_minerva_close_pairs.csv rebuilt with the sample?")

    # photometric M_UV and beta
    if have_uv:
        uv = pd.read_csv(a.muv_csv)
        uv["ID"] = uv["ID"].astype(int)
        missing = [c for c in UV_COLS + ["fit_flag"] if c not in uv.columns]
        if missing:
            raise SystemExit(f"{a.muv_csv} has no {missing}. Rerun fit_muv_beta.py")
        bad = uv["fit_flag"] != "ok"
        uv.loc[bad, [c for c in UV_COLS if c != "beta_fit_good"]] = np.nan
        out = out.merge(uv[["ID"] + UV_COLS], on="ID", how="left")
        out["beta_fit_good"] = out["beta_fit_good"].fillna(0).astype(int)
        no_uv = out.loc[out["muv_fit"].isna(), "ID"].tolist()
        print(f"\n  muv_fit, beta_fit  {int(out['muv_fit'].notna().sum())} of {len(out)}"
              f"{'  missing ' + str(no_uv) if no_uv else ''}")
        print(f"  beta_fit_good      {int(out['beta_fit_good'].sum())}")
    else:
        for c in UV_COLS:
            out[c] = np.nan

    cols = ["ID", "z_sys", "z_sys_err", "z_sys_grating", "z_lya", "z_lya_err",
            "lya_snr", "delta_v", "delta_v_err"] + flags + ["lya_status", "close_pair_group"] + UV_COLS
    out = out[cols].sort_values("ID").reset_index(drop=True)

    no_fit = out.loc[out["z_lya"].isna(), "ID"].tolist()
    no_err = out.loc[out["z_lya"].notna() & out["z_lya_err"].isna(), "ID"].tolist()
    bad = out.loc[~out["lya_status"].eq("ok"), ["ID", "lya_status"]].values.tolist()
    print(f"\n  sources            {len(out)}")
    print(f"  in a close pair    {int((out['close_pair_group'] > 0).sum())}")
    print(f"  no Lya fit         {len(no_fit)}  {no_fit if no_fit else ''}")
    if no_err:
        print(f"  WARNING fit but no MC error {no_err}, rerun the MC for these")
    if bad:
        print(f"  not tested         {bad}")

    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nWrote  {path}")


if __name__ == "__main__":
    main()