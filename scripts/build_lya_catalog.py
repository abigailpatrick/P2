#!/usr/bin/env python
"""
build_lya_catalog.py

One table of the Lya results for every source with a z_sys, for later scripts
to read instead of joining the pipeline CSVs themselves. It also decides
which sources are Lya detections, with a rule you choose on the command line.

Detection rule
--------------
A tested source is a detection (lya_det = 1) if
  1. lya_snr >= a fixed S/N (--det-snr, default 5, pc.LYA_DET_SNR), with
     lya_snr rounded to 2 decimals first (--snr-decimals) so a source shown
     as 5.00 in the PDFs is a detection, or
     instead lya_snr above a percentile of the pooled false-position S/N
     (--det-percentile, any percentile in lya_false_pos_thresholds.csv, by
     default 90, 95, 98 and 99.5). Give one or the other. For a fixed S/N
     the percentile it sits at is printed, from the snr<S/N>_percentile
     column false_pos_percentiles.py writes,
and, if --extra-cut is given, also (with the errors of the preferred fit)
  dv_err      delta_v_err <= --dv-err-max km/s. delta_v_err is the error on
              Delta_v, the Lya MC error and z_sys_err in quadrature.
  centre_err  lya_centre_err <= --centre-err-max km/s. lya_centre_err is the
              precision of the Lya line centre alone, the MC scatter of the
              fitted peak in km/s (delta_v_err_lya_kms), with no systemic
              error in it.
Both are fixed numbers, the same for every source. Every other tested source
is a non-detection (lya_det = 0) and keeps all its values. det_reason says
why: low_snr, dv_err, centre_err, no_err (a fit but no MC error, so the MC
needs rerunning) or no_fit. Sources that could not be tested (lya_status not
ok, e.g. 65863 in the AO gap) have lya_det blank.

To help choose the fixed limits, the median of delta_v_err and of
lya_centre_err, their robust sigma (1.4826 x the median absolute deviation)
and median + 3 sigma are printed for the sources above the S/N percentile,
whichever --extra-cut is used.

The rule is stored as a tag in the det_rule column, e.g.
  snr5            (the default)
  p95             --det-percentile 95
  snr5_dverr100   --extra-cut dv_err --dv-err-max 100
  p98_cenerr60    --det-percentile 98 --extra-cut centre_err --centre-err-max 60
and the PDFs and plots made from the catalogue carry that tag in their names,
so runs with different rules sit side by side.

Close pairs
-----------
close_pair_group is 0 for a source with its own NIRSpec spectrum, otherwise
the number of the group of Isaac objects sharing one spectrum (the groups
and numbering of check_close_pairs.py). pair_status is
  single       not in a close pair
  kept         owns the spectrum of its group (--pair-keep, default 50803
               and 51988, pc.CLOSE_PAIR_KEEP)
  interloper   the other member of a group that has a kept source, at a
               lower redshift. Shown nowhere
  close_pair   any member of the other groups. Left out of the analysis and
               shown in the close-pairs PDFs
in_analysis is 1 for single and kept, 0 otherwise.

Columns
-------
  ID, z_sys, z_sys_err, z_sys_grating
  lya_snr               Sigma_Lya, the peak sliding 10 AA S/N (Paper 1)
  lya_model             the preferred fit, single (the skewed Gaussian) or
                        double (two peaks, from fit_lya_double_peak.py). Blank
                        where there is no fit
  The line quantities below come from the preferred fit, the skewed Gaussian
  for single and the red peak of the double fit for double, all with MC errors.
  z_lya, z_lya_err      from the vacuum Lya peak
  delta_v, delta_v_err  km/s, the peak relative to systemic, z_sys error included
  lya_centre_err        km/s, Lya line-centre precision (no systemic error)
  lya_centre_err_ang    the same in AA (observed, air)
  fwhm_int_kms (+ _err) intrinsic (LSF-corrected) FWHM
  fwhm_obs_kms (+ _err) FWHM as MUSE sees it
  fwhm_lsf_kms          the MUSE LSF FWHM at the line
  fwhm_resolved         1 if FWHM_int >= 0.5 FWHM_LSF and the width is off its floor
  lya_flux (+ _err)     1e-20 erg/s/cm2, the whole line (blue + red for a double)
  lya_det               1 detection, 0 non-detection, blank not tested
  det_reason            blank for detections, else low_snr, dv_err,
                        centre_err, no_err or no_fit
  det_rule              the rule tag, the same on every row
  lya_status, close_pair_group, pair_status, in_analysis
  dbic_double           BIC_single - BIC_double, for every source the double
                        fit was run on, so you can see how close a call it was
  double_reason         why the single fit was kept (dbic, weak_peak,
                        peaks_merged, ...), blank for a double
  dv_blue, peak_sep, blue_red_ratio, flux_blue, flux_red, fwhm_blue_kms
  (each + _err)         only where lya_model is double, blank otherwise
  delta_v_single (+ _err)  the skewed-Gaussian Delta_v for every source, to
                        compare with delta_v where the double was preferred
  muv_fit, muv_fit_err, beta_fit, beta_fit_err, beta_fit_good,
  f1500_fit, f1500_fit_err   photometric UV fit (fit_muv_beta.py)

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv
  MUSE_catalogs/lya_fits_mc.csv                 z_lya, lya_snr, delta_v, errors
  MUSE_catalogs/lya_detections.csv              status
  MUSE_catalogs/lya_false_pos_thresholds.csv    p90, p95, p98, p99_5, snr5_percentile
  jwst_catalogs/primer_minerva_close_pairs.csv
  jwst_catalogs/muv_beta_fits.csv               UV columns (optional)
  MUSE_catalogs/lya_double_fits.csv             double-peak columns (optional)

Outputs
-------
  MUSE_catalogs/lya_catalog.csv                 the latest run, read by default
  MUSE_catalogs/lya_catalog_<tag>.csv           a copy kept per rule
  MUSE_catalogs/lya_detection_cuts.csv          the latest rule and its counts
  MUSE_catalogs/lya_detection_cuts_<tag>.csv    a copy kept per rule

Usage
-----
python build_lya_catalog.py                                   # S/N >= 5, tag snr5
python build_lya_catalog.py --det-snr 4.5
python build_lya_catalog.py --det-percentile 95
python build_lya_catalog.py --extra-cut dv_err --dv-err-max 100
python build_lya_catalog.py --det-percentile 95 --extra-cut centre_err --centre-err-max 60
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc
from check_close_pairs import groups_from_pairs

CAT = pc.MUSE_CAT_DIR
CLOSE_PAIRS = f"{pc.P2_ROOT}/jwst_catalogs/primer_minerva_close_pairs.csv"
MUV_FITS = f"{pc.P2_ROOT}/jwst_catalogs/muv_beta_fits.csv"
DOUBLE_FITS = f"{pc.MUSE_CAT_DIR}/lya_double_fits.csv"
# Line quantities measured from whichever model is preferred: the column of
# lya_fits_mc.csv (single skewed Gaussian) and of lya_double_fits.csv (red
# peak of the double fit) that fills each catalogue column.
LINE_COLS = {  # catalogue          single fit             double fit
    "z_lya":               ("z_lya",                 "z_lya_red"),
    "z_lya_err":           ("z_lya_err_mc",          "z_lya_red_err"),
    "delta_v":             ("delta_v_kms",           "dv_red"),
    "delta_v_err":         ("delta_v_err_kms",       "dv_red_err"),
    "lya_centre_err":      ("delta_v_err_lya_kms",   "dv_red_err_lya"),
    "lya_centre_err_ang":  ("lya_peak_err_mc",       "lam_red_err"),
    "fwhm_int_kms":        ("fwhm_int_kms",          "fwhm_red_kms"),
    "fwhm_int_kms_err":    ("fwhm_int_kms_err_mc",   "fwhm_red_kms_err"),
    "fwhm_obs_kms":        ("fwhm_obs_kms",          "fwhm_red_obs_kms"),
    "fwhm_obs_kms_err":    ("fwhm_obs_kms_err_mc",   "fwhm_red_obs_kms_err"),
    "fwhm_lsf_kms":        ("fwhm_lsf_kms",          "fwhm_lsf_kms"),
    "fwhm_resolved":       ("resolved",              "red_resolved"),
    "lya_flux":            ("flux_fit",              "flux_total"),
    "lya_flux_err":        ("flux_fit_err_mc",       "flux_total_err"),
}
# Only filled where the double fit is preferred.
DOUBLE_ONLY = ["dv_blue", "dv_blue_err", "peak_sep", "peak_sep_err", "blue_red_ratio",
               "blue_red_ratio_err", "flux_blue", "flux_blue_err", "flux_red", "flux_red_err",
               "fwhm_blue_kms", "fwhm_blue_kms_err"]
UV_COLS = ["muv_fit", "muv_fit_err", "beta_fit", "beta_fit_err", "beta_fit_good",
           "f1500_fit", "f1500_fit_err"]
EXTRA = {"dv_err": ("delta_v_err", "dverr", "Delta_v error"),
         "centre_err": ("lya_centre_err", "cenerr", "Lya line-centre error")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det-snr", type=float, default=None,
                   help=f"fixed S/N that defines a detection (default {pc.LYA_DET_SNR:g})")
    p.add_argument("--det-percentile", type=float, default=None,
                   help="S/N percentile of the false positions that defines a detection, "
                        "instead of --det-snr")
    p.add_argument("--snr-decimals", type=int, default=2,
                   help="a fixed S/N cut compares lya_snr rounded to this many decimals, "
                        "as printed in the PDFs (default 2, so 4.996 counts as 5.00)")
    p.add_argument("--extra-cut", choices=["none", "dv_err", "centre_err"], default="none")
    p.add_argument("--dv-err-max", type=float, default=None,
                   help="km/s, needed with --extra-cut dv_err")
    p.add_argument("--centre-err-max", type=float, default=None,
                   help="km/s, needed with --extra-cut centre_err")
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=f"{CAT}/lya_fits_mc.csv")
    p.add_argument("--detections", default=f"{CAT}/lya_detections.csv")
    p.add_argument("--thresholds", default=f"{CAT}/lya_false_pos_thresholds.csv")
    p.add_argument("--pairs", default=CLOSE_PAIRS)
    p.add_argument("--muv-csv", default=MUV_FITS)
    p.add_argument("--double-csv", default=DOUBLE_FITS,
                   help="fit_lya_double_peak.py output, joined if it exists")
    p.add_argument("--pair-keep", type=int, nargs="*", default=list(pc.CLOSE_PAIR_KEEP))
    p.add_argument("--out", default=pc.LYA_CATALOG)
    p.add_argument("--out-cuts", default=pc.LYA_DETECTION_CUTS)
    a = p.parse_args()
    if a.det_snr is not None and a.det_percentile is not None:
        p.error("give --det-snr or --det-percentile, not both")
    if a.det_snr is None and a.det_percentile is None:
        a.det_snr = pc.LYA_DET_SNR
    if a.extra_cut == "dv_err" and a.dv_err_max is None:
        p.error("--extra-cut dv_err needs --dv-err-max (km/s)")
    if a.extra_cut == "centre_err" and a.centre_err_max is None:
        p.error("--extra-cut centre_err needs --centre-err-max (km/s)")
    return a


def pname(p_):
    return f"{p_:g}".replace(".", "_")


def rule_tag(a):
    tag = (f"snr{pname(a.det_snr)}" if a.det_percentile is None
           else f"p{pname(a.det_percentile)}")
    if a.extra_cut != "none":
        lim = a.dv_err_max if a.extra_cut == "dv_err" else a.centre_err_max
        tag += f"_{EXTRA[a.extra_cut][1]}{lim:g}".replace(".", "_")
    return tag


def close_pair_groups(pairs_csv):
    pairs = pd.read_csv(pairs_csv)
    groups = groups_from_pairs(pairs)
    lookup = {i: gi for gi, ids in enumerate(groups, start=1) for i in ids}
    return lookup, groups


def robust_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    med = float(np.median(v))
    sig = float(1.4826 * np.median(np.abs(v - med)))
    return med, sig, med + 3 * sig, int(v.size)


def tagged(path, tag):
    root, ext = os.path.splitext(os.path.abspath(path))
    return f"{root}_{tag}{ext}"


def main():
    a = parse_args()
    tag = rule_tag(a)
    print("build_lya_catalog.py")
    for path in (a.zsys, a.fits_csv, a.detections, a.thresholds, a.pairs):
        print(f"  reading  {os.path.abspath(path)}")
    print(f"  reading  {os.path.abspath(a.muv_csv)}"
          f"{'' if os.path.exists(a.muv_csv) else '  (not found, UV columns left blank)'}")
    print(f"  reading  {os.path.abspath(a.double_csv)}"
          f"{'' if os.path.exists(a.double_csv) else '  (not found, double-peak columns left blank)'}")

    zs_all = pd.read_csv(a.zsys)
    zs = zs_all[["ID", "z_sys", "z_sys_err", "z_sys_grating"]].copy()
    fits = pd.read_csv(a.fits_csv)
    det = pd.read_csv(a.detections)[["ID", "status"]]
    thr = pd.read_csv(a.thresholds).iloc[0]
    for d in (zs_all, zs, fits, det):
        d["ID"] = d["ID"].astype(int)

    if a.det_percentile is None:
        pcol = None
        snr_thr = a.det_snr
        scol = f"snr{pname(a.det_snr)}_percentile"
        thr_pct = float(thr[scol]) if scol in thr else np.nan
    else:
        pcol = f"p{pname(a.det_percentile)}"
        if pcol not in thr:
            have = [c for c in thr.index if c.startswith("p") and c[1:2].isdigit()]
            raise SystemExit(f"No {pcol} in {a.thresholds} (it has {have}). Rerun "
                             f"false_pos_percentiles.py --percentiles including "
                             f"{a.det_percentile:g}")
        snr_thr = float(thr[pcol])
        thr_pct = a.det_percentile

    def passes(snr):
        """>= for a fixed S/N, on the S/N rounded to --snr-decimals (as printed
        in the PDFs), > for a percentile threshold."""
        if a.det_percentile is None:
            return snr.round(a.snr_decimals) >= snr_thr
        return snr > snr_thr

    need = ["lya_snr"] + [v[0] for v in LINE_COLS.values()]
    need = list(dict.fromkeys(need))
    missing = [c for c in need if c not in fits.columns]
    if missing:
        raise SystemExit(f"{a.fits_csv} has no {missing}. Run the MC merge "
                         "(merge_mc_partials.py) first")
    single = fits[["ID"] + need].copy()
    out = zs.merge(single[["ID", "lya_snr"]], on="ID", how="left").merge(det, on="ID", how="left")
    out = out.rename(columns={"status": "lya_status"})
    out["lya_status"] = out["lya_status"].fillna("missing")
    ok = out["lya_status"] == "ok"

    # ---------------------------------------------------------------- preferred model
    # Every line quantity comes from the single skewed Gaussian, or from the
    # red peak of the double fit where that is preferred (double_peak == 1 in
    # fit_lya_double_peak.py). The double-only columns stay blank otherwise.
    s_idx = single.set_index("ID").reindex(out["ID"])
    for col, (scol, _) in LINE_COLS.items():
        out[col] = s_idx[scol].values
    out["delta_v_single"] = out["delta_v"]
    out["delta_v_single_err"] = out["delta_v_err"]
    has_single = out["z_lya"].notna()
    out["lya_model"] = np.where(has_single, "single", "")
    for c in ["dbic_double", "double_reason"] + DOUBLE_ONLY:
        out[c] = np.nan
    out["double_reason"] = out["double_reason"].astype(object)
    if os.path.exists(a.double_csv):
        dbl = pd.read_csv(a.double_csv)
        dbl["ID"] = dbl["ID"].astype(int)
        miss = [v[1] for v in LINE_COLS.values() if v[1] not in dbl.columns]
        if miss:
            raise SystemExit(f"{a.double_csv} has no {miss}. Rerun fit_lya_double_peak.py")
        d_idx = dbl.set_index("ID").reindex(out["ID"])
        is_dbl = (d_idx["double_peak"].values == 1) & has_single.values
        out["dbic_double"] = d_idx["dbic"].values
        out["double_reason"] = np.where(is_dbl, "", d_idx["double_reason"].fillna("").values)
        out.loc[is_dbl, "lya_model"] = "double"
        for col, (_, dcol) in LINE_COLS.items():
            out.loc[is_dbl, col] = d_idx.loc[is_dbl, dcol].values
        for c in DOUBLE_ONLY:
            out.loc[is_dbl, c] = d_idx.loc[is_dbl, c].values
    out["fwhm_resolved"] = out["fwhm_resolved"].map(
        lambda v: pd.NA if pd.isna(v) else int(str(v).lower() in ("true", "1", "1.0"))
    ).astype("Int64")

    # ---------------------------------------------------------------- detection
    snr_ok = ok & passes(out["lya_snr"])
    has_fit = out["z_lya"].notna()
    good = snr_ok.copy()
    out["det_reason"] = ""
    out.loc[ok & ~snr_ok, "det_reason"] = "low_snr"
    if a.extra_cut != "none":
        col = EXTRA[a.extra_cut][0]
        lim = a.dv_err_max if a.extra_cut == "dv_err" else a.centre_err_max
        has_err = out[col].notna()
        good = snr_ok & has_fit & has_err & (out[col] <= lim)
        out.loc[snr_ok & has_fit & has_err & ~good, "det_reason"] = a.extra_cut
        out.loc[snr_ok & has_fit & ~has_err, "det_reason"] = "no_err"
        out.loc[snr_ok & ~has_fit, "det_reason"] = "no_fit"
    out["lya_det"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[ok, "lya_det"] = 0
    out.loc[good, "lya_det"] = 1
    out["det_rule"] = tag

    # ---------------------------------------------------------------- close pairs
    lookup, groups = close_pair_groups(a.pairs)
    out["close_pair_group"] = out["ID"].map(lookup).fillna(0).astype(int)
    keep = set(a.pair_keep)
    keep_groups = {lookup[i] for i in keep if i in lookup}
    bad_keep = sorted(keep - set(lookup))
    if bad_keep:
        print(f"  WARNING --pair-keep IDs not in any close pair: {bad_keep}")
    g = out["close_pair_group"]
    out["pair_status"] = np.where(g == 0, "single", "close_pair")
    out.loc[out["ID"].isin(keep) & (g > 0), "pair_status"] = "kept"
    out.loc[g.isin(keep_groups) & ~out["ID"].isin(keep), "pair_status"] = "interloper"
    out["in_analysis"] = out["pair_status"].isin(["single", "kept"]).astype(int)
    if "close_pair" in zs_all.columns:
        cp = zs_all.set_index("ID")["close_pair"].reindex(out["ID"]).fillna(0).astype(int)
        odd = out.loc[(cp.values == 1) != (g.values > 0), "ID"].tolist()
        if odd:
            print(f"  WARNING close_pair in the z_sys CSV disagrees with the groups for {odd}")

    # ---------------------------------------------------------------- UV columns
    if os.path.exists(a.muv_csv):
        uv = pd.read_csv(a.muv_csv)
        uv["ID"] = uv["ID"].astype(int)
        miss = [c for c in UV_COLS + ["fit_flag"] if c not in uv.columns]
        if miss:
            raise SystemExit(f"{a.muv_csv} has no {miss}. Rerun fit_muv_beta.py")
        bad = uv["fit_flag"] != "ok"
        uv.loc[bad, [c for c in UV_COLS if c != "beta_fit_good"]] = np.nan
        out = out.merge(uv[["ID"] + UV_COLS], on="ID", how="left")
        out["beta_fit_good"] = out["beta_fit_good"].fillna(0).astype(int)
    else:
        for c in UV_COLS:
            out[c] = np.nan

    cols = (["ID", "z_sys", "z_sys_err", "z_sys_grating", "lya_snr", "lya_model"]
            + list(LINE_COLS)
            + ["lya_det", "det_reason", "det_rule", "lya_status",
               "close_pair_group", "pair_status", "in_analysis",
               "dbic_double", "double_reason"] + DOUBLE_ONLY
            + ["delta_v_single", "delta_v_single_err"] + UV_COLS)
    out = out[cols].sort_values("ID").reset_index(drop=True)

    # ---------------------------------------------------------------- report
    ok = out["lya_status"] == "ok"
    snr_ok = ok & passes(out["lya_snr"])
    ana = ok & (out["in_analysis"] == 1)
    print(f"\n  rule {tag}")
    if a.det_percentile is None:
        pct_txt = (f"the {thr_pct:.2f}th percentile of the false positions" if np.isfinite(thr_pct)
                   else f"percentile unknown, run false_pos_percentiles.py --snr-values "
                        f"{a.det_snr:g}")
        print(f"  S/N cut   lya_snr >= {snr_thr:g} (rounded to {a.snr_decimals} decimals), "
              f"{pct_txt}")
        edge = out[ok & passes(out["lya_snr"]) & (out["lya_snr"] < snr_thr)]
        if len(edge):
            print("            in only because of the rounding: " + ", ".join(
                f"{int(i)} ({v:.4f})" for i, v in zip(edge["ID"], edge["lya_snr"])))
    else:
        print(f"  S/N cut   lya_snr > {snr_thr:.2f}, the {a.det_percentile:g}th percentile of the "
              f"false positions")
    print(f"            {int(snr_ok.sum())} of {int(ok.sum())} tested sources pass "
          f"({int((snr_ok & (out['in_analysis'] == 1)).sum())} in the analysis)")
    others = [c for c in thr.index if c.startswith("p") and c[1:2].isdigit() and c != pcol]
    if others:
        print("            for comparison  " + "   ".join(
            f"{c[1:].replace('_', '.')}th: {float(thr[c]):.2f} "
            f"({int((ok & (out['lya_snr'] > float(thr[c]))).sum())})" for c in others))

    print(f"\n  error distributions of the {int(snr_ok.sum())} sources above the S/N cut "
          "(robust sigma = 1.4826 x MAD)")
    print(f"    {'km/s':<16}{'N':>4} {'median':>8} {'sigma':>8} {'median+3sigma':>14}")
    for key, (col, _, label) in EXTRA.items():
        med, sig, lim3, n = robust_stats(out.loc[snr_ok, col])
        print(f"    {col:<16}{n:>4} {med:8.1f} {sig:8.1f} {lim3:14.1f}   {label}")
    if a.extra_cut != "none":
        col, _, label = EXTRA[a.extra_cut]
        lim = a.dv_err_max if a.extra_cut == "dv_err" else a.centre_err_max
        n_fail = int((out["det_reason"] == a.extra_cut).sum())
        print(f"\n  extra cut {col} <= {lim:g} km/s, {n_fail} of the sources above the S/N cut fail it")
        fails = out[out["det_reason"] == a.extra_cut].sort_values(col)
        for _, r in fails.iterrows():
            print(f"      {int(r['ID']):>6d}  {col} {r[col]:6.1f} km/s  S/N {r['lya_snr']:5.1f}"
                  f"{'' if r['in_analysis'] == 1 else '  (' + r['pair_status'] + ')'}")
    else:
        print("\n  no extra cut (--extra-cut dv_err or centre_err adds one)")

    print("\n  close-pair groups")
    for gi, ids in enumerate(groups, start=1):
        here = out[out["close_pair_group"] == gi]
        desc = ", ".join(f"{i} {s}" for i, s in zip(here["ID"], here["pair_status"]))
        print(f"    group {gi}  {desc if desc else 'no member with z_sys'}")

    print(f"\n  sources                 {len(out)}")
    print(f"  in the analysis         {int((out['in_analysis'] == 1).sum())}")
    print(f"    detections            {int((ana & (out['lya_det'] == 1)).sum())}")
    print(f"    non-detections        {int((ana & (out['lya_det'] == 0)).sum())}")
    for r in ("low_snr", "dv_err", "centre_err", "no_err", "no_fit"):
        ids = out.loc[ana & (out["det_reason"] == r), "ID"].tolist()
        if ids:
            print(f"      {r:<20}  {len(ids):>3d}" + (f"  {ids}" if r != "low_snr" else ""))
    cpair = out[out["pair_status"] == "close_pair"]
    print(f"  close pairs (own PDFs)  {len(cpair)}, of which "
          f"{int((cpair['lya_det'] == 1).sum())} pass the detection rule")
    print(f"  interlopers (dropped)   {out.loc[out['pair_status'] == 'interloper', 'ID'].tolist()}")
    if out["dbic_double"].notna().any():
        dp = out[(out["lya_model"] == "double") & ana]
        print(f"  double preferred        {len(dp)} in the analysis"
              + (f" {dp['ID'].tolist()}" if len(dp) else "")
              + f", {int((dp['lya_det'] == 1).sum())} of them detections")
        for _, r in dp.iterrows():
            print(f"      {int(r['ID']):>6d}  dBIC {r['dbic_double']:6.1f}  Delta_v red "
                  f"{r['delta_v']:6.0f} (single fit {r['delta_v_single']:6.0f})  blue "
                  f"{r['dv_blue']:6.0f} km/s  B/R {r['blue_red_ratio']:.2f}")
    else:
        print("  double preferred        not known, no lya_double_fits.csv, every line "
              "quantity is from the single fit")
    nt = out.loc[~ok, ["ID", "lya_status"]].values.tolist()
    if nt:
        print(f"  not tested              {nt}")
    no_err = out.loc[ok & out["z_lya"].notna() & out["delta_v_err"].isna(), "ID"].tolist()
    if no_err:
        print(f"  WARNING fit but no MC error {no_err}. Rerun the MC for these")

    cuts = pd.DataFrame([dict(
        det_rule=tag, det_mode="snr" if a.det_percentile is None else "percentile",
        det_snr=a.det_snr if a.det_percentile is None else np.nan,
        det_percentile=a.det_percentile if a.det_percentile is not None else np.nan,
        snr_threshold=round(snr_thr, 3),
        snr_threshold_percentile=round(thr_pct, 3) if np.isfinite(thr_pct) else np.nan,
        extra_cut=a.extra_cut,
        dv_err_max=a.dv_err_max if a.extra_cut == "dv_err" else np.nan,
        centre_err_max=a.centre_err_max if a.extra_cut == "centre_err" else np.nan,
        n_snr_pass=int(snr_ok.sum()),
        n_detections=int((ana & (out["lya_det"] == 1)).sum()),
        n_non_detections=int((ana & (out["lya_det"] == 0)).sum()),
        pair_keep=" ".join(map(str, sorted(keep))))])
    written = []
    for path, d in ((a.out, out), (tagged(a.out, tag), out),
                    (a.out_cuts, cuts), (tagged(a.out_cuts, tag), cuts)):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_csv(path, index=False)
        written.append(path)
    print("")
    for path in written:
        print(f"Wrote  {path}")


if __name__ == "__main__":
    main()