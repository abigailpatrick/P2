#!/usr/bin/env python3
"""
Build the P2 master catalogue, one row per source, from the current outputs of
every pipeline stage.

Nothing is fitted or measured here. The script only reads, renames and joins,
plus a few cross-checks. Rerun it whenever any upstream CSV has been
regenerated (new z_sys, new Lya fits, new labels, SED or fesc values) and the
master catalogue picks up the changes. Each run prints which columns changed
for which sources compared with the previous master.

Rows
----
Every source in grating_sources_by_JELS_ID.csv (all NIRSpec grating sources,
in or out of MUSE). Filter on in_muse, is_primary, tier or manual as needed.

Blocks (in column order), each read from one upstream file
-----------------------------------------------------------
  sample      grating_sources_by_JELS_ID.csv, _good.csv
  zsys        systemic_redshifts_by_JELS_ID.csv, with the rule in
              p2_common.pick_zsys ([OIII] S/N > 13, then Halpha S/N > 13,
              else no z_sys)
  lya_det     lya_sliding_snr_grating.csv, false_pos_snr_percentiles.csv,
              optimal_offsets_grating.csv
  lya_fit     lya_properties_mc.csv
  delta_v     delta_v_from_best_zsys_line.csv
  tier        lya_group_<tier>.csv  (automatic tier and pass/fail criteria)
  labels      lya_manual_labels.csv (by-eye a/b/c/d)
  uv          muv_beta_by_JELS_ID.csv
  lines       Ha_summary_by_JELS_ID.csv, Hbeta_summary_by_JELS_ID.csv
  ha_corr     ha_hb_flux_corrections.csv (path set by --ha-csv)
  sed         placeholder, filled once the SED input script exists
  fesc        placeholder, filled once the escape-fraction script exists

A missing file leaves its block blank and prints a warning, so the catalogue
always builds. A missing column inside a file does the same for that column.
To add or rename a column, edit the block definitions in BLOCKS below.

Provenance columns (at the end, for information only)
------------------------------------------------------
  zsys_muse             z the MUSE products (contsub, extraction, Lya fit)
                        were made with, read from lya_properties_mc.csv
  zsys_muse_source      z_sys or z_dja, as recorded by the Lya fitter
  zsys_mismatch_kms     zsys_muse relative to z_best. z_best is z_sys, or
                        z_dja where there is no z_sys, which is what the MUSE
                        steps read from grating_sources_with_zsys.csv
  z_sys_dv              z_sys used in delta_v_from_best_zsys_line.csv
The redshift used for M_UV and beta is uv_z in the uv block.

Outputs (default directory <p2-root>/master_catalog)
---------------------------------------------------
  p2_master_catalog.csv    the catalogue, overwritten each run
  p2_master_columns.csv    column dictionary: block, source file, source
                           column, unit, description

Usage
-----
python build_master_catalog.py
python build_master_catalog.py --dry-run          # report changes, write nothing
python build_master_catalog.py --ha-csv /ceph/cephfs/apatrick/P2/jwst_catalogs/ha_hb_flux_corrections_2as.csv
"""

import argparse
import io
import os
import time

import numpy as np
import pandas as pd

import p2_common as pc

TIERS = ["gold", "silver", "bronze", "stone", "bad"]
MUSE_FLUX_UNIT = 1e-20     # erg/s/cm2 per MUSE cube flux unit (check BUNIT)

# ---------------------------------------------------------------------------
# Simple blocks: (source column, master column, unit, description).
# The file path is relative to --p2-root unless it is absolute.
# ---------------------------------------------------------------------------
BLOCKS = {
    "lya_det_snr": dict(
        file="MUSE_catalogs/lya_sliding_snr_grating.csv",
        cols=[("peak_snr", "lya_peak_snr", "", "Peak sliding-window Lya S/N in the search band"),
              ("best_center", "lya_snr_centre_air", "AA", "Observed (air) wavelength of the peak S/N window")]),
    "lya_det_offsets": dict(
        file="MUSE_catalogs/optimal_offsets_grating.csv",
        cols=[("dx_arcsec", "lya_dx_arcsec", "arcsec", "Optimised aperture offset in x from the JWST position"),
              ("dy_arcsec", "lya_dy_arcsec", "arcsec", "Optimised aperture offset in y from the JWST position")]),
    "lya_fit": dict(
        file="MUSE_catalogs/lya_properties_mc.csv",
        cols=[("fit_success", "lya_fit_success", "", "Skewed-Gaussian Lya fit converged"),
              ("quality_flag", "lya_quality_flag", "", "1 if the fitter raised no quality issue"),
              ("quality_reason", "lya_quality_reason", "", "Fitter quality notes, ';' separated"),
              ("z_lya", "z_lya", "", "Lya redshift from the model peak used (peak_used)"),
              ("z_lya_err_mc", "z_lya_err", "", "MC error on z_lya"),
              ("peak_used", "lya_peak_used", "", "Model peak defining z_lya, obs (LSF-convolved) or int"),
              ("lya_peak_wave_vac", "lya_peak_wave_vac", "AA", "Observed vacuum wavelength of the Lya peak"),
              ("flux_fit", "lya_flux", "MUSE units (1e-20 erg/s/cm2)", "Lya flux in the 0.6 arcsec aperture, skewed-Gaussian integral"),
              ("flux_fit_err", "lya_flux_err_fit", "MUSE units (1e-20 erg/s/cm2)", "Covariance error on lya_flux"),
              ("flux_fit_err_mc", "lya_flux_err_mc", "MUSE units (1e-20 erg/s/cm2)", "MC error on lya_flux"),
              ("fwhm_int_kms", "lya_fwhm_int_kms", "km/s", "Intrinsic (LSF-corrected) Lya FWHM"),
              ("fwhm_int_kms_err_mc", "lya_fwhm_int_kms_err", "km/s", "MC error on the intrinsic FWHM"),
              ("fwhm_obs_kms", "lya_fwhm_obs_kms", "km/s", "Observed (LSF-convolved) Lya FWHM"),
              ("fwhm_obs_kms_err_mc", "lya_fwhm_obs_kms_err", "km/s", "MC error on the observed FWHM"),
              ("fwhm_lsf_kms", "lya_fwhm_lsf_kms", "km/s", "MUSE LSF FWHM at the line (Bacon+17)"),
              ("resolved", "lya_resolved", "", "Intrinsic FWHM >= 0.5 LSF and sigma off its floor"),
              ("frac_unresolved_mc", "lya_frac_unresolved_mc", "", "Fraction of MC draws that were unresolved"),
              ("alpha_skew", "lya_alpha_skew", "", "Intrinsic skewness parameter"),
              ("skew_at_max", "lya_skew_at_max", "", "Skewness reached its upper bound"),
              ("n_mc_success", "lya_n_mc_success", "", "Number of successful MC refits")]),
    "delta_v": dict(
        file="MUSE_catalogs/delta_v_from_best_zsys_line.csv",
        cols=[("delta_v_kms", "delta_v_kms", "km/s", "Lya velocity offset c (z_lya - z_sys)/(1 + z_sys)"),
              ("delta_v_err_kms", "delta_v_err_kms", "km/s", "Total Delta_v error, Lya and systemic in quadrature"),
              ("delta_v_err_lya_kms", "delta_v_err_lya_kms", "km/s", "Lya part of the Delta_v error"),
              ("delta_v_err_sys_kms", "delta_v_err_sys_kms", "km/s", "Systemic part of the Delta_v error"),
              ("delta_v_obs_kms", "delta_v_obs_kms", "km/s", "Delta_v from the LSF-convolved model peak"),
              ("delta_v_int_kms", "delta_v_int_kms", "km/s", "Delta_v from the intrinsic model peak"),
              ("z_sys", "z_sys_dv", "", "z_sys used in delta_v_from_best_zsys_line.csv")]),
    "uv": dict(
        file="jwst_catalogs/muv_beta_by_JELS_ID.csv",
        cols=[("M_UV", "M_UV", "AB mag", "Absolute UV magnitude at 1500 AA, no dust correction"),
              ("M_UV_err", "M_UV_err", "AB mag", "Error on M_UV"),
              ("m_UV", "m_UV", "AB mag", "Apparent magnitude at rest 1500 AA"),
              ("beta", "beta", "", "UV slope, f_lambda propto lambda^beta"),
              ("beta_err", "beta_err", "", "Error on beta"),
              ("f1500_nJy", "f1500_nJy", "nJy", "Power-law flux density at rest 1500 AA"),
              ("n_filters", "uv_n_filters", "", "Filters used in the power-law fit"),
              ("filters_used", "uv_filters_used", "", "Filters used, ';' separated"),
              ("fit_flag", "uv_fit_flag", "", "ok, too_few_filters or fit_failed"),
              ("good_muv", "good_muv", "", "Fit ok and M_UV_err < 0.5"),
              ("good_beta", "good_beta", "", "Fit ok and beta_err < 0.5"),
              ("z", "uv_z", "", "Redshift used for the UV fit"),
              ("z_type", "uv_z_type", "", "Which redshift the UV fit used")]),
    "lines_ha": dict(
        file="jwst_catalogs/Ha_summary_by_JELS_ID.csv",
        cols=[("Ha_flux_best", "ha_flux_best", "erg/s/cm2", "Observed [NII]-deblended Halpha flux, best-S/N grating, no corrections"),
              ("Ha_flux_best_err", "ha_flux_best_err", "erg/s/cm2", "Error on ha_flux_best"),
              ("Ha_flux_best_snr", "ha_flux_best_snr", "", "S/N of ha_flux_best"),
              ("Ha_flux_best_grating", "ha_flux_best_grating", "", "Grating of ha_flux_best"),
              ("Ha_fwhm_int_best", "ha_fwhm_int_kms", "km/s", "Intrinsic Halpha FWHM"),
              ("Ha_fwhm_int_best_err", "ha_fwhm_int_kms_err", "km/s", "Error on the intrinsic Halpha FWHM"),
              ("Ha_fwhm_obs_best", "ha_fwhm_obs_kms", "km/s", "Observed Halpha FWHM"),
              ("Ha_fwhm_unresolved_best", "ha_fwhm_unresolved", "", "Halpha unresolved, intrinsic FWHM is a limit"),
              ("Ha_fwhm_best_grating", "ha_fwhm_grating", "", "Grating of the Halpha FWHM")]),
    "lines_hb": dict(
        file="jwst_catalogs/Hbeta_summary_by_JELS_ID.csv",
        cols=[("Hbeta_flux_best", "hb_flux_best", "erg/s/cm2", "Observed Hbeta flux, best-S/N grating, no corrections"),
              ("Hbeta_flux_best_err", "hb_flux_best_err", "erg/s/cm2", "Error on hb_flux_best"),
              ("Hbeta_flux_best_snr", "hb_flux_best_snr", "", "S/N of hb_flux_best"),
              ("Hbeta_flux_best_grating", "hb_flux_best_grating", "", "Grating of hb_flux_best")]),
    "ha_corr": dict(
        file="jwst_catalogs/ha_hb_flux_corrections.csv",   # --ha-csv overrides
        cols=[("grating_ha", "ha_corr_grating", "", "Grating the corrected Halpha flux was taken from"),
              ("same_grating", "ha_hb_same_grating", "", "Halpha and Hbeta from the same spectrum"),
              ("slit_reference", "ha_slit_reference", "", "Photometry the slit factor corrects to"),
              ("ha_flux_uncorr", "ha_flux_uncorr", "erg/s/cm2", "Observed Halpha flux fed into the corrections"),
              ("ha_flux_uncorr_err", "ha_flux_uncorr_err", "erg/s/cm2", "Error on ha_flux_uncorr"),
              ("ha_slit_factor", "ha_slit_factor", "", "Slit-loss factor applied to Halpha"),
              ("ha_slit_source", "ha_slit_source", "", "direct, sample_median or dja"),
              ("ha_flux_slitcorr", "ha_flux_slitcorr", "erg/s/cm2", "Slit-loss corrected Halpha flux"),
              ("ha_flux_slitcorr_err", "ha_flux_slitcorr_err", "erg/s/cm2", "Error on ha_flux_slitcorr"),
              ("balmer_ratio", "balmer_ratio", "", "Halpha/Hbeta used for the dust correction"),
              ("ebv_neb", "ebv_neb", "mag", "Nebular E(B-V) from the Balmer decrement, Calzetti+00"),
              ("ebv_neb_err_lo", "ebv_neb_err_lo", "mag", "Lower error on ebv_neb"),
              ("ebv_neb_err_hi", "ebv_neb_err_hi", "mag", "Upper error on ebv_neb"),
              ("A_ha", "A_ha", "mag", "Attenuation at Halpha"),
              ("ha_flux_fullcorr", "ha_flux_fullcorr", "erg/s/cm2", "Slit-loss and dust corrected Halpha flux"),
              ("ha_flux_fullcorr_err_lo", "ha_flux_fullcorr_err_lo", "erg/s/cm2", "Lower error on ha_flux_fullcorr"),
              ("ha_flux_fullcorr_err_hi", "ha_flux_fullcorr_err_hi", "erg/s/cm2", "Upper error on ha_flux_fullcorr"),
              ("ha_flux_fullcorr_dja", "ha_flux_fullcorr_dja", "erg/s/cm2", "Dust corrected Halpha with the DJA path-loss only"),
              ("dust_method", "ha_dust_method", "", "balmer, or blank if no dust correction"),
              ("balmer_basis", "ha_balmer_basis", "", "observed_same_grating or slitcorr")]),
    # Placeholders. Column names are the ones the new scripts will be asked to
    # write, so filling these needs no edit here beyond the file path.
    "sed": dict(
        file="sed/sed_properties_by_JELS_ID.csv",
        placeholder=True,
        cols=[("sed_logM", "sed_logM", "log Msun", "Stellar mass (BAGPIPES, aperture corrected)"),
              ("sed_logM_err_lo", "sed_logM_err_lo", "dex", "Lower error"),
              ("sed_logM_err_hi", "sed_logM_err_hi", "dex", "Upper error"),
              ("sed_sfr10", "sed_sfr10", "Msun/yr", "SFR averaged over 10 Myr"),
              ("sed_sfr10_err_lo", "sed_sfr10_err_lo", "Msun/yr", "Lower error"),
              ("sed_sfr10_err_hi", "sed_sfr10_err_hi", "Msun/yr", "Upper error"),
              ("sed_ssfr10", "sed_ssfr10", "1/yr", "sSFR, SFR10 / M*"),
              ("sed_av", "sed_av", "mag", "V-band attenuation from the SED fit"),
              ("sed_av_err_lo", "sed_av_err_lo", "mag", "Lower error"),
              ("sed_av_err_hi", "sed_av_err_hi", "mag", "Upper error"),
              ("sed_ebv", "sed_ebv", "mag", "E(B-V) = A_V / 4.05")]),
    "fesc": dict(
        file="MUSE_catalogs/lya_fesc_by_JELS_ID.csv",
        placeholder=True,
        cols=[("fesc_lya", "fesc_lya", "", "Lya escape fraction, F_Lya / (8.7 F_Halpha,corr)"),
              ("fesc_lya_err_lo", "fesc_lya_err_lo", "", "Lower error"),
              ("fesc_lya_err_hi", "fesc_lya_err_hi", "", "Upper error"),
              ("fesc_lya_is_limit", "fesc_lya_is_limit", "", "True if fesc_lya is a 5 sigma upper limit"),
              ("fesc_ha_dust", "fesc_ha_dust", "", "Dust correction behind the Halpha flux, balmer or sed")]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build the P2 master catalogue.")
    p.add_argument("--p2-root", default=pc.P2_ROOT)
    p.add_argument("--out-dir", default=None,
                   help="Default <p2-root>/master_catalog")
    p.add_argument("--ha-csv", default=None,
                   help="Halpha corrections CSV. Default "
                        "<p2-root>/jwst_catalogs/ha_hb_flux_corrections.csv")
    p.add_argument("--sed-csv", default=None, help="Override the SED placeholder path.")
    p.add_argument("--fesc-csv", default=None, help="Override the fesc placeholder path.")
    p.add_argument("--labels-csv", default=None,
                   help="Default <p2-root>/MUSE_catalogs/lya_manual_labels.csv")
    p.add_argument("--oiii-snr-min", type=float, default=pc.SNR_MIN["OIII"])
    p.add_argument("--ha-snr-min", type=float, default=pc.SNR_MIN["Ha"])
    p.add_argument("--muse-flux-unit", type=float, default=MUSE_FLUX_UNIT,
                   help="erg/s/cm2 per MUSE flux unit, for lya_flux_cgs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and compare with the existing master, write nothing.")
    a = p.parse_args()
    a.p2_root = os.path.abspath(a.p2_root)
    a.out_dir = os.path.abspath(a.out_dir or os.path.join(a.p2_root, "master_catalog"))
    a.labels_csv = os.path.abspath(a.labels_csv or os.path.join(
        a.p2_root, "MUSE_catalogs", "lya_manual_labels.csv"))
    return a


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def file_stamp(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


class Dictionary:
    """Collects the column dictionary rows as blocks are added."""

    def __init__(self):
        self.rows = []

    def add(self, col, block, src_file, src_col, unit, desc):
        self.rows.append(dict(column=col, block=block, source_file=src_file,
                              source_column=src_col, unit=unit, description=desc))

    def frame(self, keep):
        df = pd.DataFrame(self.rows)
        df = df[df["column"].isin(keep)].drop_duplicates("column")
        return df.set_index("column").reindex(keep).reset_index()


def read_ids(path):
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(int)
    dup = df["ID"].duplicated()
    if dup.any():
        print(f"    [WARN] {int(dup.sum())} repeated IDs in {path}, first kept")
        df = df[~dup]
    return df.set_index("ID")


def add_simple_block(master, name, spec, root, dct, report, path_override=None):
    path = os.path.abspath(path_override or resolve(root, spec["file"]))
    dst_cols = [d for _, d, _, _ in spec["cols"]]
    for src, dst, unit, desc in spec["cols"]:
        dct.add(dst, name, path, src, unit, desc)
    if not os.path.exists(path):
        tag = "placeholder, not built yet" if spec.get("placeholder") else "MISSING"
        report.append((name, path, tag, "", 0))
        empty = pd.DataFrame(np.nan, index=master.index, columns=dst_cols)
        return master.join(empty).copy()
    df = read_ids(path)
    missing = [s for s, _, _, _ in spec["cols"] if s not in df.columns]
    if missing:
        print(f"    [WARN] {name}: columns not in {path}: {missing}")
    sub = pd.DataFrame(index=df.index)
    for src, dst, _, _ in spec["cols"]:
        sub[dst] = df[src] if src in df.columns else np.nan
    master = master.join(sub, how="left").copy()
    report.append((name, path, "ok", file_stamp(path), int(master.index.isin(df.index).sum())))
    return master


def to_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip().lower() in ("true", "1", "1.0")


# ---------------------------------------------------------------------------
# Blocks with their own logic
# ---------------------------------------------------------------------------

def sample_block(root, dct, report):
    path = os.path.join(root, "jwst_catalogs", "grating_sources_by_JELS_ID.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Base catalogue missing: {path}")
    g = read_ids(path)
    m = pd.DataFrame(index=g.index)
    m.index.name = "ID"
    cols = [("ra", "ra", "deg", "Right ascension (JELS)"),
            ("dec", "dec", "deg", "Declination (JELS)")]
    for gr in pc.GRATINGS:
        cols.append((gr, f"has_{gr}", "", f"Source has a {gr} spectrum"))
    for gr in pc.GRATINGS:
        cols.append((f"z_{gr}", f"z_dja_{gr}", "", f"DJA redshift from {gr}"))
    cols += [("z_av", "z_av", "", "Mean DJA redshift over gratings"),
             ("in_muse", "in_muse", "", "Position falls on valid MUSE data"),
             ("edge", "edge", "arcsec", "0 if >= 6 arcsec from a MUSE edge, else the distance"),
             ("duplicate", "duplicate", "", "0 = unique, shared nonzero number for a duplicate pair"),
             ("AO_block", "AO_block", "", "Lya at z_av falls in the sodium AO gap")]
    for src, dst, unit, desc in cols:
        m[dst] = g[src] if src in g.columns else np.nan
        dct.add(dst, "sample", path, src, unit, desc)
    report.append(("sample", path, "ok", file_stamp(path), len(m)))

    good_path = os.path.join(root, "jwst_catalogs", "grating_sources_by_JELS_ID_good.csv")
    dct.add("in_good", "sample", good_path, "ID", "",
            "In the in-MUSE, non-edge, non-AO, deduplicated subsample")
    if os.path.exists(good_path):
        good = set(pd.read_csv(good_path)["ID"].astype(int))
        m["in_good"] = m.index.isin(good)
        report.append(("sample_good", good_path, "ok", file_stamp(good_path), len(good)))
    else:
        m["in_good"] = np.nan
        report.append(("sample_good", good_path, "MISSING", "", 0))
    return m


def zsys_block(m, root, cuts, dct, report):
    path = os.path.join(root, "jwst_catalogs", "systemic_redshifts_by_JELS_ID.csv")
    new = ["z_sys", "z_sys_err", "z_sys_snr", "z_sys_line", "z_sys_grating",
           "z_sys_quality", "z_dja", "z_best", "z_best_source"]
    desc = {
        "z_sys": ("", f"Systemic z: {pc.rule_text(cuts)}"),
        "z_sys_err": ("", "Error on z_sys"),
        "z_sys_snr": ("", "S/N of the line that set z_sys"),
        "z_sys_line": ("", "Line that set z_sys"),
        "z_sys_grating": ("", "Grating of the line that set z_sys"),
        "z_sys_quality": ("", "a = [OIII], b = Halpha, d = no z_sys"),
        "z_dja": ("", "DJA z of the z_sys grating, or of the fallback grating if no z_sys (p2_common.fallback_grating)"),
        "z_best": ("", "z_sys, or z_dja where there is no z_sys (what the MUSE steps use)"),
        "z_best_source": ("", "z_sys or z_dja"),
    }
    for c in new:
        dct.add(c, "zsys", path, "(computed)", *desc[c])
    line_cols = []
    for line in pc.LINE_PRIORITY:
        for suff, d in (("", "redshift"), ("_err", "error"), ("_snr", "S/N")):
            c = f"z_{line}{suff}"
            line_cols.append(c)
            dct.add(c, "zsys", path, c, "", f"{line} {d}, best successful grating")

    oiii_path = os.path.join(root, "jwst_catalogs", "OIII_results_by_JELS_ID.csv")
    oiii = read_ids(oiii_path) if os.path.exists(oiii_path) else pd.DataFrame()
    g = read_ids(os.path.join(root, "jwst_catalogs", "grating_sources_by_JELS_ID.csv"))
    if not os.path.exists(path):
        report.append(("zsys", path, "MISSING", "", 0))
        for c in new + line_cols:
            m[c] = np.nan
        sysd = pd.DataFrame()
    else:
        sysd = read_ids(path)
        report.append(("zsys", path, "ok", file_stamp(path), int(m.index.isin(sysd.index).sum())))
        for c in line_cols:
            m[c] = sysd[c].reindex(m.index) if c in sysd.columns else np.nan

    picks = []
    for sid in m.index:
        if sid in sysd.index:
            z, ze, zs, line, gr = pc.pick_zsys(sysd.loc[sid], cuts)
        else:
            z, ze, zs, line, gr = np.nan, np.nan, np.nan, None, None
        if gr is None:
            gr = pc.fallback_grating(g.loc[sid], oiii.loc[sid] if sid in oiii.index else None)
        zdja = m.at[sid, f"z_dja_{gr}"] if gr else np.nan
        picks.append(dict(ID=sid, z_sys=z, z_sys_err=ze, z_sys_snr=zs, z_sys_line=line,
                          z_sys_grating=gr if line else None,
                          z_sys_quality=pc.zsys_quality(line), z_dja=zdja,
                          z_best=z if np.isfinite(z) else zdja,
                          z_best_source="z_sys" if np.isfinite(z) else
                          ("z_dja" if np.isfinite(zdja) else None)))
    p = pd.DataFrame(picks).set_index("ID")
    for c in new:
        m[c] = p[c]
    # Put the per-line columns after the chosen z_sys.
    order = [c for c in m.columns if c not in line_cols] + line_cols
    return m[order]


def muse_provenance_block(m, root, dct, report):
    path = os.path.join(root, "MUSE_catalogs", "lya_properties_mc.csv")
    cols = {
        "zsys_muse": ("", "z used for the MUSE contsub, extraction and Lya fit"),
        "zsys_muse_source": ("", "z_sys or z_dja, as recorded by the Lya fitter"),
        "zsys_mismatch_kms": ("km/s", "zsys_muse relative to z_best"),
    }
    for c, (u, d) in cols.items():
        dct.add(c, "provenance", path, "z_sys / z_sys_source", u, d)
    if os.path.exists(path):
        lp = read_ids(path)
        m["zsys_muse"] = lp["z_sys"].reindex(m.index)
        m["zsys_muse_source"] = lp["z_sys_source"].reindex(m.index)
    else:
        m["zsys_muse"] = np.nan
        m["zsys_muse_source"] = np.nan
    m["zsys_mismatch_kms"] = pc.dz_to_kms(m["zsys_muse"], m["z_best"])
    return m


def detection_block(m, root, dct, report):
    path = os.path.join(root, "MUSE_catalogs", "false_pos_snr_percentiles.csv")
    for c, pct in (("lya_det_98", "pct_98.0"), ("lya_det_99p5", "pct_99.5")):
        dct.add(c, "lya_det", path, pct, "",
                f"lya_peak_snr above the {pct[4:]} percentile of the false-positive S/N")
    if not os.path.exists(path):
        report.append(("lya_det_thresholds", path, "MISSING", "", 0))
        m["lya_det_98"] = np.nan
        m["lya_det_99p5"] = np.nan
        return m
    pct = pd.read_csv(path).iloc[0]
    t98, t995 = float(pct["pct_98.0"]), float(pct["pct_99.5"])
    report.append(("lya_det_thresholds", path, "ok", file_stamp(path), 0))
    print(f"    detection thresholds  98% S/N {t98:.2f}   99.5% S/N {t995:.2f}")
    snr = m["lya_peak_snr"]
    m["lya_det_98"] = (snr >= t98).astype("boolean").mask(snr.isna())
    m["lya_det_99p5"] = (snr >= t995).astype("boolean").mask(snr.isna())
    return m


def tier_block(m, root, dct, report):
    tdir = os.path.join(root, "MUSE_catalogs")
    crit = ["zsys_is_OIII", "dv_positive", "dv_err<120", "fwhm>lsf", "snr>3", "not_oob"]
    parts = []
    for t in TIERS:
        path = os.path.join(tdir, f"lya_group_{t}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["tier"] = t
            parts.append(df)
    src = os.path.join(tdir, "lya_group_<tier>.csv")
    dct.add("tier", "tier", src, "(file name)", "", "Automatic tier, gold to bad")
    dct.add("n_criteria_failed", "tier", src, "n_criteria_failed", "", "Number of the six criteria failed")
    for c in crit:
        dct.add(f"pass_{c}", "tier", src, f"pass_{c}", "", f"Passes the {c} criterion")
    if not parts:
        report.append(("tier", src, "MISSING", "", 0))
        for c in ["tier", "n_criteria_failed"] + [f"pass_{c}" for c in crit]:
            m[c] = np.nan
        return m, set()
    t = pd.concat(parts, ignore_index=True)
    t["ID"] = t["ID"].astype(int)
    t = t.drop_duplicates("ID").set_index("ID")
    keep = ["tier", "n_criteria_failed"] + [f"pass_{c}" for c in crit]
    for c in keep:
        m[c] = t[c].reindex(m.index) if c in t.columns else np.nan
    report.append(("tier", src, "ok", "", int(m.index.isin(t.index).sum())))
    return m, set(t.index)


def labels_block(m, labels_csv, root, dct, report):
    for c, d in (("manual", "By-eye label a, b, c or d"),
                 ("tier_when_labelled", "Automatic tier when the label was given"),
                 ("manual_note", "Free-text note from the label file")):
        dct.add(c, "labels", labels_csv, c if c != "manual_note" else "note", "", d)
    if os.path.exists(labels_csv):
        lab = read_ids(labels_csv)
        m["manual"] = lab["manual"].reindex(m.index)
        m["tier_when_labelled"] = lab.get("tier_when_labelled", pd.Series(dtype=object)).reindex(m.index)
        m["manual_note"] = lab.get("note", pd.Series(dtype=object)).reindex(m.index)
        report.append(("labels", labels_csv, "ok", file_stamp(labels_csv), int(lab["manual"].notna().sum())))
        return m
    # Fall back to the manual column of the tier CSVs.
    print(f"    [WARN] no label file at {labels_csv}. Run update_manual_labels.py. "
          f"Using the tier CSVs' manual column for now.")
    labs = {}
    for t in TIERS:
        path = os.path.join(root, "MUSE_catalogs", f"lya_group_{t}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "manual" in df.columns:
                for sid, lab in zip(df["ID"].astype(int), df["manual"]):
                    if isinstance(lab, str) and lab.strip():
                        labs[sid] = (lab.strip().lower(), t)
    m["manual"] = [labs.get(i, (np.nan, np.nan))[0] for i in m.index]
    m["tier_when_labelled"] = [labs.get(i, (np.nan, np.nan))[1] for i in m.index]
    m["manual_note"] = np.nan
    report.append(("labels", "tier CSVs (fallback)", "ok", "", len(labs)))
    return m


def primary_flag(m, tier_ids, dct):
    """is_primary: False only for the dropped member of a duplicate pair.

    Within a pair the member kept by group_lya_sample.py (the one in the tier
    CSVs) is primary. If neither member is tiered, the one with the smaller
    delta_v_err_kms, then the lower ID, is primary.
    """
    dct.add("is_primary", "sample", "(computed)", "duplicate", "",
            "False for the dropped member of a duplicate pair")
    m["is_primary"] = True
    dup = pd.to_numeric(m["duplicate"], errors="coerce").fillna(0).astype(int)
    for grp in sorted(set(dup[dup > 0])):
        ids = list(m.index[dup == grp])
        tiered = [i for i in ids if i in tier_ids]
        if tiered:
            keep = tiered[0]
        else:
            err = pd.to_numeric(m.loc[ids, "delta_v_err_kms"], errors="coerce").fillna(np.inf)
            keep = sorted(ids, key=lambda i: (err[i], i))[0]
        for i in ids:
            m.at[i, "is_primary"] = (i == keep)
    return m


# ---------------------------------------------------------------------------
# Comparison with the previous master
# ---------------------------------------------------------------------------

def roundtrip(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def compare(old_path, new_df, max_ids=10):
    if not os.path.exists(old_path):
        print("  no previous master catalogue, nothing to compare")
        return
    old = pd.read_csv(old_path).set_index("ID")
    new = roundtrip(new_df).set_index("ID")
    print(f"  previous  {old_path}  ({file_stamp(old_path)})")
    add_c = [c for c in new.columns if c not in old.columns]
    del_c = [c for c in old.columns if c not in new.columns]
    add_i = sorted(set(new.index) - set(old.index))
    del_i = sorted(set(old.index) - set(new.index))
    if add_c:
        print(f"  new columns      {add_c}")
    if del_c:
        print(f"  removed columns  {del_c}")
    if add_i:
        print(f"  new sources      {add_i}")
    if del_i:
        print(f"  removed sources  {del_i}")
    ids = old.index.intersection(new.index)
    n_changed_cols = 0
    for c in [c for c in new.columns if c in old.columns]:
        a, b = old.loc[ids, c], new.loc[ids, c]
        an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        numeric = (an.notna() == a.notna()).all() and (bn.notna() == b.notna()).all()
        if numeric:
            same = (an.isna() & bn.isna()) | np.isclose(an, bn, rtol=1e-9, atol=0.0, equal_nan=False)
        else:
            same = (a.isna() & b.isna()) | (a.astype(str) == b.astype(str))
        changed = list(ids[~same.to_numpy()])
        if changed:
            n_changed_cols += 1
            more = f" ... +{len(changed) - max_ids}" if len(changed) > max_ids else ""
            print(f"  {c:28s} {len(changed):3d} changed  {changed[:max_ids]}{more}")
    if not (add_c or del_c or add_i or del_i or n_changed_cols):
        print("  no changes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    a = parse_args()
    root = a.p2_root
    out_csv = os.path.join(a.out_dir, "p2_master_catalog.csv")
    out_cols = os.path.join(a.out_dir, "p2_master_columns.csv")
    print("[CONFIG]")
    print(f"  P2 root            {root}")
    print(f"  master catalogue   {out_csv}")
    print(f"  column dictionary  {out_cols}")
    print(f"  label file         {a.labels_csv}")
    print("")
    print("[READ]")

    dct = Dictionary()
    report = []
    m = sample_block(root, dct, report)
    cuts = {"OIII": a.oiii_snr_min, "Ha": a.ha_snr_min}
    m = zsys_block(m, root, cuts, dct, report)
    m = add_simple_block(m, "lya_det", BLOCKS["lya_det_snr"], root, dct, report)
    m = detection_block(m, root, dct, report)
    m = add_simple_block(m, "lya_det", BLOCKS["lya_det_offsets"], root, dct, report)
    m = add_simple_block(m, "lya_fit", BLOCKS["lya_fit"], root, dct, report)
    m["lya_flux_cgs"] = m["lya_flux"] * a.muse_flux_unit
    err = m["lya_flux_err_mc"].where(m["lya_flux_err_mc"].notna(), m["lya_flux_err_fit"])
    m["lya_flux_cgs_err"] = err * a.muse_flux_unit
    dct.add("lya_flux_cgs", "lya_fit", "(computed)", "flux_fit", "erg/s/cm2",
            f"lya_flux x {a.muse_flux_unit:g}, 0.6 arcsec aperture, not total")
    dct.add("lya_flux_cgs_err", "lya_fit", "(computed)", "flux_fit_err_mc", "erg/s/cm2",
            "MC error (covariance error where MC is missing)")
    m = muse_provenance_block(m, root, dct, report)
    m = add_simple_block(m, "delta_v", BLOCKS["delta_v"], root, dct, report)
    m, tier_ids = tier_block(m, root, dct, report)
    m = labels_block(m, a.labels_csv, root, dct, report)
    m = primary_flag(m, tier_ids, dct)
    m = add_simple_block(m, "uv", BLOCKS["uv"], root, dct, report)
    m = add_simple_block(m, "lines", BLOCKS["lines_ha"], root, dct, report)
    m = add_simple_block(m, "lines", BLOCKS["lines_hb"], root, dct, report)
    m = add_simple_block(m, "ha_corr", BLOCKS["ha_corr"], root, dct, report, a.ha_csv)
    m = add_simple_block(m, "sed", BLOCKS["sed"], root, dct, report, a.sed_csv)
    m = add_simple_block(m, "fesc", BLOCKS["fesc"], root, dct, report, a.fesc_csv)

    # Column order: ID first, then the provenance columns at the end.
    qc = ["zsys_muse", "zsys_muse_source", "zsys_mismatch_kms", "z_sys_dv"]
    head = ["ra", "dec", "is_primary"]
    order = head + [c for c in m.columns if c not in qc + head] + qc
    m = m[order].reset_index()
    cols = dct.frame(list(m.columns[1:]))
    cols = pd.concat([pd.DataFrame([dict(column="ID", block="sample", source_file="",
                                         source_column="ID", unit="", description="JELS ID")]),
                      cols], ignore_index=True)

    for name, path, status, stamp, n in report:
        print(f"  {name:20s} {status:28s} {stamp:16s} {n:4d}  {path}")
    print("")

    print("[CHANGES vs previous master]")
    compare(out_csv, m)
    print("")

    in_muse = m["in_muse"].map(to_bool)
    prim = m["is_primary"].map(to_bool)
    print("[SUMMARY]")
    print(f"  sources                    {len(m)}")
    print(f"  in MUSE / primary in MUSE  {int(in_muse.sum())} / {int((in_muse & prim).sum())}")
    print(f"  with z_sys                 {int(m['z_sys'].notna().sum())}  "
          + ", ".join(f"{k} {v}" for k, v in m["z_sys_line"].value_counts().items()))
    print(f"  Lya detected 98 / 99.5     {int((m['lya_det_98'] == 1).sum())} / "
          f"{int((m['lya_det_99p5'] == 1).sum())}")
    print(f"  with Delta_v               {int(m['delta_v_kms'].notna().sum())}")
    print("  manual labels              " + ", ".join(
        f"{k} {v}" for k, v in m["manual"].value_counts().sort_index().items()))
    print(f"  with M_UV (good)           {int(m['good_muv'].map(to_bool).sum())}")
    print(f"  with corrected Halpha      {int(m['ha_flux_fullcorr'].notna().sum())}")
    print(f"  with SED / fesc            {int(m['sed_logM'].notna().sum())} / "
          f"{int(m['fesc_lya'].notna().sum())}")

    if a.dry_run:
        print("\n[DRY RUN] nothing written")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    m.to_csv(out_csv, index=False)
    cols.to_csv(out_cols, index=False)
    print("")
    print(f"written  {out_csv}  ({len(m)} rows, {m.shape[1]} columns)")
    print(f"written  {out_cols}")


if __name__ == "__main__":
    main()
