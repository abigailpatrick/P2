#!/usr/bin/env python
"""Fit the [OIII] 4959,5007 doublet for ONE source, for manual review.

Companion to lime_OIII_jointfit.py. Same fit recipe (tied doublet blend, with
fallback to a single 5007 fit), same outputs, but for a single JELS ID so the
sources that failed the batch run can be re-fitted with per-source overrides
and inspected one at a time.

Given an ID, this walks all four grating folders under JWST_SPECTRA_ROOT,
finds every <ID>_<grating>_spectra_lime.fits that exists, and fits each one.
It then patches just that source's row in both batch CSVs (every other row is
left untouched) and writes figures to the same OIII_fits/<ID>/ dir.

Per-source overrides (all optional)
-----------------------------------
--z FLOAT           Corrected absolute redshift used to build the band. The
                    DJA z is still kept in z_<grating>; the value actually
                    used is recorded in a new z_used_<grating> column. The
                    fitted z_OIII always comes from the real line centre, so
                    it stays honest regardless of --z.
--line-margin FLOAT Rest-frame Angstrom the line region extends beyond each
                    outer line. Default matches the batch script (15).
--cont-gap FLOAT    Gap between line region and each continuum flank (5).
--cont-width FLOAT  Width of each continuum flank (20).
--snr-min FLOAT     Minimum snr_line for a detection (13).
--centre-err-max FLOAT  Max fitted centre error in Angstrom (2.5).
--force-single      Force the single-5007 fit even where 4959 is in range.
--gratings LIST     Comma-separated subset to fit, e.g. G235M,G395H. Default
                    is every grating the source has coverage in.
--dry-run           Fit and plot, print everything, but do NOT write the CSVs.

Examples
--------
python lime_OIII_single_review.py 59737
python lime_OIII_single_review.py 59737 --z 5.9412
python lime_OIII_single_review.py 59737 --z 5.9412 --line-margin 25
python lime_OIII_single_review.py 59737 --gratings G235M --force-single --dry-run

The band shift you usually want is --z. Widen the windows only when needed.
"""

import os
import sys
import re
import glob
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lime


# ----------------------------------------------------------------------
# Fixed configuration. Matches lime_OIII_jointfit.py.
# ----------------------------------------------------------------------
CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"

# Output CSVs, patched in place (one row updated per run).
RESULTS_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/OIII_results_by_JELS_ID.csv"
SUMMARY_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/OIII_summary_by_JELS_ID.csv"

# Figures root. One subdir per source ID underneath.
FIG_ROOT = os.path.join(JWST_SPECTRA_ROOT, "OIII_fits")

# Vacuum rest wavelengths of the [OIII] doublet in Angstrom.
OIII_5007_VAC = 5006.843
OIII_4959_VAC = 4958.911

# Theoretical 5007/4959 flux ratio, fixed by atomic transition probabilities.
OIII_RATIO_THEORY = 2.98

# LiMe line labels.
OIII_5007_LABEL = "O3_5007A"
OIII_4959_LABEL = "O3_4959A"
OIII_BLEND_LABEL = "O3_5007A_b"

# --- Default band geometry, rest-frame Angstrom. CLI can override. ---
DEFAULT_LINE_MARGIN = 15.0     # w3-w4 extends this far beyond each outer line
DEFAULT_CONT_GAP = 5.0         # gap between line region and each continuum flank
DEFAULT_CONT_WIDTH = 20.0      # width of each continuum flank
# --------------------------------------------------------------------

# The four gratings and their catalogue files.
GRATINGS = ["G235M_F170LP", "G235H_F170LP", "G395M_F290LP", "G395H_F290LP"]

# DJA catalogue column names.
DJA_Z_COL = "z"
DJA_ID_COL = "ID"

# --- Default success criteria. Both must pass. CLI can override. ---
DEFAULT_SNR_MIN = 13.0               # minimum LiMe snr_line for a detection
DEFAULT_CENTRE_ERR_MAX_AA = 2.5      # maximum fitted centre error in Angstrom
# ------------------------------------------------------------------
# ----------------------------------------------------------------------


def grating_short(grating):
    """G235M_F170LP -> G235M, for column and filename use."""
    return grating.split("_")[0]


def catalogue_path(grating):
    return os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{grating}_match_0p3as.fits")


def spectrum_path(spectra_root, src_id, grating):
    return os.path.join(spectra_root, grating,
                        f"{src_id}_{grating}_spectra_lime.fits")


def get_dja_redshift(cat, src_id):
    """DJA redshift for this ID from an already-loaded catalogue table."""
    mask = cat[DJA_ID_COL] == src_id
    if mask.sum() == 0:
        return None
    return float(cat[DJA_Z_COL][mask][0])


def load_lime_spectrum(lime_path, redshift):
    """Load the _lime.fits SPECTRUM table into a lime.Spectrum."""
    with fits.open(lime_path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave, flux, err = wave[good], flux[good], err[good]

    norm_flux = np.nanmedian(np.abs(flux[flux > 0])) if np.any(flux > 0) else 1e-20

    spec = lime.Spectrum(
        input_wave=wave,
        input_flux=flux,
        input_err=err,
        redshift=redshift,
        units_wave="AA",
        units_flux="FLAM",
        norm_flux=norm_flux,
    )
    return spec, wave


def build_doublet_band(line_margin, cont_gap, cont_width):
    """One-row bands frame whose line region spans BOTH [OIII] lines."""
    w3 = OIII_4959_VAC - line_margin
    w4 = OIII_5007_VAC + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [OIII_5007_VAC],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[OIII_5007_LABEL],
    )


def single_5007_band(line_margin, cont_gap, cont_width):
    """A plain 5007-only band for the fallback fit."""
    w3 = OIII_5007_VAC - line_margin
    w4 = OIII_5007_VAC + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [OIII_5007_VAC],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[OIII_5007_LABEL],
    )


def band_in_range(band_row, wave, zf):
    lo = float(band_row["w1"]) * zf
    hi = float(band_row["w6"]) * zf
    return (lo >= np.nanmin(wave)) and (hi <= np.nanmax(wave))


def line_in_range(centre_rest, wave, zf, margin):
    lo = (centre_rest - margin) * zf
    hi = (centre_rest + margin) * zf
    return (lo >= np.nanmin(wave)) and (hi <= np.nanmax(wave))


def centre_to_redshift(centre_obs_aa):
    return centre_obs_aa / OIII_5007_VAC - 1.0


def centre_err_to_redshift_err(centre_err_aa):
    if centre_err_aa is None or not np.isfinite(centre_err_aa):
        return np.nan
    return centre_err_aa / OIII_5007_VAC


def build_joint_fit_cfg():
    """fit_cfg for the fully tied [OIII] blend."""
    return {
        OIII_BLEND_LABEL: f"{OIII_4959_LABEL}+{OIII_5007_LABEL}",
        f"{OIII_4959_LABEL}_amp": {"expr": f"{OIII_5007_LABEL}_amp/{OIII_RATIO_THEORY}"},
        f"{OIII_4959_LABEL}_kinem": OIII_5007_LABEL,
    }


def get_5007_row(spec):
    """Return centre, centre_err, and detection S/N for 5007 from the frame.

    S/N is LiMe's own snr_line, read off the 5007 row (amplitude-based, Rola
    et al. definition), not the band-integrated flux over its error, which
    washes out narrow lines in the wide doublet band. Falls back to
    profile_flux / its error only if snr_line is absent.
    """
    if OIII_5007_LABEL not in spec.frame.index:
        return np.nan, np.nan, np.nan
    row = spec.frame.loc[OIII_5007_LABEL]
    cols = spec.frame.columns

    centre = float(row["center"]) if "center" in cols else np.nan
    centre_err = (
        float(row["center_err"])
        if "center_err" in cols and np.isfinite(row["center_err"]) else np.nan
    )

    snr = np.nan
    if "snr_line" in cols and np.isfinite(row["snr_line"]):
        snr = float(row["snr_line"])
    elif "profile_flux" in cols and "profile_flux_err" in cols:
        f, fe = row["profile_flux"], row["profile_flux_err"]
        if np.isfinite(f) and np.isfinite(fe) and fe > 0:
            snr = float(f / fe)
    return centre, centre_err, snr


def is_success(snr, centre_err_aa, snr_min, centre_err_max):
    """Both criteria must pass."""
    snr_ok = np.isfinite(snr) and snr >= snr_min
    err_ok = np.isfinite(centre_err_aa) and centre_err_aa <= centre_err_max
    return bool(snr_ok and err_ok)


def fit_one(lime_path, src_id, grating, z_fit, z_dja, cfg):
    """Fit [OIII] in one spectrum. cfg carries the per-run overrides.

    z_fit is the redshift used to build the band (DJA z, or --z if given).
    z_dja is the original DJA z, kept for the CSV. Returns a dict with keys
    z_oiii, z_oiii_err, snr, method, success, status. status is 'fitted' or
    'out_of_range'. Raises on genuine fit errors.
    """
    gr = grating_short(grating)
    print(f"\n--- ID {src_id}  {grating} ---")
    print(f"spectrum  {lime_path}")
    print(f"DJA z     {z_dja:.5f}")
    if abs(z_fit - z_dja) > 0:
        print(f"z used    {z_fit:.5f}   (override)")

    spec, wave = load_lime_spectrum(lime_path, z_fit)
    zf = 1.0 + z_fit

    if not line_in_range(OIII_5007_VAC, wave, zf, cfg["line_margin"]):
        print("OIII 5007 out of range")
        return {"status": "out_of_range"}

    # Figure dir for this source, grouping all gratings together.
    fig_dir = os.path.join(FIG_ROOT, str(src_id))
    os.makedirs(fig_dir, exist_ok=True)
    contsub_png = os.path.join(fig_dir, f"{src_id}_{grating}_contsub.png")
    fit_png = os.path.join(fig_dir, f"{src_id}_{grating}_OIII_fit.png")

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=contsub_png)

    doublet_band = build_doublet_band(cfg["line_margin"], cfg["cont_gap"],
                                      cfg["cont_width"])
    do_joint = (not cfg["force_single"]) and band_in_range(
        doublet_band.iloc[0], wave, zf)

    if do_joint:
        method = "joint_4959_5007"
        spec.fit.bands(OIII_BLEND_LABEL, bands=doublet_band,
                       fit_cfg=build_joint_fit_cfg())
    else:
        method = "single_5007"
        if cfg["force_single"]:
            print("single 5007 fit (forced)")
        else:
            print("4959 band out of range, single 5007 fit")
        spec.fit.bands(OIII_5007_LABEL,
                       bands=single_5007_band(cfg["line_margin"],
                                              cfg["cont_gap"],
                                              cfg["cont_width"]))

    centre, centre_err, snr = get_5007_row(spec)
    z_oiii = centre_to_redshift(centre)
    z_oiii_err = centre_err_to_redshift_err(centre_err)
    success = is_success(snr, centre_err, cfg["snr_min"], cfg["centre_err_max"])

    print(f"method    {method}")
    print(f"z_OIII    {z_oiii:.6f} +/- {z_oiii_err:.6f}")
    print(f"S/N       {snr:.2f}   center_err {centre_err} AA")
    print(f"success   {success}")

    spec.plot.bands(OIII_5007_LABEL, fname=fit_png)
    print(f"contsub   {contsub_png}")
    print(f"fit plot  {fit_png}")

    return {
        "status": "fitted",
        "z_oiii": z_oiii,
        "z_oiii_err": z_oiii_err,
        "snr": snr,
        "method": method,
        "success": success,
    }


def patch_results_csv(src_id, res_row, z_override_used):
    """Read RESULTS_CSV, update this source's row, write back.

    res_row is a dict of column->value for this source. If the CSV or the row
    does not exist yet, it is created. Any z_used_<gr> columns are added if a
    --z override was used and are not already present.
    """
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
    else:
        df = pd.DataFrame(columns=["ID"])

    # Make sure every column in res_row exists in the frame.
    for col in res_row:
        if col not in df.columns:
            df[col] = np.nan

    mask = df["ID"] == src_id
    if mask.any():
        for col, val in res_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([res_row])], ignore_index=True)

    df = df.sort_values("ID")
    df.to_csv(RESULTS_CSV, index=False)


def patch_summary_csv(src_id, sum_row):
    """Read SUMMARY_CSV, update this source's row, write back."""
    if os.path.exists(SUMMARY_CSV):
        df = pd.read_csv(SUMMARY_CSV)
    else:
        df = pd.DataFrame(columns=["ID"])

    for col in sum_row:
        if col not in df.columns:
            df[col] = np.nan

    mask = df["ID"] == src_id
    if mask.any():
        for col, val in sum_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)

    df = df.sort_values("ID")
    df.to_csv(SUMMARY_CSV, index=False)


def run_single(src_id, cfg):
    """Fit every available grating for one source and patch the CSVs."""
    spectra_root = cfg["spectra_root"]

    # Which gratings to attempt.
    want = cfg["gratings"] if cfg["gratings"] else GRATINGS

    # Accumulators for this one source.
    res_row = {"ID": src_id}
    sum_row = {"ID": src_id, "OIII_success": 0}
    gratings_seen = []

    any_found = False
    for grating in want:
        gr = grating_short(grating)
        lime_path = spectrum_path(spectra_root, src_id, grating)

        if not os.path.exists(lime_path):
            print(f"\n--- ID {src_id}  {grating} ---")
            print(f"no spectrum at {lime_path}, skipping")
            continue

        any_found = True

        # DJA z from the matching catalogue.
        cp = catalogue_path(grating)
        if not os.path.exists(cp):
            print(f"catalogue missing for {grating}: {cp}, skipping")
            continue
        cat = Table.read(cp)
        z_dja = get_dja_redshift(cat, src_id)
        if z_dja is None:
            print(f"\n--- ID {src_id}  {grating} ---")
            print(f"no DJA z in catalogue for this ID, skipping")
            continue

        # Redshift used to build the band: override if given, else DJA.
        z_fit = cfg["z_override"] if cfg["z_override"] is not None else z_dja

        gratings_seen.append(gr)
        res_row[f"z_{gr}"] = z_dja
        res_row[f"z_used_{gr}"] = z_fit

        try:
            r = fit_one(lime_path, src_id, grating, z_fit, z_dja, cfg)
        except Exception as err:
            print(f"FIT FAILED  {os.path.basename(lime_path)}  reason: {err}")
            sum_row[f"OIII_{gr}_success"] = False
            continue

        if r["status"] == "out_of_range":
            sum_row[f"OIII_{gr}_success"] = False
            continue

        res_row[f"z_OIII_{gr}"] = r["z_oiii"]
        res_row[f"z_OIII_{gr}_err"] = r["z_oiii_err"]
        res_row[f"z_OIII_{gr}_snr"] = r["snr"]

        sum_row[f"OIII_{gr}_success"] = r["success"]
        if r["success"]:
            sum_row["OIII_success"] += 1

    if not any_found:
        print(f"\nNo spectra found for ID {src_id} in any grating folder under")
        print(f"  {spectra_root}")
        print("Nothing written.")
        return

    sum_row["gratings"] = ", ".join(gratings_seen)

    # Report.
    print("\n" + "=" * 60)
    print(f"DONE  ID {src_id}")
    print(f"  gratings fitted     {sum_row['gratings']}")
    print(f"  successes           {sum_row['OIII_success']}")

    if cfg["dry_run"]:
        print("  dry run, CSVs NOT written")
        print(f"  would patch results {RESULTS_CSV}")
        print(f"  would patch summary {SUMMARY_CSV}")
        print(f"  figures root        {os.path.join(FIG_ROOT, str(src_id))}")
        print("=" * 60)
        return

    patch_results_csv(src_id, res_row, cfg["z_override"] is not None)
    patch_summary_csv(src_id, sum_row)

    print(f"  results csv         {RESULTS_CSV}")
    print(f"  summary csv         {SUMMARY_CSV}")
    print(f"  figures root        {os.path.join(FIG_ROOT, str(src_id))}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit [OIII] for one source with per-source overrides.")
    p.add_argument("src_id", type=int, help="JELS source ID")
    p.add_argument("--z", type=float, default=None, dest="z_override",
                   help="corrected absolute redshift for the band")
    p.add_argument("--line-margin", type=float, default=DEFAULT_LINE_MARGIN)
    p.add_argument("--cont-gap", type=float, default=DEFAULT_CONT_GAP)
    p.add_argument("--cont-width", type=float, default=DEFAULT_CONT_WIDTH)
    p.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    p.add_argument("--centre-err-max", type=float,
                   default=DEFAULT_CENTRE_ERR_MAX_AA)
    p.add_argument("--force-single", action="store_true",
                   help="force the single-5007 fit")
    p.add_argument("--gratings", type=str, default=None,
                   help="comma-separated subset, e.g. G235M,G395H")
    p.add_argument("--spectra-root", type=str, default=JWST_SPECTRA_ROOT)
    p.add_argument("--dry-run", action="store_true",
                   help="fit and plot but do not write the CSVs")
    return p.parse_args()


def resolve_gratings(subset):
    """Map a G235M,G395H style subset onto the full grating names."""
    if not subset:
        return None
    wanted = [s.strip().upper() for s in subset.split(",")]
    out = []
    for w in wanted:
        matches = [g for g in GRATINGS if grating_short(g) == w]
        if not matches:
            raise ValueError(f"unknown grating '{w}', choose from "
                             f"{[grating_short(g) for g in GRATINGS]}")
        out.append(matches[0])
    return out


def main():
    args = parse_args()
    cfg = {
        "z_override": args.z_override,
        "line_margin": args.line_margin,
        "cont_gap": args.cont_gap,
        "cont_width": args.cont_width,
        "snr_min": args.snr_min,
        "centre_err_max": args.centre_err_max,
        "force_single": args.force_single,
        "gratings": resolve_gratings(args.gratings),
        "spectra_root": args.spectra_root,
        "dry_run": args.dry_run,
    }

    print(f"single-source [OIII] review")
    print(f"  ID            {args.src_id}")
    print(f"  spectra root  {cfg['spectra_root']}")
    print(f"  results csv   {RESULTS_CSV}")
    print(f"  summary csv   {SUMMARY_CSV}")
    print(f"  figures root  {FIG_ROOT}")
    if cfg["z_override"] is not None:
        print(f"  z override    {cfg['z_override']:.5f}")
    if cfg["gratings"]:
        print(f"  gratings      {[grating_short(g) for g in cfg['gratings']]}")
    print(f"  line_margin {cfg['line_margin']}  cont_gap {cfg['cont_gap']}  "
          f"cont_width {cfg['cont_width']}")
    print(f"  snr_min {cfg['snr_min']}  centre_err_max {cfg['centre_err_max']}")
    if cfg["force_single"]:
        print("  force single 5007 fit")
    if cfg["dry_run"]:
        print("  DRY RUN, CSVs will not be written")

    os.makedirs(FIG_ROOT, exist_ok=True)
    run_single(args.src_id, cfg)


if __name__ == "__main__":
    main()