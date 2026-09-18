#!/usr/bin/env python
"""Fit the [OIII] 4959,5007 doublet across all four gratings and build result CSVs.

Written for LiMe 2.4.3. For every source in each grating catalogue this fits
the [OIII] doublet as a single blended profile (the _b suffix) over one wide
band whose line region spans BOTH lines, with 4959 tied to 5007 in amplitude
(fixed 2.98 ratio) and kinematics (_kinem). Where only 5007 is in range it
falls back to a single 5007 fit.

A source often has coverage in more than one grating (G235M, G235H, G395M,
G395H). Running all four maximises the chance of at least one clean [OIII]
detection per source, which is what the systemic redshift work needs.

Outputs, all written fresh (old files overwritten)
--------------------------------------------------
1. RESULTS_CSV: one row per source. Columns are the per-grating DJA redshift
   (z_<grating>), and for each grating actually fitted, the [OIII] redshift
   (z_OIII_<grating>), its error, and its detection S/N.
2. SUMMARY_CSV: one row per source. A 'gratings' string listing every grating
   the source has coverage in, a success flag per grating (True/False/blank),
   and OIII_success, the count of successful [OIII] fits for that source.
3. Figures under JWST_SPECTRA_ROOT/OIII_fits/<ID>/, grouping the contsub and
   line-fit plots for every grating of one source together.

Success. A fit is successful when BOTH the detection S/N clears SNR_MIN and
the fitted centre error is finite and below CENTRE_ERR_MAX_AA.

Usage
-----
python lime_OIII_jointfit.py                 # walks the default grating dirs
python lime_OIII_jointfit.py <parent_dir>    # override the spectra root

Filename convention expected: <ID>_<grating>_spectra_lime.fits
"""

import os
import sys
import re
import glob

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lime


# ----------------------------------------------------------------------
# Fixed configuration.
# ----------------------------------------------------------------------
CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"

# Output CSVs, written fresh each run.
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

# --- Band geometry, rest-frame Angstrom. ---
LINE_MARGIN = 15.0     # w3-w4 extends this far beyond each outer line
CONT_GAP = 5.0         # gap between line region and each continuum flank
CONT_WIDTH = 20.0      # width of each continuum flank
# -------------------------------------------

# The four gratings and their catalogue files.
GRATINGS = ["G235M_F170LP", "G235H_F170LP", "G395M_F290LP", "G395H_F290LP"]

# DJA catalogue column names.
DJA_Z_COL = "z"
DJA_ID_COL = "ID"

# --- Success criteria. Both must pass. ---
SNR_MIN = 13.0               # minimum LiMe snr_line for a detection
CENTRE_ERR_MAX_AA = 2.5      # maximum fitted centre error in Angstrom
# -----------------------------------------
# ----------------------------------------------------------------------


def grating_short(grating):
    """G235M_F170LP -> G235M, for column and filename use."""
    return grating.split("_")[0]


def catalogue_path(grating):
    return os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{grating}_match_0p3as.fits")


def parse_id_grating(lime_path):
    """Pull the integer ID and grating string from the filename."""
    name = os.path.basename(lime_path)
    m = re.match(r"^(\d+)_([A-Z0-9]+_[A-Z0-9]+)_spectra_lime\.fits$", name)
    if m is None:
        raise ValueError(f"Cannot parse ID and grating from filename: {name}")
    src_id = int(m.group(1))
    grating = m.group(2)
    if grating not in GRATINGS:
        raise ValueError(f"Grating {grating} not one of {GRATINGS}")
    return src_id, grating


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


def build_doublet_band():
    """One-row bands frame whose line region spans BOTH [OIII] lines."""
    w3 = OIII_4959_VAC - LINE_MARGIN
    w4 = OIII_5007_VAC + LINE_MARGIN
    w2 = w3 - CONT_GAP
    w1 = w2 - CONT_WIDTH
    w5 = w4 + CONT_GAP
    w6 = w5 + CONT_WIDTH
    return pd.DataFrame(
        {"wavelength": [OIII_5007_VAC],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[OIII_5007_LABEL],
    )


def single_5007_band():
    """A plain 5007-only band for the fallback fit."""
    w3 = OIII_5007_VAC - LINE_MARGIN
    w4 = OIII_5007_VAC + LINE_MARGIN
    w2 = w3 - CONT_GAP
    w1 = w2 - CONT_WIDTH
    w5 = w4 + CONT_GAP
    w6 = w5 + CONT_WIDTH
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


def line_in_range(centre_rest, wave, zf, margin=LINE_MARGIN):
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

    S/N is LiMe's own snr_line, read off the 5007 row. This is the amplitude
    based line detection statistic (Rola et al. definition, amplitude scaled
    by continuum noise and band width), not the band-integrated flux over its
    error. The integrated version washes out narrow lines in our wide doublet
    band, since it sums the noisy inter-line region. snr_line does not, so it
    is the correct detection metric here. Falls back to profile_flux / its
    error only if snr_line is absent.
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


def is_success(snr, centre_err_aa):
    """Both criteria must pass."""
    snr_ok = np.isfinite(snr) and snr >= SNR_MIN
    err_ok = np.isfinite(centre_err_aa) and centre_err_aa <= CENTRE_ERR_MAX_AA
    return bool(snr_ok and err_ok)


def fit_one(lime_path, z_dja):
    """Fit [OIII] in one spectrum.

    Returns a dict with keys z_oiii, z_oiii_err, snr, method, success,
    status. status is 'fitted' or 'out_of_range'. Raises on genuine errors.
    """
    src_id, grating = parse_id_grating(lime_path)
    gr = grating_short(grating)
    print(f"\n--- ID {src_id}  {grating} ---")
    print(f"spectrum  {lime_path}")
    print(f"DJA z     {z_dja:.5f}")

    spec, wave = load_lime_spectrum(lime_path, z_dja)
    zf = 1.0 + z_dja

    if not line_in_range(OIII_5007_VAC, wave, zf):
        print("OIII 5007 out of range")
        return {"status": "out_of_range"}

    # Figure dir for this source, grouping all gratings together.
    fig_dir = os.path.join(FIG_ROOT, str(src_id))
    os.makedirs(fig_dir, exist_ok=True)
    contsub_png = os.path.join(fig_dir, f"{src_id}_{grating}_contsub.png")
    fit_png = os.path.join(fig_dir, f"{src_id}_{grating}_OIII_fit.png")

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=contsub_png)

    doublet_band = build_doublet_band()
    if band_in_range(doublet_band.iloc[0], wave, zf):
        method = "joint_4959_5007"
        spec.fit.bands(OIII_BLEND_LABEL, bands=doublet_band, fit_cfg=build_joint_fit_cfg())
    else:
        method = "single_5007"
        print("4959 band out of range, single 5007 fit")
        spec.fit.bands(OIII_5007_LABEL, bands=single_5007_band())

    centre, centre_err, snr = get_5007_row(spec)
    z_oiii = centre_to_redshift(centre)
    z_oiii_err = centre_err_to_redshift_err(centre_err)
    success = is_success(snr, centre_err)

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


def run_all(spectra_root):
    """Walk all four grating dirs, fit every source, build both CSVs."""
    # Load the four catalogues once.
    catalogues = {}
    for grating in GRATINGS:
        cp = catalogue_path(grating)
        if os.path.exists(cp):
            catalogues[grating] = Table.read(cp)
        else:
            print(f"catalogue missing, skipping grating: {cp}")

    # Per-source accumulators.
    results = {}   # src_id -> {col: value} for RESULTS_CSV
    summary = {}   # src_id -> {col: value} for SUMMARY_CSV
    gratings_seen = {}  # src_id -> list of gratings with coverage

    for grating in GRATINGS:
        if grating not in catalogues:
            continue
        gr = grating_short(grating)
        cat = catalogues[grating]
        spec_dir = os.path.join(spectra_root, grating)
        files = sorted(glob.glob(os.path.join(spec_dir, "*_lime.fits")))
        print(f"\n{'#'*60}\n# {grating}   {len(files)} files\n{'#'*60}")

        for f in files:
            try:
                src_id, _ = parse_id_grating(f)
            except ValueError as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                continue

            z_dja = get_dja_redshift(cat, src_id)
            if z_dja is None:
                print(f"SKIPPED  {os.path.basename(f)}  no DJA z in catalogue")
                continue

            # Ensure rows exist.
            results.setdefault(src_id, {"ID": src_id})
            summary.setdefault(src_id, {"ID": src_id, "OIII_success": 0})
            gratings_seen.setdefault(src_id, [])

            # This source has coverage in this grating.
            gratings_seen[src_id].append(gr)
            results[src_id][f"z_{gr}"] = z_dja

            try:
                r = fit_one(f, z_dja)
            except Exception as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                summary[src_id][f"OIII_{gr}_success"] = False
                continue

            if r["status"] == "out_of_range":
                # Coverage in grating, but 5007 not on the detector.
                summary[src_id][f"OIII_{gr}_success"] = False
                continue

            # Record fit outputs.
            results[src_id][f"z_OIII_{gr}"] = r["z_oiii"]
            results[src_id][f"z_OIII_{gr}_err"] = r["z_oiii_err"]
            results[src_id][f"z_OIII_{gr}_snr"] = r["snr"]

            summary[src_id][f"OIII_{gr}_success"] = r["success"]
            if r["success"]:
                summary[src_id]["OIII_success"] += 1

    # Finalise the gratings string per source.
    for src_id, grs in gratings_seen.items():
        summary[src_id]["gratings"] = ", ".join(grs)

    # Build and write RESULTS_CSV.
    res_df = pd.DataFrame(list(results.values())).sort_values("ID")
    # Column order: ID, DJA zs, then per-grating OIII blocks.
    res_cols = ["ID"]
    for gr in [grating_short(g) for g in GRATINGS]:
        if f"z_{gr}" in res_df.columns:
            res_cols.append(f"z_{gr}")
    for gr in [grating_short(g) for g in GRATINGS]:
        for suff in ("", "_err", "_snr"):
            c = f"z_OIII_{gr}{suff}"
            if c in res_df.columns:
                res_cols.append(c)
    res_df = res_df.reindex(columns=res_cols)
    res_df.to_csv(RESULTS_CSV, index=False)

    # Build and write SUMMARY_CSV.
    sum_df = pd.DataFrame(list(summary.values())).sort_values("ID")
    sum_cols = ["ID", "gratings"]
    for gr in [grating_short(g) for g in GRATINGS]:
        c = f"OIII_{gr}_success"
        if c in sum_df.columns:
            sum_cols.append(c)
    sum_cols.append("OIII_success")
    sum_df = sum_df.reindex(columns=sum_cols)
    sum_df.to_csv(SUMMARY_CSV, index=False)

    # Final report.
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  sources processed   {len(results)}")
    print(f"  with >=1 success    {int((sum_df['OIII_success'] > 0).sum())}")
    print(f"  results csv         {RESULTS_CSV}")
    print(f"  summary csv         {SUMMARY_CSV}")
    print(f"  figures root        {FIG_ROOT}")
    print("=" * 60)


def main():
    spectra_root = sys.argv[1] if len(sys.argv) == 2 else JWST_SPECTRA_ROOT
    print(f"spectra root  {spectra_root}")
    os.makedirs(FIG_ROOT, exist_ok=True)
    run_all(spectra_root)


if __name__ == "__main__":
    main()