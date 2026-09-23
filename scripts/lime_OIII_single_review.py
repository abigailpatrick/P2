#!/usr/bin/env python
"""Fit the [OIII] 4959,5007 doublet for ONE source, for manual review.

Companion to lime_OIII_jointfit.py. Same fit recipe (tied doublet blend, with
fallback to a single 5007 fit), same fitting outputs, but for a single JELS ID
so the sources that failed the batch run can be re-fitted with per-source
overrides and inspected one at a time.

Given an ID, this walks all four grating folders under JWST_SPECTRA_ROOT,
finds every <ID>_<grating>_spectra_lime.fits that exists, and fits each one.
It then rebuilds the best-grating selection for that source and patches just
that source's row in grating_sources_with_zsys.csv (every other row is left
untouched) and writes figures to the same OIII_fits/<ID>/ dir.

For each source the grating with the highest OIII SNR (across the gratings
fitted in this run) is selected, and that grating provides:
    grating        -> the selected grating short name (e.g. G395M)
    z_dja          -> z_<grating>        (DJA redshift for that grating)
    z_sys          -> z_OIII_<grating>   (fitted systemic redshift)
    z_sys_err      -> z_OIII_<grating>_err
    z_sys_snr      -> z_OIII_<grating>_snr
    z_sys_quality  -> a if z_sys_snr > 13.0, c if <= 13.0, d if no z_sys

Sources with no z_sys take their first fitted grating (in the order below) and
are flagged d.

Per-source overrides (all optional)
-----------------------------------
--z FLOAT           Corrected absolute redshift used to build the band. The
                    DJA z is still used for z_dja; the fitted z_OIII always
                    comes from the real line centre, so z_sys stays honest
                    regardless of --z.
--line-margin FLOAT Rest-frame Angstrom the line region extends beyond each
                    outer line. Default matches the batch script (15).
--cont-gap FLOAT    Gap between line region and each continuum flank (5).
--cont-width FLOAT  Width of each continuum flank (20).
--snr-min FLOAT     Minimum snr_line for a detection (13).
--centre-err-max FLOAT  Max fitted centre error in Angstrom (2.5).
--force-single      Force the single-5007 fit even where 4959 is in range.
--force-double      Force the joint doublet fit even when a continuum flank
                    runs off the good data. Pair with --cont-side to drop the
                    off-edge flank so the continuum is anchored on real data.
                    Overrides the automatic joint-vs-single range check. Has no
                    effect together with --force-single (single wins).
--cont-side SIDE    Keep only one continuum flank when the other falls in the
                    detector gap. 'blue' keeps the blue flank (left of the
                    lines) and drops the red, 'red' does the mirror, 'both'
                    is the default. The line region is untouched, so both
                    lines are still fitted; only the continuum changes.
--mask LO,HI        Observed-Angstrom range to drop as bad pixels before the
                    fit, e.g. --mask 29285,29320 to remove an isolated spike.
                    Repeatable for more than one spike.
--gratings LIST     Comma-separated subset to fit, e.g. G235M,G395H. Default
                    is every grating the source has coverage in. Note the best
                    grating is chosen only among the gratings fitted this run.
--dry-run           Fit and plot, print everything, but do NOT write the CSV.

Examples
--------
python lime_OIII_single_review.py 59737
python lime_OIII_single_review.py 59737 --z 5.9412
python lime_OIII_single_review.py 59737 --z 5.9412 --line-margin 25
python lime_OIII_single_review.py 59737 --gratings G235M --force-single --dry-run
python lime_OIII_single_review.py 49939 --force-double --cont-side blue --dry-run

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

# Output CSV, patched in place (one row updated per run).
ZSYS_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv"

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

# Quality flag threshold on z_sys_snr.
QUALITY_SNR_THRESHOLD = 13.0

# Column order of the z_sys CSV.
ZSYS_COLUMNS = ["ID", "grating", "z_dja", "z_sys",
                "z_sys_err", "z_sys_snr", "z_sys_quality"]
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


def load_lime_spectrum(lime_path, redshift, mask_ranges=None):
    """Load the _lime.fits SPECTRUM table into a lime.Spectrum.

    mask_ranges is an optional list of (lo, hi) observed-Angstrom pairs. Pixels
    whose wavelength falls in any of these ranges are dropped before the
    Spectrum is built, so LiMe never sees them. Use it to remove isolated
    bad-pixel or cosmic-ray spikes near the lines, the same way detector-gap
    NaNs are already excluded.
    """
    with fits.open(lime_path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)

    # Drop any pixels inside a requested mask range (observed Angstrom).
    if mask_ranges:
        in_mask = np.zeros_like(wave, dtype=bool)
        for lo, hi in mask_ranges:
            in_mask |= (wave >= lo) & (wave <= hi)
        n_masked = int((good & in_mask).sum())
        good = good & ~in_mask
        print(f"masked    {n_masked} pixels in "
              f"{', '.join(f'{lo:.1f}-{hi:.1f}' for lo, hi in mask_ranges)} AA")

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


def apply_cont_side(w1, w2, w3, w4, w5, w6, cont_side):
    """Collapse one continuum flank when a side is affected by the gap.

    'both' leaves the flanks as built. 'blue' keeps only the blue flank
    (w1-w2) and collapses the red flank onto the red edge of the line region,
    so w5=w6=w4 and LiMe fits the continuum from blue data only. 'red' does
    the mirror. The line region (w3-w4) is never touched, so both lines are
    still fitted exactly as before; only the continuum anchoring changes.
    """
    if cont_side == "blue":
        w5 = w4
        w6 = w4
    elif cont_side == "red":
        w1 = w3
        w2 = w3
    return w1, w2, w3, w4, w5, w6


def build_doublet_band(line_margin, cont_gap, cont_width, cont_side="both"):
    """One-row bands frame whose line region spans BOTH [OIII] lines."""
    w3 = OIII_4959_VAC - line_margin
    w4 = OIII_5007_VAC + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    w1, w2, w3, w4, w5, w6 = apply_cont_side(w1, w2, w3, w4, w5, w6, cont_side)
    return pd.DataFrame(
        {"wavelength": [OIII_5007_VAC],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[OIII_5007_LABEL],
    )


def single_5007_band(line_margin, cont_gap, cont_width, cont_side="both"):
    """A plain 5007-only band for the fallback fit."""
    w3 = OIII_5007_VAC - line_margin
    w4 = OIII_5007_VAC + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    w1, w2, w3, w4, w5, w6 = apply_cont_side(w1, w2, w3, w4, w5, w6, cont_side)
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

    spec, wave = load_lime_spectrum(lime_path, z_fit, cfg["mask_ranges"])
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
                                      cfg["cont_width"], cont_side=cfg["cont_side"])

    # Decide joint vs single.
    #   --force-single always wins and gives a single 5007 fit.
    #   --force-double forces the doublet regardless of the flank range check
    #     (use with --cont-side to keep the off-edge flank off the data).
    #   Otherwise the original behaviour: the doublet is used only when the
    #     FULL-width band fits inside the good data, so a flank running off the
    #     edge sends it to single 5007.
    if cfg["force_single"]:
        do_joint = False
    elif cfg["force_double"]:
        do_joint = True
    else:
        doublet_full = build_doublet_band(cfg["line_margin"], cfg["cont_gap"],
                                          cfg["cont_width"], cont_side="both")
        do_joint = band_in_range(doublet_full.iloc[0], wave, zf)

    if cfg["cont_side"] != "both":
        print(f"cont side  {cfg['cont_side']} flank only")

    if do_joint:
        method = "joint_4959_5007"
        if cfg["force_double"]:
            print("joint doublet fit (forced)")
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
                                              cfg["cont_width"],
                                              cont_side=cfg["cont_side"]))

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


def quality_flag(z_sys, z_sys_snr):
    """Initial quality flag from the systemic SNR.

    a if z_sys_snr > threshold, c if <= threshold, d if no z_sys measurement.
    Manual b flags are set later by hand and are not produced here.
    """
    if z_sys is None or not np.isfinite(z_sys):
        return "d"
    if np.isfinite(z_sys_snr) and z_sys_snr > QUALITY_SNR_THRESHOLD:
        return "a"
    return "c"


def select_best(fit_results, z_dja_by_grating, gratings_seen):
    """Collapse per-grating fits into one best-grating row for the z_sys CSV.

    fit_results maps grating short name -> the dict returned by fit_one for
    gratings that produced a z_OIII. z_dja_by_grating maps short name -> DJA z.
    gratings_seen is the ordered list of gratings that had a spectrum this run,
    used to pick a fallback grating when no fit yielded a z_sys.

    Returns the row dict with columns matching ZSYS_COLUMNS (minus ID).
    """
    # Gratings that actually produced a systemic redshift with a finite SNR.
    snr_by_grating = {
        gr: r["snr"] for gr, r in fit_results.items()
        if np.isfinite(r.get("z_oiii", np.nan)) and np.isfinite(r.get("snr", np.nan))
    }

    if snr_by_grating:
        best = max(snr_by_grating, key=snr_by_grating.get)
        r = fit_results[best]
        z_sys = r["z_oiii"]
        z_sys_err = r["z_oiii_err"]
        z_sys_snr = r["snr"]
        z_dja = z_dja_by_grating.get(best, np.nan)
        flag = quality_flag(z_sys, z_sys_snr)
        return {
            "grating": best,
            "z_dja": z_dja,
            "z_sys": z_sys,
            "z_sys_err": z_sys_err,
            "z_sys_snr": z_sys_snr,
            "z_sys_quality": flag,
        }

    # No z_sys at all: fall back to the first grating fitted this run.
    fallback = gratings_seen[0] if gratings_seen else np.nan
    z_dja = z_dja_by_grating.get(fallback, np.nan) if isinstance(fallback, str) else np.nan
    return {
        "grating": fallback,
        "z_dja": z_dja,
        "z_sys": np.nan,
        "z_sys_err": np.nan,
        "z_sys_snr": np.nan,
        "z_sys_quality": "d",
    }


def patch_zsys_csv(src_id, row):
    """Read ZSYS_CSV, update this source's single row, write back.

    row is a dict of column->value (without ID). If the CSV or the row does
    not exist yet, it is created. Every other row is left untouched.
    """
    if os.path.exists(ZSYS_CSV):
        df = pd.read_csv(ZSYS_CSV)
    else:
        df = pd.DataFrame(columns=ZSYS_COLUMNS)

    # Make sure every expected column exists.
    for col in ZSYS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    full_row = {"ID": src_id, **row}

    mask = df["ID"] == src_id
    if mask.any():
        for col, val in full_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)

    df = df.sort_values("ID")
    df = df[ZSYS_COLUMNS]
    df.to_csv(ZSYS_CSV, index=False)


def run_single(src_id, cfg):
    """Fit every available grating for one source and patch the z_sys CSV."""
    spectra_root = cfg["spectra_root"]

    # Which gratings to attempt.
    want = cfg["gratings"] if cfg["gratings"] else GRATINGS

    # Per-grating accumulators for this one source.
    fit_results = {}          # grating short -> fit_one dict
    z_dja_by_grating = {}     # grating short -> DJA z
    gratings_seen = []        # ordered list of gratings with a spectrum

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
        z_dja_by_grating[gr] = z_dja

        try:
            r = fit_one(lime_path, src_id, grating, z_fit, z_dja, cfg)
        except Exception as err:
            print(f"FIT FAILED  {os.path.basename(lime_path)}  reason: {err}")
            continue

        if r["status"] == "out_of_range":
            continue

        fit_results[gr] = r

    if not any_found:
        print(f"\nNo spectra found for ID {src_id} in any grating folder under")
        print(f"  {spectra_root}")
        print("Nothing written.")
        return

    # Collapse into the single best-grating row for the z_sys CSV.
    row = select_best(fit_results, z_dja_by_grating, gratings_seen)

    # Report.
    print("\n" + "=" * 60)
    print(f"DONE  ID {src_id}")
    print(f"  gratings fitted     {', '.join(gratings_seen)}")
    print(f"  best grating        {row['grating']}")
    print(f"  z_dja               {row['z_dja']}")
    print(f"  z_sys               {row['z_sys']}")
    print(f"  z_sys_err           {row['z_sys_err']}")
    print(f"  z_sys_snr           {row['z_sys_snr']}")
    print(f"  z_sys_quality       {row['z_sys_quality']}")

    if cfg["dry_run"]:
        print("  dry run, CSV NOT written")
        print(f"  would patch         {ZSYS_CSV}")
        print(f"  figures root        {os.path.join(FIG_ROOT, str(src_id))}")
        print("=" * 60)
        return

    patch_zsys_csv(src_id, row)

    print(f"  patched csv         {ZSYS_CSV}")
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
    p.add_argument("--force-double", action="store_true",
                   help="force the joint doublet fit even when a continuum "
                        "flank runs off the good data (pair with --cont-side)")
    p.add_argument("--cont-side", type=str, default="both",
                   choices=["both", "blue", "red"],
                   help="keep only this continuum flank when the other is in "
                        "the detector gap (default both)")
    p.add_argument("--mask", action="append", default=None, dest="mask",
                   metavar="LO,HI",
                   help="observed-Angstrom range to drop as bad pixels, e.g. "
                        "--mask 29285,29320. Repeatable for multiple spikes.")
    p.add_argument("--gratings", type=str, default=None,
                   help="comma-separated subset, e.g. G235M,G395H")
    p.add_argument("--spectra-root", type=str, default=JWST_SPECTRA_ROOT)
    p.add_argument("--dry-run", action="store_true",
                   help="fit and plot but do not write the CSV")
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


def resolve_masks(mask_args):
    """Parse repeated --mask LO,HI strings into a list of (lo, hi) float pairs."""
    if not mask_args:
        return None
    out = []
    for m in mask_args:
        parts = m.split(",")
        if len(parts) != 2:
            raise ValueError(f"--mask expects LO,HI, got '{m}'")
        lo, hi = float(parts[0]), float(parts[1])
        if lo >= hi:
            raise ValueError(f"--mask LO must be < HI, got '{m}'")
        out.append((lo, hi))
    return out


def main():
    args = parse_args()

    if args.force_single and args.force_double:
        print("NOTE: both --force-single and --force-double given; single wins.")

    cfg = {
        "z_override": args.z_override,
        "line_margin": args.line_margin,
        "cont_gap": args.cont_gap,
        "cont_width": args.cont_width,
        "snr_min": args.snr_min,
        "centre_err_max": args.centre_err_max,
        "force_single": args.force_single,
        "force_double": args.force_double,
        "cont_side": args.cont_side,
        "mask_ranges": resolve_masks(args.mask),
        "gratings": resolve_gratings(args.gratings),
        "spectra_root": args.spectra_root,
        "dry_run": args.dry_run,
    }

    print(f"single-source [OIII] review")
    print(f"  ID            {args.src_id}")
    print(f"  spectra root  {cfg['spectra_root']}")
    print(f"  zsys csv      {ZSYS_CSV}")
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
    if cfg["force_double"]:
        print("  force joint doublet fit")
    if cfg["cont_side"] != "both":
        print(f"  cont side     {cfg['cont_side']} flank only")
    if cfg["mask_ranges"]:
        print("  mask ranges   "
              + ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in cfg["mask_ranges"])
              + " AA")
    if cfg["dry_run"]:
        print("  DRY RUN, CSV will not be written")

    os.makedirs(FIG_ROOT, exist_ok=True)
    run_single(args.src_id, cfg)


if __name__ == "__main__":
    main()