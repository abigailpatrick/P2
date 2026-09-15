#!/usr/bin/env python
"""Fit [OIII] 5007 in DJA/LiMe spectra and record the systemic redshift.

Single or batch mode
--------------------
- Pass a _lime.fits file  -> fits that one spectrum.
- Pass a directory        -> fits every *_lime.fits inside it, skipping any
                             that fail, and prints a summary at the end.



E.g. 
python lime_OIII_singlefit.py /ceph/cephfs/apatrick/P2/jwst_spectra/G235H_F170LP
python lime_OIII_singlefit.py /ceph/cephfs/apatrick/P2/jwst_spectra/G235M_F170LP
python lime_OIII_singlefit.py /ceph/cephfs/apatrick/P2/jwst_spectra/G395H_F290LP
python lime_OIII_singlefit.py /ceph/cephfs/apatrick/P2/jwst_spectra/G395M_F290LP

Per source
----------
1. Parse ID and grating from the filename.
2. Read the DJA redshift from the matching grating catalogue (for placement
   and the in-range check).
3. Load into lime.Spectrum (Angstrom, FLAM).
4. Fit a single Gaussian plus linear continuum to [OIII] 5007 via LiMe's
   bands frame, in vacuum wavelengths.
5. Convert the fitted centre and its error to z_OIII and z_OIII_err, and
   write them to columns z_OIII_<grating> and z_OIII_<grating>_err of the
   results CSV, matched on ID.
6. Save a continuum plot and a line-fit plot next to the spectrum.

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
RESULTS_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv"

# Vacuum rest wavelength of [OIII] 5007 in Angstrom, for the z_OIII conversion.
OIII_5007_VAC = 5006.843

# LiMe line label.
OIII_LABEL = "O3_5007A"

# DJA catalogue column names.
DJA_Z_COL = "z"
DJA_ID_COL = "ID"

# The four gratings and their catalogue files.
GRATINGS = ["G235H_F170LP", "G235M_F170LP", "G395H_F290LP", "G395M_F290LP"]

# Per-source diagnostic prints. Quieter is better in batch, so default off.
VERBOSE_DIAGNOSTICS = True
# ----------------------------------------------------------------------


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


def get_dja_redshift(src_id, grating):
    """Read the DJA redshift for this ID from the matching grating catalogue."""
    cat_path = os.path.join(
        CATALOG_DIR, f"JELS_F356W_DJA_{grating}_match_0p3as.fits"
    )
    cat = Table.read(cat_path)
    mask = cat[DJA_ID_COL] == src_id
    if mask.sum() == 0:
        raise ValueError(f"ID {src_id} not found in {cat_path}")
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


def centre_to_redshift(centre_obs_aa):
    """Convert an observed-frame Gaussian centre to the [OIII] redshift."""
    return centre_obs_aa / OIII_5007_VAC - 1.0


def centre_err_to_redshift_err(centre_err_aa):
    """Propagate the Gaussian centre error to the redshift error."""
    if centre_err_aa is None or not np.isfinite(centre_err_aa):
        return np.nan
    return centre_err_aa / OIII_5007_VAC


def update_results_csv(src_id, grating, z_oiii, z_oiii_err):
    """Add or update z_OIII_<grating> and its error for this ID, in place."""
    gr = grating.split('_')[0]          # e.g. G235H
    col = f"z_OIII_{gr}"
    col_err = f"z_OIII_{gr}_err"

    df = pd.read_csv(RESULTS_CSV)

    id_col = df.columns[0]  # first column holds the ID
    for c in (col, col_err):
        if c not in df.columns:
            df[c] = np.nan

    row_mask = df[id_col] == src_id
    if row_mask.sum() == 0:
        raise ValueError(f"ID {src_id} not found in {RESULTS_CSV} column {id_col}")

    df.loc[row_mask, col] = z_oiii
    df.loc[row_mask, col_err] = z_oiii_err
    df.to_csv(RESULTS_CSV, index=False)
    return col, col_err


def process_one(lime_path):
    """Fit one spectrum. Returns a status string: 'ok', 'out_of_range'.

    Raises on genuine errors so the batch layer can catch and skip.
    """
    src_id, grating = parse_id_grating(lime_path)
    print(f"\n--- ID {src_id}  {grating} ---")
    print(f"spectrum  {lime_path}")

    z_dja = get_dja_redshift(src_id, grating)
    print(f"DJA z     {z_dja:.5f}")

    spec, wave = load_lime_spectrum(lime_path, z_dja)

    lines_df = spec.retrieve.lines_frame(vacuum_waves=True)
    if OIII_LABEL not in lines_df.index:
        raise RuntimeError(f"{OIII_LABEL} not in LiMe bands frame")

    oiii_bands = lines_df.loc[[OIII_LABEL]]

    # In-range check. LiMe w columns are rest-frame, shift to observed.
    zf = 1.0 + z_dja
    w1_obs = float(oiii_bands["w1"].iloc[0]) * zf
    w6_obs = float(oiii_bands["w6"].iloc[0]) * zf
    if (w1_obs < np.nanmin(wave)) or (w6_obs > np.nanmax(wave)):
        print("OIII out of range")
        return "out_of_range"

    spec_dir = os.path.dirname(lime_path)
    contsub_png = os.path.join(spec_dir, f"{src_id}_{grating}_contsub.png")
    fit_png = os.path.join(spec_dir, f"{src_id}_{grating}_OIII_fit.png")

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=contsub_png)

    spec.fit.bands(OIII_LABEL, bands=oiii_bands)

    row = spec.frame.loc[OIII_LABEL]
    centre_obs = float(row["center"])
    centre_err = float(row["center_err"]) if "center_err" in spec.frame.columns else np.nan

    z_oiii = centre_to_redshift(centre_obs)
    z_oiii_err = centre_err_to_redshift_err(centre_err)
    print(f"z_OIII    {z_oiii:.6f} +/- {z_oiii_err:.6f}")

    spec.plot.bands(OIII_LABEL, fname=fit_png)

    col, col_err = update_results_csv(src_id, grating, z_oiii, z_oiii_err)
    print(f"wrote to  {col}, {col_err}")
    return "ok"


def run_batch(directory):
    """Process every *_lime.fits in a directory, skipping failures."""
    pattern = os.path.join(directory, "*_lime.fits")
    files = sorted(glob.glob(pattern))

    print(f"batch directory  {directory}")
    print(f"found {len(files)} _lime.fits files")

    ok, out_of_range, skipped = [], [], []

    for f in files:
        try:
            status = process_one(f)
            if status == "ok":
                ok.append(f)
            elif status == "out_of_range":
                out_of_range.append(f)
        except Exception as err:
            print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
            skipped.append((f, str(err)))

    # Summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  total files     {len(files)}")
    print(f"  fitted ok       {len(ok)}")
    print(f"  out of range    {len(out_of_range)}")
    print(f"  skipped (error) {len(skipped)}")

    if out_of_range:
        print("\n  out of range:")
        for f in out_of_range:
            print(f"    {os.path.basename(f)}")

    if skipped:
        print("\n  skipped:")
        for f, reason in skipped:
            print(f"    {os.path.basename(f)}  ->  {reason}")

    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("usage: python lime_OIII_singlefit.py <_lime.fits file OR directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        run_batch(path)
    else:
        process_one(path)
        print(f"\nupdated  {RESULTS_CSV}")


if __name__ == "__main__":
    main()