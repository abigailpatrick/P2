#!/usr/bin/env python
"""Download JELS/DJA NIRSpec spectra and plot each one as a 1D PNG.

For every FITS catalogue listed in CATALOGUE_FILES this script reads the
'root', 'file' and 'ID' columns, works out the grating from the catalogue
filename, downloads the matching .spec.fits from the public DJA S3 bucket,
and writes a simple 1D flux vs wavelength plot alongside it.

Downloaded spectra and plots go to:
    OUTPUT_BASE / <grating> / <ID>_<grating>_spectra.fits
    OUTPUT_BASE / <grating> / <ID>_<grating>_spectra.png
"""

import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, we only save PNGs
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.table import Table


# ----------------------------------------------------------------------
# User configuration. Edit these paths only.
# ----------------------------------------------------------------------
CATALOGUE_FILES = [
    "/ceph/cephfs/apatrick/P2/jwst_catalogs/JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    "/ceph/cephfs/apatrick/P2/jwst_catalogs/JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    "/ceph/cephfs/apatrick/P2/jwst_catalogs/JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    "/ceph/cephfs/apatrick/P2/jwst_catalogs/JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
]

OUTPUT_BASE = "/ceph/cephfs/apatrick/P2/jwst_spectra"

ROOT_SERVER_PATH = "https://s3.amazonaws.com/msaexp-nirspec/extractions"

# Column names in the input catalogues.
ROOT_COL = "root"
FILE_COL = "file"
ID_COL = "ID"

OVERWRITE = True  # always re-download, as requested
# ----------------------------------------------------------------------


def grating_from_catalogue(catalogue_path):
    """Pull the grating string out of a JELS_F356W_DJA_<grating>_match_0p3as.fits name."""
    name = os.path.basename(catalogue_path)
    stem = name.replace("JELS_F356W_DJA_", "").replace("_match_0p3as.fits", "")
    return stem


def curl_download(remote_url, local_output_path, overwrite=False):
    """Download a single file with curl. Returns the local path."""
    if os.path.exists(local_output_path) and not overwrite:
        print(f"    skip (exists)  {local_output_path}")
        return local_output_path

    curl_cmd = ["curl", "-fL", "-#", remote_url, "-o", local_output_path]
    try:
        subprocess.run(curl_cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to download with curl: {remote_url}") from err

    return local_output_path


def plot_spectrum_1d(fits_path, png_path, title=None):
    """Make a simple 1D flux vs wavelength PNG from a .spec.fits file.

    The 1D extraction lives in HDU 1 with columns 'wave', 'flux', 'err'
    (wave in micron, flux in uJy). Flux is converted to f_lambda in
    1e-20 erg/s/cm2/AA for the plot.
    """
    with fits.open(fits_path) as hdu:
        spec1d = hdu[1].data

    wave = spec1d["wave"] * u.micron
    fnu = spec1d["flux"] * u.uJy
    fnu_err = spec1d["err"] * u.uJy

    flambda = fnu.to(
        u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(wave)
    )
    flambda_err = fnu_err / fnu * flambda

    norm = 1e-20 * u.erg / u.s / u.cm ** 2 / u.AA

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.0))

    ax.axhline(0.0, ls="--", color="0.6", lw=1)
    ax.fill_between(
        wave.value,
        ((flambda + flambda_err) / norm).value,
        ((flambda - flambda_err) / norm).value,
        color="0.5",
        alpha=0.5,
        step="mid",
        linewidth=0.0,
    )
    ax.step(
        wave.value,
        (flambda / norm).value,
        where="mid",
        color="k",
        lw=1,
        solid_joinstyle="miter",
    )

    finite = np.isfinite((flambda / norm).value)
    if finite.sum() > 0:
        ymax = 1.4 * np.nanmax((flambda / norm).value[finite])
        ymin = np.nanpercentile((flambda / norm).value[finite], 1)
        ax.set_ylim([ymin, ymax])

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"$F_{\lambda}\ [10^{-20}\,{\rm erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}]$")
    if title is not None:
        ax.set_title(title, size=10)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_catalogue(catalogue_path):
    grating = grating_from_catalogue(catalogue_path)
    out_dir = os.path.join(OUTPUT_BASE, grating)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {grating} ===")
    print(f"catalogue      {catalogue_path}")
    print(f"output folder  {out_dir}")

    cat = Table.read(catalogue_path)
    print(f"rows           {len(cat)}")

    for row in cat:
        src_id = row[ID_COL]
        root = row[ROOT_COL]
        fname = row[FILE_COL]

        remote_url = f"{ROOT_SERVER_PATH}/{root}/{fname}"

        base = f"{src_id}_{grating}_spectra"
        fits_out = os.path.join(out_dir, base + ".fits")
        png_out = os.path.join(out_dir, base + ".png")

        print(f"\n  ID {src_id}")
        print(f"    url      {remote_url}")
        print(f"    fits ->  {fits_out}")

        try:
            curl_download(remote_url, fits_out, overwrite=OVERWRITE)
        except RuntimeError as err:
            print(f"    ERROR download failed: {err}")
            continue

        print(f"    png  ->  {png_out}")
        try:
            plot_spectrum_1d(fits_out, png_out, title=f"{src_id}  {grating}")
        except Exception as err:  # keep going if one plot breaks
            print(f"    ERROR plot failed: {err}")


def main():
    print("NIRSpec spectra download")
    print(f"output base    {OUTPUT_BASE}")
    print(f"server         {ROOT_SERVER_PATH}")
    print(f"overwrite      {OVERWRITE}")

    for catalogue_path in CATALOGUE_FILES:
        process_catalogue(catalogue_path)

    print("\nDone.")


if __name__ == "__main__":
    main()