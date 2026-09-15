#!/usr/bin/env python
"""Convert DJA/msaexp NIRSpec .spec.fits files into LiMe-ready spectrum files.

Reads the 1D extraction (HDU 1, columns 'wave' [micron], 'flux' and 'err'
[uJy]), converts to Angstrom and FLAM (erg/s/cm2/Angstrom), and writes a
clean two-part FITS to <input path>_lime.fits.

Single or batch mode
--------------------
- Pass a *_spectra.fits file -> converts that one.
- Pass a directory           -> converts every *_spectra.fits inside it
                                (ignoring already-made *_spectra_lime.fits),
                                skipping any that fail, with a summary.

No fitting and no redshift here. The output holds only the spectrum, ready
to be read into lime.Spectrum by a separate fitting script.

E.g. 
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/G235H_F170LP
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/G235M_F170LP
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/G395H_F290LP
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/G395M_F290LP

Output HDUs
-----------
0  primary, empty, with provenance keywords in the header
1  BinTable 'SPECTRUM' with columns WAVE, FLUX, ERR
   (WAVE in Angstrom, FLUX and ERR in FLAM)
"""

import os
import sys
import glob

import numpy as np
import astropy.units as u
from astropy.io import fits


# Units the output file will hold.
UNITS_WAVE = "Angstrom"
UNITS_FLUX = "FLAM"  # erg / s / cm2 / Angstrom


def output_path_for(input_path):
    """Return <input path without .fits>_lime.fits."""
    root, ext = os.path.splitext(input_path)
    return f"{root}_lime{ext}"


def convert_dja_to_lime(input_path):
    """Read a DJA .spec.fits and return WAVE [AA], FLUX [FLAM], ERR [FLAM]."""
    with fits.open(input_path) as hdul:
        spec1d = hdul[1].data

    wave = np.asarray(spec1d["wave"], dtype=float) * u.micron
    fnu = np.asarray(spec1d["flux"], dtype=float) * u.uJy
    fnu_err = np.asarray(spec1d["err"], dtype=float) * u.uJy

    # Convert flux density from f_nu to f_lambda at each wavelength.
    flam = fnu.to(
        u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(wave)
    )
    # Convert the error array directly, avoiding divide-by-zero at empty pixels.
    flam_err = fnu_err.to(
        u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(wave)
    )

    wave_aa = wave.to(u.AA).value
    return wave_aa, flam.value, flam_err.value


def write_lime_fits(output_path, wave_aa, flam, flam_err, source_path):
    """Write a clean spectrum FITS with a SPECTRUM BinTable."""
    col_wave = fits.Column(name="WAVE", array=wave_aa, format="D", unit=UNITS_WAVE)
    col_flux = fits.Column(name="FLUX", array=flam, format="D", unit=UNITS_FLUX)
    col_err = fits.Column(name="ERR", array=flam_err, format="D", unit=UNITS_FLUX)

    table_hdu = fits.BinTableHDU.from_columns(
        [col_wave, col_flux, col_err], name="SPECTRUM"
    )

    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = (os.path.basename(source_path), "Source DJA spectrum")
    primary.header["UNITWAVE"] = (UNITS_WAVE, "Wavelength units")
    primary.header["UNITFLUX"] = (UNITS_FLUX, "Flux density units, erg/s/cm2/AA")
    primary.header["CONVERT"] = ("fnu uJy -> flam", "Conversion applied from DJA product")

    hdul = fits.HDUList([primary, table_hdu])
    hdul.writeto(output_path, overwrite=True)


def process_one(input_path):
    """Convert one spectrum. Prints paths and a short summary line."""
    output_path = output_path_for(input_path)
    print(f"\ninput   {input_path}")
    print(f"output  {output_path}")

    wave_aa, flam, flam_err = convert_dja_to_lime(input_path)

    n_total = wave_aa.size
    n_finite = int(np.sum(np.isfinite(flam)))
    print(f"pixels  {n_total} total, {n_finite} finite flux")
    print(f"wave    {np.nanmin(wave_aa):.1f} to {np.nanmax(wave_aa):.1f} {UNITS_WAVE}")

    write_lime_fits(output_path, wave_aa, flam, flam_err, input_path)
    print(f"written {output_path}")


def run_batch(directory):
    """Convert every *_spectra.fits in a directory, skipping the _lime outputs."""
    pattern = os.path.join(directory, "*_spectra.fits")
    files = sorted(
        f for f in glob.glob(pattern) if not f.endswith("_spectra_lime.fits")
    )

    print(f"batch directory  {directory}")
    print(f"found {len(files)} _spectra.fits files to convert")

    ok, skipped = [], []

    for f in files:
        try:
            process_one(f)
            ok.append(f)
        except Exception as err:
            print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
            skipped.append((f, str(err)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  total files      {len(files)}")
    print(f"  converted ok     {len(ok)}")
    print(f"  skipped (error)  {len(skipped)}")
    if skipped:
        print("\n  skipped:")
        for f, reason in skipped:
            print(f"    {os.path.basename(f)}  ->  {reason}")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("usage: python make_lime_spectra_format.py <_spectra.fits file OR directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        run_batch(path)
    else:
        process_one(path)


if __name__ == "__main__":
    main()