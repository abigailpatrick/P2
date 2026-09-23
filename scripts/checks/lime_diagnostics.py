#!/usr/bin/env python
"""Check real [OIII] coverage for one source across all gratings.

For each grating the source has a _lime.fits in, this reports the good-data
wavelength extent, whether the 4959 and 5007 line windows fall inside good
pixels, and whether any detector gap sits on the lines. Tells apart three
cases: line on good data, line in a gap, line past the end of good data.

Usage
-----
python check_coverage.py <ID>
python check_coverage.py <ID> --z 4.86   # override the DJA z used for the test
"""

import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table

CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"
GRATINGS = ["G235M_F170LP", "G235H_F170LP", "G395M_F290LP", "G395H_F290LP"]

OIII_5007_VAC = 5006.843
OIII_4959_VAC = 4958.911
LINE_MARGIN = 15.0

DJA_Z_COL = "z"
DJA_ID_COL = "ID"


def grating_short(g):
    return g.split("_")[0]


def catalogue_path(g):
    return os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{g}_match_0p3as.fits")


def spectrum_path(src_id, g):
    return os.path.join(JWST_SPECTRA_ROOT, g, f"{src_id}_{g}_spectra_lime.fits")


def dja_z(g, src_id):
    cp = catalogue_path(g)
    if not os.path.exists(cp):
        return None
    cat = Table.read(cp)
    m = cat[DJA_ID_COL] == src_id
    if m.sum() == 0:
        return None
    return float(cat[DJA_Z_COL][m][0])


def line_status(centre_rest, wave_good, wave_all, zf):
    """Classify one line: on_good_data / in_gap / past_end / before_start."""
    lo = (centre_rest - LINE_MARGIN) * zf
    hi = (centre_rest + LINE_MARGIN) * zf
    gmin, gmax = wave_good.min(), wave_good.max()

    if lo < gmin and hi < gmin:
        return "before_start_of_good_data"
    if lo > gmax and hi > gmax:
        return "past_end_of_good_data"

    # Window overlaps good extent: are there actually good pixels under it?
    n_in = int(((wave_good >= lo) & (wave_good <= hi)).sum())
    if n_in == 0:
        return "in_gap_no_good_pixels"
    # partial coverage?
    n_expected = (hi - lo) / np.median(np.diff(np.sort(wave_good)))
    if n_in < 0.5 * n_expected:
        return f"partial_{n_in}_pixels"
    return f"on_good_data_{n_in}_pixels"


def check_grating(src_id, g, z_override):
    path = spectrum_path(src_id, g)
    if not os.path.exists(path):
        return
    print(f"\n--- {g} ---")
    print(f"file  {path}")

    z = z_override if z_override is not None else dja_z(g, src_id)
    if z is None:
        print("no DJA z in catalogue, skipping")
        return
    zf = 1.0 + z
    print(f"z used  {z:.5f}")

    with fits.open(path) as h:
        tab = h["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wg = np.sort(wave[good])
    print(f"good data extent  {wg.min():.1f} to {wg.max():.1f} AA "
          f"({wg.size}/{wave.size} pixels good)")

    print(f"5007 expected at  {OIII_5007_VAC*zf:.1f} AA  ->  "
          f"{line_status(OIII_5007_VAC, wg, wave, zf)}")
    print(f"4959 expected at  {OIII_4959_VAC*zf:.1f} AA  ->  "
          f"{line_status(OIII_4959_VAC, wg, wave, zf)}")

    # detector gaps
    dw = np.diff(wg)
    med = np.median(dw)
    big = np.where(dw > 5 * med)[0]
    if big.size:
        for i in big:
            print(f"gap  {wg[i]:.1f} to {wg[i+1]:.1f}  ({wg[i+1]-wg[i]:.1f} AA)")
    else:
        print("no large internal gaps in good data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_id", type=int)
    p.add_argument("--z", type=float, default=None)
    args = p.parse_args()

    print(f"coverage check  ID {args.src_id}")
    any_found = False
    for g in GRATINGS:
        if os.path.exists(spectrum_path(args.src_id, g)):
            any_found = True
        check_grating(args.src_id, g, args.z)
    if not any_found:
        print("no spectra found in any grating folder")


if __name__ == "__main__":
    main()