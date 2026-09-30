#!/usr/bin/env python3
"""
Print the structure of a FITS catalogue so we can decide where M_UV comes from.

Prints the HDUs, the number of rows, every column with its unit, and then the
columns that look relevant to M_UV (fluxes, magnitudes, filters, UV slope and
SED outputs). Writes nothing to disk.

Usage
-----
python inspect_catalog_columns.py
python inspect_catalog_columns.py /path/to/other_catalogue.fits
"""

import re
import sys

from astropy.io import fits
from astropy.table import Table

DEFAULT = ("/ceph/cephfs/apatrick/P2/jwst_catalogs/"
           "JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits")

GROUPS = {
    "Existing UV / SED outputs": r"muv|m_uv|uv|beta|mass|sfr|ebv|e_bv|av\b|a_v|"
                                 r"dust|lum|l1500|age|chi|z_?phot|zphot|z_?best",
    "Fluxes": r"flux|^f_|_flux|fnu|flam",
    "Errors": r"err|unc|sig|^e_",
    "Magnitudes": r"mag",
    "Filter names (HST / NIRCam / ground)": r"f\d{3,4}[wmn]|acs|wfc3|nircam|"
                                            r"hsc|subaru|uvista|cfht|irac",
    "Apertures": r"ap|aper|kron|auto|0p3|0p6|0p9|2p0",
    "Redshift / IDs": r"^id$|_id$|^id_|z_|redshift|zspec|z_av|srcid",
}

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
print(f"[FILE] {path}\n")

with fits.open(path) as hdul:
    print("[HDUs]")
    for i, h in enumerate(hdul):
        shape = getattr(h, "shape", None) or getattr(h.data, "shape", None)
        print(f"  {i}: {h.name:<20s} {type(h).__name__:<16s} {shape}")
    table_hdus = [i for i, h in enumerate(hdul)
                  if isinstance(h, (fits.BinTableHDU, fits.TableHDU))]

for hdu in table_hdus:
    t = Table.read(path, hdu=hdu)
    print(f"\n[TABLE hdu={hdu}] {len(t)} rows, {len(t.colnames)} columns\n")

    print("[ALL COLUMNS]")
    for name in t.colnames:
        unit = t[name].unit if t[name].unit is not None else ""
        print(f"  {name:<40s} {str(t[name].dtype):<10s} {unit}")

    print("\n[COLUMNS BY TYPE]")
    for label, pattern in GROUPS.items():
        hits = [c for c in t.colnames if re.search(pattern, c, re.IGNORECASE)]
        print(f"\n  {label} ({len(hits)}):")
        for c in hits:
            print(f"    {c}")

    # Distinct filter names mentioned anywhere in the column names
    filt = sorted({m.upper() for c in t.colnames
                   for m in re.findall(r"f\d{3,4}[wmn]", c, re.IGNORECASE)})
    print(f"\n[FILTERS FOUND IN COLUMN NAMES] {', '.join(filt) if filt else 'none'}")