#!/usr/bin/env python
"""
find_duplicate_sources.py

Finds the flagged duplicate sources (by JELS ID) in each of the four NIRSpec
grating match catalogues and writes every matching full row to a single CSV.

One output row per time an ID is found in a catalogue. Two helper columns are
added: source_grating (which catalogue the row came from) and duplicate_pair
(the duplicate group the ID belongs to).
"""

import os
import numpy as np
import pandas as pd
from astropy.table import Table

# ---------------------------------------------------------------------------
# Input catalogues
# ---------------------------------------------------------------------------
CAT_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"

CATALOGUES = {
    "G235H": os.path.join(CAT_DIR, "JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits"),
    "G235M": os.path.join(CAT_DIR, "JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits"),
    "G395H": os.path.join(CAT_DIR, "JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits"),
    "G395M": os.path.join(CAT_DIR, "JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits"),
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/duplicate_sources.csv"

# ---------------------------------------------------------------------------
# Duplicate groups. Keyed by a readable label, value is the list of IDs.
# ---------------------------------------------------------------------------
DUPLICATE_GROUPS = {
    "dup1_ra150.07949246_dec2.36779416": [49482, 49694],
    "dup2_ra150.10372593_dec2.34859301": [43604, 43713],
    "dup3_ra150.11015839_dec2.34666959": [42990, 43129],
    "dup4_ra150.130085_dec2.36714937":   [49289, 49296],
}

ID_COLUMN = "ID"  # matching key


def main():
    # Flatten the wanted IDs and remember which group each belongs to
    id_to_group = {}
    for group, ids in DUPLICATE_GROUPS.items():
        for i in ids:
            id_to_group[int(i)] = group
    wanted_ids = set(id_to_group.keys())

    print("Looking for IDs: " + ", ".join(str(i) for i in sorted(wanted_ids)))
    print("")

    collected = []

    for grating, path in CATALOGUES.items():
        print("Reading " + grating + ": " + path)
        if not os.path.exists(path):
            print("  WARNING: file not found, skipping.")
            continue

        tab = Table.read(path)
        df = tab.to_pandas()

        if ID_COLUMN not in df.columns:
            print("  WARNING: no '" + ID_COLUMN + "' column in this catalogue, skipping.")
            print("  Columns present: " + ", ".join(map(str, df.columns)))
            continue

        # Cast the ID column to int for a clean integer match
        ids_int = df[ID_COLUMN].astype(float).round().astype("Int64")

        mask = ids_int.isin(wanted_ids)
        n_found = int(mask.sum())
        print("  Matched " + str(n_found) + " row(s) in " + grating)

        if n_found == 0:
            continue

        sub = df[mask].copy()
        sub.insert(0, "source_grating", grating)
        sub.insert(1, "duplicate_pair",
                   ids_int[mask].map(id_to_group).values)
        collected.append(sub)

    print("")

    if not collected:
        print("No matching rows found in any catalogue. Nothing written.")
        return

    out = pd.concat(collected, ignore_index=True)

    # Report which IDs were and were not found
    found_ids = set(out[ID_COLUMN].astype(float).round().astype(int).tolist())
    missing = sorted(wanted_ids - found_ids)
    print("Total rows collected: " + str(len(out)))
    print("Unique IDs found: " + ", ".join(str(i) for i in sorted(found_ids)))
    if missing:
        print("IDs NOT found in any catalogue: " +
              ", ".join(str(i) for i in missing))
    else:
        print("All requested IDs were found.")
    print("")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("Output CSV written to:")
    print("  " + os.path.abspath(OUT_CSV))


if __name__ == "__main__":
    main()