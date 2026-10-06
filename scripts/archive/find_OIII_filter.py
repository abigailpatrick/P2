#!/usr/bin/env python
"""Add an OIII_filter column to the merged JELS catalogue.

For each source, work out which NIRSpec band the [OIII] 5007 line falls in
based on z_av, and label it G235 (Band II) or G395 (Band III). Within a band
the M and H gratings share the same nominal range, so they collapse to one
label. Overlap gives 'G235,G395' and no coverage gives an empty string.

Reads and overwrites the CSV in place, leaving all existing columns intact.
"""

import numpy as np
import pandas as pd

csv_path = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv"

# [OIII] 5007 line, rest frame in Angstroms.
OIII_REST = 5006.84

# Nominal observed coverage per band, in Angstroms. Within a band the M and H
# gratings share the same nominal range (same long-pass filter), so we collapse
# to G235 (Band II, F170LP) or G395 (Band III, F290LP). Adjust if you want
# tighter numbers.
BAND_II_LO, BAND_II_HI = 16600.0, 31700.0    # G235, 1.66-3.17 um
BAND_III_LO, BAND_III_HI = 28700.0, 52700.0  # G395, 2.87-5.27 um

print(f"Reading catalogue from: {csv_path}")
out = pd.read_csv(csv_path)

if "z_av" not in out.columns:
    raise KeyError("z_av column not found in the CSV, cannot assign OIII_filter")

oiii_obs = OIII_REST * (1.0 + out["z_av"].values)

in_235 = (oiii_obs >= BAND_II_LO) & (oiii_obs <= BAND_II_HI)
in_395 = (oiii_obs >= BAND_III_LO) & (oiii_obs <= BAND_III_HI)

oiii_filter = np.full(len(out), "", dtype=object)
oiii_filter[in_235 & ~in_395] = "G235"
oiii_filter[in_395 & ~in_235] = "G395"
oiii_filter[in_235 & in_395] = "G235,G395"
# Neither band reaches the line -> stays empty.
out["OIII_filter"] = oiii_filter

n_235 = int((oiii_filter == "G235").sum())
n_395 = int((oiii_filter == "G395").sum())
n_both = int((oiii_filter == "G235,G395").sum())
n_none = int((oiii_filter == "").sum())

print("")
print("--- OIII filter assignment ---")
print(f"[OIII] 5007 in G235 only: {n_235}")
print(f"[OIII] 5007 in G395 only: {n_395}")
print(f"[OIII] 5007 in both (overlap): {n_both}")
print(f"[OIII] 5007 in neither band: {n_none}")

out.to_csv(csv_path, index=False)
print("")
print(f"Wrote updated catalogue with OIII_filter to: {csv_path}")