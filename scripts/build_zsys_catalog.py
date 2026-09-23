#!/usr/bin/env python3
"""
Build a per-source best-z_sys catalogue by combining the OIII results and
grating-source catalogues, both keyed on JELS ID.

For each source, the grating with the highest OIII SNR is selected. That
grating provides:
    z_dja      -> z_<grating>            (from grating catalogue)
    z_sys      -> z_OIII_<grating>       (from OIII results catalogue)
    z_sys_err  -> z_OIII_<grating>_err
    z_sys_snr  -> z_OIII_<grating>_snr

Quality flag (initial, before manual inspection):
    a  -> z_sys_snr > 13.0
    c  -> z_sys_snr <= 13.0
    d  -> no z_sys measurement available

The columns in_muse, edge, duplicate, AO_block, ra and dec are carried through
from the grating catalogue, placed after z_sys_quality.

Sources with no z_sys take their first available grating (in the order below)
and are flagged d.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
OIII_PATH = "/ceph/cephfs/apatrick/P2/jwst_catalogs/OIII_results_by_JELS_ID.csv"
GRATING_PATH = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv"
OUTPUT_PATH = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv"

# Gratings in priority order (used when a source has no z_sys)
GRATINGS = ["G235M", "G235H", "G395M", "G395H"]

SNR_THRESHOLD = 13.0

# Extra columns carried through from the grating catalogue, in output order.
EXTRA_COLS = ["in_muse", "edge", "duplicate", "AO_block", "ra", "dec"]

# Full output column order.
OUTPUT_COLS = [
    "ID", "grating", "z_dja", "z_sys", "z_sys_err", "z_sys_snr",
    "z_sys_quality",
] + EXTRA_COLS

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Reading OIII results from:    {os.path.abspath(OIII_PATH)}")
oiii = pd.read_csv(OIII_PATH)

print(f"Reading grating sources from: {os.path.abspath(GRATING_PATH)}")
grating = pd.read_csv(GRATING_PATH)

# Index both on ID for easy row lookup
oiii = oiii.set_index("ID")
grating = grating.set_index("ID")


# ---------------------------------------------------------------------------
# Per-source selection
# ---------------------------------------------------------------------------
def build_row(source_id):
    """Return a dict with the output columns for one source ID."""
    o_row = oiii.loc[source_id] if source_id in oiii.index else None
    if source_id not in grating.index:
        # ID in OIII catalogue but not the grating catalogue: skip it.
        print(f"  WARNING: ID {source_id} not in grating catalogue, skipping")
        return None
    g_row = grating.loc[source_id]

    # Collect SNR per grating from the OIII catalogue
    snr_by_grating = {}
    if o_row is not None:
        for g in GRATINGS:
            snr_col = f"z_OIII_{g}_snr"
            if snr_col in oiii.columns:
                val = o_row[snr_col]
                if pd.notna(val):
                    snr_by_grating[g] = float(val)

    if snr_by_grating:
        # Grating with the best OIII SNR
        best_grating = max(snr_by_grating, key=snr_by_grating.get)

        z_dja = g_row.get(f"z_{best_grating}", np.nan)
        z_sys = o_row.get(f"z_OIII_{best_grating}", np.nan)
        z_sys_err = o_row.get(f"z_OIII_{best_grating}_err", np.nan)
        z_sys_snr = o_row.get(f"z_OIII_{best_grating}_snr", np.nan)

        if pd.notna(z_sys_snr) and float(z_sys_snr) > SNR_THRESHOLD:
            quality = "a"
        else:
            quality = "c"
    else:
        # No z_sys available: pick first grating the source actually has
        best_grating = np.nan
        for g in GRATINGS:
            has_col = g if g in grating.columns else None
            if has_col is not None and pd.notna(g_row.get(g)) and g_row.get(g) == 1:
                best_grating = g
                break
        # Fall back to first grating with a z_<grating> value if flags absent
        if best_grating is np.nan or (isinstance(best_grating, float) and np.isnan(best_grating)):
            for g in GRATINGS:
                if pd.notna(g_row.get(f"z_{g}", np.nan)):
                    best_grating = g
                    break

        z_dja = g_row.get(f"z_{best_grating}", np.nan) if isinstance(best_grating, str) else np.nan
        z_sys = np.nan
        z_sys_err = np.nan
        z_sys_snr = np.nan
        quality = "d"

    row = {
        "ID": source_id,
        "grating": best_grating,
        "z_dja": z_dja,
        "z_sys": z_sys,
        "z_sys_err": z_sys_err,
        "z_sys_snr": z_sys_snr,
        "z_sys_quality": quality,
    }

    # Carry through the extra columns from the grating catalogue.
    for col in EXTRA_COLS:
        row[col] = g_row.get(col, np.nan) if col in grating.columns else np.nan

    return row


# Use the union of IDs across both catalogues, preserving grating-catalogue order
all_ids = list(grating.index)
for sid in oiii.index:
    if sid not in all_ids:
        all_ids.append(sid)

rows = [build_row(sid) for sid in all_ids]
rows = [r for r in rows if r is not None]
out = pd.DataFrame(rows, columns=OUTPUT_COLS)

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"\nWrote {len(out)} rows to: {os.path.abspath(OUTPUT_PATH)}")
print(f"Quality breakdown: "
      f"a={sum(out.z_sys_quality == 'a')}, "
      f"c={sum(out.z_sys_quality == 'c')}, "
      f"d={sum(out.z_sys_quality == 'd')}")