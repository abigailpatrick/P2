#!/usr/bin/env python
"""Explain every source that has NO z_OIII measurement in ANY grating.

A source is fine if it has a z_OIII in at least one grating. This audit
targets rows where all four z_OIII_<grating> columns are blank, and works
out why each such source has no systemic redshift anywhere.

For every grating in which the source has a spectrum (flag == 1), it checks
whether [OIII] 5007 at the DJA redshift falls inside that spectrum's
wavelength range. A source is only a genuine problem if [OIII] was in range
in at least one of its available gratings yet still went unmeasured.

Output is a printed report plus a CSV of the findings.

Per-source verdict
------------------
  out_of_range_everywhere   [OIII] outside coverage in every available grating
  no_spectra                flags say no spectrum in any grating
  missing_inputs            spectra flagged but _lime files or DJA z absent
  should_be_measurable      [OIII] in range in >=1 grating yet blank -> investigate
"""

import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


# ----------------------------------------------------------------------
CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
RESULTS_CSV = "/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv"
SPECTRA_BASE = "/ceph/cephfs/apatrick/P2/jwst_spectra"
AUDIT_OUT = "/ceph/cephfs/apatrick/P2/jwst_catalogs/audit_missing_OIII.csv"

OIII_5007_VAC = 5006.843

# grating flag column -> spectra subfolder
GRATINGS = {
    "G235H": "G235H_F170LP",
    "G235M": "G235M_F170LP",
    "G395H": "G395H_F290LP",
    "G395M": "G395M_F290LP",
}

DJA_Z_COL = "z"
DJA_ID_COL = "ID"
# ----------------------------------------------------------------------


def lime_path(src_id, folder):
    return os.path.join(SPECTRA_BASE, folder, f"{src_id}_{folder}_spectra_lime.fits")


def wave_range(path):
    with fits.open(path) as hdul:
        w = np.asarray(hdul["SPECTRUM"].data["WAVE"], dtype=float)
    w = w[np.isfinite(w)]
    return float(np.min(w)), float(np.max(w))


def dja_redshift(src_id, folder):
    cat_path = os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{folder}_match_0p3as.fits")
    cat = Table.read(cat_path)
    mask = cat[DJA_ID_COL] == src_id
    if mask.sum() == 0:
        return None
    return float(cat[DJA_Z_COL][mask][0])


def main():
    df = pd.read_csv(RESULTS_CSV)
    id_col = df.columns[0]

    z_cols = [f"z_OIII_{g}" for g in GRATINGS]
    z_cols = [c for c in z_cols if c in df.columns]

    findings = []

    for _, row in df.iterrows():
        src_id = row[id_col]

        # Skip sources that have a measurement in at least one grating.
        has_any = any(pd.notna(row.get(c)) for c in z_cols)
        if has_any:
            continue

        # This source has no z_OIII anywhere. Work out why, grating by grating.
        available = [g for g in GRATINGS if (g in df.columns) and (row.get(g, 0) == 1)]

        if len(available) == 0:
            findings.append({
                "ID": src_id, "verdict": "no_spectra",
                "detail": "no grating flag set",
            })
            continue

        any_in_range = False
        any_input_missing = False
        details = []

        for g in available:
            folder = GRATINGS[g]
            path = lime_path(src_id, folder)
            if not os.path.exists(path):
                any_input_missing = True
                details.append(f"{g}:no_lime_file")
                continue
            z = dja_redshift(src_id, folder)
            if z is None:
                any_input_missing = True
                details.append(f"{g}:no_dja_z")
                continue
            oiii_obs = OIII_5007_VAC * (1.0 + z)
            wmin, wmax = wave_range(path)
            if (oiii_obs >= wmin) and (oiii_obs <= wmax):
                any_in_range = True
                details.append(f"{g}:in_range({oiii_obs:.0f}AA in {wmin:.0f}-{wmax:.0f})")
            else:
                details.append(f"{g}:out_of_range({oiii_obs:.0f}AA vs {wmin:.0f}-{wmax:.0f})")

        if any_in_range:
            verdict = "should_be_measurable"
        elif any_input_missing:
            verdict = "missing_inputs"
        else:
            verdict = "out_of_range_everywhere"

        findings.append({
            "ID": src_id, "verdict": verdict, "detail": "; ".join(details),
        })

    out = pd.DataFrame(findings)
    out.to_csv(AUDIT_OUT, index=False)

    print(f"results CSV   {RESULTS_CSV}")
    print(f"audit written {AUDIT_OUT}")
    print(f"\nsources with NO z_OIII in any grating: {len(out)}")
    if len(out) == 0:
        print("Every source has a systemic redshift in at least one grating.")
        return

    print("\nbreakdown by verdict:")
    print(out["verdict"].value_counts().to_string())

    flagged = out[out["verdict"] == "should_be_measurable"]
    if len(flagged) > 0:
        print(f"\n*** {len(flagged)} should be measurable, investigate these ***")
        print(flagged.to_string(index=False))
    else:
        print("\nNo should-be-measurable cases. Every fully-blank source is explained.")


if __name__ == "__main__":
    main()