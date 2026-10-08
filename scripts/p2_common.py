#!/usr/bin/env python
"""Shared constants and the systemic-redshift rule for the P2 pipeline.

Every script that needs z_sys takes it from pick_zsys here, so the MUSE steps,
Delta_v, M_UV / beta, the stacks and the master catalogue all use one rule.

The rule, applied to a row of systemic_redshifts.csv (build_zsys_catalog.py)
-----------------------------------------------------------------------------
    1. [OIII] if z_OIII_snr >= 5
    2. Halpha if z_Ha_snr   >= 5
    3. otherwise no z_sys

z_<line>_snr is A / noise, the fitted line amplitude over the flux scatter in
the adjacent continuum bands (lime_fit_lines.py). 5 is the LiMe paper's
detection boundary (Fernandez et al. 2024, Sect. 5.3). Only fits that also
passed the width and centre-error checks reach systemic_redshifts.csv.

systemic_redshifts.csv only carries a z_<line> value when a fit of that line
was flagged a success in lime_<line>_fits.csv (lime_fit_lines.py), and it
already picked the highest-S/N grating where more than one succeeded. The S/N
is A / noise for the line that set the redshift. IDs are Isaac's
PRIMER + MINERVA Numbers.

Hbeta, [NII] and [OII] redshifts are still fitted and kept in that CSV, but are
not used for z_sys.
"""

import numpy as np
import pandas as pd

P2_ROOT = "/ceph/cephfs/apatrick/P2"

C_KMS = 299792.458
LYA_REST = 1215.67          # AA, vacuum

# Lines tried in order, each with its S/N threshold (strictly greater than).
LINE_PRIORITY = ["OIII", "Ha"]
SNR_MIN = {"OIII": 5.0, "Ha": 5.0}

GRATINGS = ["G235M", "G235H", "G395M", "G395H"]


def pick_zsys(row, snr_min=None):
    """Return (z_sys, z_sys_err, z_sys_snr, z_sys_line, z_sys_grating).

    row is one row of systemic_redshifts.csv (a Series or dict).
    snr_min optionally overrides SNR_MIN, e.g. {"OIII": 5, "Ha": 7}.
    Everything is NaN / None when no line passes.
    """
    cuts = dict(SNR_MIN)
    if snr_min:
        cuts.update(snr_min)
    for line in LINE_PRIORITY:
        z = row.get(f"z_{line}")
        snr = row.get(f"z_{line}_snr")
        if z is None or pd.isna(z) or snr is None or pd.isna(snr):
            continue
        if not float(snr) >= cuts[line]:
            continue
        z_err = row.get(f"z_{line}_err")
        grating = row.get(f"z_{line}_grating")
        return (float(z),
                float(z_err) if pd.notna(z_err) else np.nan,
                float(snr),
                line,
                grating if isinstance(grating, str) else None)
    return np.nan, np.nan, np.nan, None, None


def rule_text(snr_min=None):
    """One-line description of the rule, for printing."""
    cuts = dict(SNR_MIN)
    if snr_min:
        cuts.update(snr_min)
    return ", then ".join(f"{l} A/noise >= {cuts[l]:g}" for l in LINE_PRIORITY) + ", else none"


def fallback_grating(g_row, oiii_row=None):
    """Grating whose DJA redshift stands in when there is no z_sys.

    The DJA redshifts of different gratings can disagree by ~1000 km/s for
    the same source, so the choice matters for the MUSE search window. Take
    the grating with the highest attempted [OIII] S/N in
    OIII_results_by_JELS_ID.csv (even an unsuccessful fit shows where the
    rest-optical lines sit), otherwise the first grating observed in
    GRATINGS order. g_row is a row of grating_sources_by_JELS_ID.csv.
    """
    if oiii_row is not None:
        best, best_snr = None, -np.inf
        for g in GRATINGS:
            snr = oiii_row.get(f"z_OIII_{g}_snr")
            if snr is not None and pd.notna(snr) and float(snr) > best_snr:
                best, best_snr = g, float(snr)
        if best is not None and pd.notna(g_row.get(f"z_{best}", np.nan)):
            return best
    for g in GRATINGS:
        flag = g_row.get(g)
        if flag is not None and pd.notna(flag) and int(flag) == 1:
            return g
    for g in GRATINGS:
        if pd.notna(g_row.get(f"z_{g}", np.nan)):
            return g
    return None


def zsys_quality(line):
    """a = [OIII], b = Halpha, d = no z_sys."""
    return {"OIII": "a", "Ha": "b"}.get(line, "d")


def dz_to_kms(z_a, z_b):
    """Velocity of z_a relative to z_b, c (z_a - z_b) / (1 + z_b)."""
    return C_KMS * (np.asarray(z_a, float) - np.asarray(z_b, float)) / (1.0 + np.asarray(z_b, float))