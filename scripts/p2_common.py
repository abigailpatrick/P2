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

# ----------------------------------------------------------------------
# MUSE per-source files
# ----------------------------------------------------------------------
# Every MUSE product for a source lives in MUSE_subcubes/<ID>/, named
# <ID>_<kind>. Use muse_path so all scripts agree on the names.
ZSYS_CATALOG = f"{P2_ROOT}/jwst_catalogs/primer_minerva_in_muse_zsys.csv"
MUSE_SUBCUBE_DIR = f"{P2_ROOT}/MUSE_subcubes"
MUSE_CAT_DIR = f"{P2_ROOT}/MUSE_catalogs"

MUSE_FILES = {
    "subcube": "subcube.fits",                  # extract_subcubes.py
    "subcube_png": "subcube_white.png",
    "contsub": "lya_contsub_cube.fits",         # lya_local_contsub.py
    "continuum": "continuum_model_cube.fits",   # the model, not for Lya
    "contsub_png": "contsub_QA.png",
    "position_png": "lya_position.png",         # optimize_lya_position_grating.py
    "spectrum": "lya_spectrum.npz",             # ap_extract_specs_grating.py
    "sliding_png": "lya_sliding_snr.png",       # sliding_snr_lya_grating.py
    "fit_png": "lya_fit.png",                   # fit_lya_properties_grating.py
    "nb": "lya_nb.fits",
    "mc": "lya_mc.csv",                         # mc_lya_errors_grating.py
    "false_pos": "false_positions.csv",         # false_pos_grating.py
    "false_pos_png": "false_positions.png",
    "false_snr": "false_optimal_snr.csv",       # optimize_lya_positions_f_grating.py
    "systemic_panels_png": "lya_systemic_panels.png",    # plot_lya_vs_systemic.py
    "systemic_overlay_png": "lya_systemic_overlay.png",
    "double_fit_png": "lya_double_fit.png",      # fit_lya_double_peak.py
}

LYA_CATALOG = f"{MUSE_CAT_DIR}/lya_catalog.csv"   # build_lya_catalog.py
LYA_DETECTION_CUTS = f"{MUSE_CAT_DIR}/lya_detection_cuts.csv"   # build_lya_catalog.py

# ----------------------------------------------------------------------
# Lya detection rule (build_lya_catalog.py). By default a source is a
# detection if lya_snr (Sigma_Lya, the 10 AA sliding-window S/N) >= LYA_DET_SNR.
# Instead the threshold can be a percentile of the pooled false-position S/N
# (--det-percentile 90, 95, 98 or 99.5). Optionally also a fixed limit on the
# Delta_v error or on the Lya line-centre error (--extra-cut). The rule used
# is stored in the catalogue as det_rule, e.g. snr5, p95, snr5_dverr100 or
# p98_cenerr60, and every PDF and plot made from the catalogue carries that
# tag in its name. Every other tested source is a non-detection and stays in
# the catalogue.
LYA_DET_SNR = 5.0

# Close pairs. In the groups of Isaac objects sharing one NIRSpec spectrum,
# these IDs own the spectrum and the other member is a lower-redshift
# interloper (groups 4 and 5 of check_close_pairs.py). They stay in the
# analysis and the interlopers are dropped. Every member of the other groups
# is left out of the analysis and shown in its own close-pairs PDFs.
CLOSE_PAIR_KEEP = (50803, 51988)

FALSE_POS_SOURCES = f"{MUSE_CAT_DIR}/lya_false_pos_sources.csv"   # manifest


def muse_path(sid, kind, root=MUSE_SUBCUBE_DIR):
    """Full path of a per-source MUSE file, e.g. muse_path(23070, "contsub")."""
    sid = int(sid)
    return f"{root}/{sid}/{sid}_{MUSE_FILES[kind]}"


def load_sample(path=ZSYS_CATALOG, ids=None, need=("ID", "ra", "dec", "z_sys")):
    """Read the z_sys sample, with ID as int, optionally only some IDs."""
    df = pd.read_csv(path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR {path} has no {missing} column")
    df["ID"] = df["ID"].astype(int)
    if ids:
        absent = sorted(set(ids) - set(df["ID"]))
        if absent:
            print(f"  WARNING IDs not in the catalogue: {absent}")
        df = df[df["ID"].isin(ids)]
    return df.sort_values("ID").reset_index(drop=True)


def load_muse_cube(path):
    """mpdaf Cube with flux [1] and variance [2], taken by index because the
    megacube primary header carries a stray EXTNAME='DATA'."""
    from mpdaf.obj import Cube
    return Cube(path, ext=(1, 2))


def air_to_vac(wave_air):
    """Air to vacuum wavelength in AA, Ciddor (1996), as fit_lya_properties."""
    wave_air = np.asarray(wave_air, dtype=float)
    s2 = (1e4 / wave_air) ** 2
    return wave_air * (1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2))


def vac_to_air(wave_vac):
    """Vacuum to air wavelength in AA, the inverse of air_to_vac."""
    wave_vac = np.asarray(wave_vac, dtype=float)
    s2 = (1e4 / wave_vac) ** 2
    return wave_vac / (1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2))


def lya_obs_air(z):
    """Observed Lya wavelength in AA at redshift z, in air like the MUSE cube.

    z_sys is a vacuum redshift (NIRSpec), so 1215.67 (1 + z) is a vacuum
    wavelength. Placing it on the air-calibrated cube without converting puts
    it about 1.4 AA (about 80 km/s) too red.
    """
    return float(vac_to_air(LYA_REST * (1.0 + z)))


def velocity_band(lya_wave, dv_blue, dv_red):
    """Wavelength edges of the band -dv_blue to +dv_red km/s about lya_wave."""
    return lya_wave * (1.0 - dv_blue / C_KMS), lya_wave * (1.0 + dv_red / C_KMS)


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

# ----------------------------------------------------------------------
# Line S/N
# ----------------------------------------------------------------------
# One definition, used by the position optimiser, the sliding search and the
# false positives. Flux is summed over the channels whose centres fall in the
# window, F = sum(f) dlam, with error sqrt(sum(var)) dlam.
SLIDE_WINDOW = 10.0     # AA
SLIDE_STEP = 1.0        # AA
SLIDE_SMOOTH = 2.0      # Gaussian sigma, in steps


def window_snr(wave, flux, var, lo, hi):
    """(flux, error, S/N) summed over channels with lo <= wave <= hi."""
    m = (wave >= lo) & (wave <= hi) & np.isfinite(flux) & np.isfinite(var)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    dw = float(np.median(np.diff(wave)))
    f = float(np.sum(flux[m]) * dw)
    e = float(np.sqrt(np.sum(var[m])) * dw)
    return f, e, (f / e if e > 0 else np.nan)


def sliding_snr(wave, flux, var, window=SLIDE_WINDOW, step=SLIDE_STEP,
                smooth=SLIDE_SMOOTH, lo=None, hi=None):
    """S/N of a window slid along the spectrum.

    Returns (centres, snr, snr_smooth). lo and hi limit the centres (AA), to
    save time when only a band is needed. The smoothing is a Gaussian of
    sigma = smooth steps, as in the JELS-ID pipeline.
    """
    from scipy.ndimage import gaussian_filter1d
    half = window / 2.0
    c0 = wave.min() + half if lo is None else max(lo, wave.min() + half)
    c1 = wave.max() - half if hi is None else min(hi, wave.max() - half)
    centres = np.arange(c0, c1 + step / 2, step)
    snr = np.array([window_snr(wave, flux, var, c - half, c + half)[2] for c in centres])
    smooth_snr = gaussian_filter1d(np.nan_to_num(snr), sigma=smooth) if smooth else snr
    return centres, snr, smooth_snr


def best_centre(wave, flux, var, band, window=SLIDE_WINDOW, step=SLIDE_STEP,
                smooth=SLIDE_SMOOTH):
    """Centre (AA) and smoothed S/N of the sliding-window peak inside band.

    The S/N is computed 50 AA either side of the band so the smoothing is
    not cut off at the band edges, then the peak is taken inside the band.
    """
    centres, _, sm = sliding_snr(wave, flux, var, window, step, smooth,
                                 lo=band[0] - 50, hi=band[1] + 50)
    m = (centres >= band[0]) & (centres <= band[1])
    if not m.any():
        return np.nan, np.nan
    i = int(np.argmax(sm[m]))
    return float(centres[m][i]), float(sm[m][i])


# ----------------------------------------------------------------------
# Nearest-object rule for the aperture shift
# ----------------------------------------------------------------------
# A trial aperture centre is allowed only if it is at least as close to the
# target as to any other object in Isaac's catalogue, so the shift cannot
# carry the aperture onto a neighbour. Used for the real sources and, with the
# random position as the target, for the false positives.
ISAAC_CATALOG = f"{P2_ROOT}/jwst_catalogs/cosmos_primer_minerva_production.fits"
NEIGHBOUR_RADIUS = 2.0      # arcsec. Beyond ~1.4" no neighbour can win for a
                            # 0.4" shift (twice the largest centre offset)


def load_isaac_positions(path=ISAAC_CATALOG):
    """Number, RA, Dec arrays of Isaac's catalogue."""
    from astropy.table import Table
    t = Table.read(path)
    return (np.asarray(t["Number"], dtype=int), np.asarray(t["RA"], dtype=float),
            np.asarray(t["Dec"], dtype=float))


def neighbours(ra0, dec0, isaac, exclude_id=None, radius=NEIGHBOUR_RADIUS):
    """(ids, ra, dec, sep_arcsec) of Isaac objects within radius of a position,
    nearest first, leaving out exclude_id (the target itself)."""
    num, ra, dec = isaac
    dx = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    dy = (dec - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    m = sep < radius
    if exclude_id is not None:
        m &= num != int(exclude_id)
    o = np.argsort(sep[m])
    return num[m][o], ra[m][o], dec[m][o], sep[m][o]


FALSE_POS_CLEAR = 1.0   # arcsec. For the false positions, an aperture "contains"
                        # an Isaac object when the object is within this of the
                        # aperture centre: the 0.6" radius plus ~0.35" (half the
                        # MUSE PSF FWHM) so light from just outside cannot leak in


def clear_of_objects(ra_c, dec_c, nb_ra, nb_dec, clear=FALSE_POS_CLEAR):
    """True if no object lies within clear arcsec of (ra_c, dec_c)."""
    if len(nb_ra) == 0:
        return True
    cosd = np.cos(np.deg2rad(dec_c))
    d = np.hypot((np.asarray(nb_ra) - ra_c) * cosd, np.asarray(nb_dec) - dec_c) * 3600.0
    return bool(d.min() > clear)


def closer_to_target(ra_c, dec_c, ra0, dec0, nb_ra, nb_dec):
    """True if (ra_c, dec_c) is at least as close to (ra0, dec0) as to every
    neighbour position. Flat-sky separations, fine on arcsecond scales."""
    if len(nb_ra) == 0:
        return True
    cosd = np.cos(np.deg2rad(dec0))
    d_t = np.hypot((ra_c - ra0) * cosd, dec_c - dec0)
    d_n = np.hypot((np.asarray(nb_ra) - ra_c) * cosd, np.asarray(nb_dec) - dec_c)
    return bool(d_t <= d_n.min())


def lya_pdf_groups(cat):
    """Split lya_catalog.csv rows into the sets shown in the PDFs.

    Returns ({"detections", "non_detections", "close_pairs"}: rows sorted by
    lya_snr, highest first), the untested IDs (lya_status not ok) and the
    interloper IDs, which appear nowhere.
    """
    for c in ("lya_det", "pair_status", "in_analysis"):
        if c not in cat.columns:
            raise SystemExit(f"lya_catalog.csv has no {c} column. Rerun build_lya_catalog.py")
    tested = cat[cat["lya_status"] == "ok"].sort_values("lya_snr", ascending=False)
    ana = tested[tested["in_analysis"] == 1]
    groups = {
        "detections": ana[ana["lya_det"] == 1],
        "non_detections": ana[ana["lya_det"] == 0],
        "close_pairs": tested[tested["pair_status"] == "close_pair"],
    }
    untested = cat.loc[cat["lya_status"] != "ok", "ID"].astype(int).tolist()
    interlopers = cat.loc[cat["pair_status"] == "interloper", "ID"].astype(int).tolist()
    return groups, untested, interlopers


def det_rule(cat):
    """The detection-rule tag stored in lya_catalog.csv, e.g. 'p95'."""
    if "det_rule" not in cat.columns or cat["det_rule"].dropna().empty:
        raise SystemExit("lya_catalog.csv has no det_rule column. Rerun build_lya_catalog.py")
    tags = cat["det_rule"].dropna().unique()
    if len(tags) > 1:
        raise SystemExit(f"lya_catalog.csv mixes detection rules {list(tags)}")
    return str(tags[0])


def detection_cuts_path(tag, cat_dir=MUSE_CAT_DIR):
    """The one-row description of a rule written by build_lya_catalog.py,
    kept next to the catalogue."""
    return f"{cat_dir}/lya_detection_cuts_{tag}.csv"


def det_label(r):
    """Short text for a source's detection state, for PDF captions."""
    if r.get("lya_status") != "ok":
        return "not tested"
    if r.get("lya_det") == 1:
        return "detection"
    reason = r.get("det_reason")
    return f"non-detection ({reason})" if isinstance(reason, str) and reason else "non-detection"