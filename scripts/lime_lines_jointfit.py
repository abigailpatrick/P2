#!/usr/bin/env python
"""Fit Hbeta, Halpha, [NII] 6584, and the [OII] 3726,3729 doublet across all
four gratings and build per-line result CSVs.

Companion to lime_OIII_jointfit.py. Same loader, same band geometry, same
success criteria, generalised over a LINES config list instead of hardcoding
[OIII]. Written for LiMe 2.4.3.

Per source, per grating, per line
----------------------------------
1. Work out whether the line (or, for OII, the padded doublet band) falls
   inside that spectrum's observed wavelength range at the DJA redshift. If
   not, skip that line for that grating.
2. Fit a local continuum, then fit the line: a single Gaussian for Hbeta, Ha
   and NII, or a kinematically-tied doublet blend for OII.
3. Unlike [OIII] 4959/5007, the OII 3726/3729 flux ratio is NOT fixed by
   atomic physics (it is density-dependent, roughly 0.35-1.5), so only the
   kinematics are tied between the two OII components. Both amplitudes are
   left free.

A NOTE ON LINE LABELS
----------------------
Labels follow the O3_5007A convention already used for [OIII]: element +
ionisation-stage digit, underscore, nearest integer vacuum wavelength, "A".
    Hbeta   -> H1_4861A   (vacuum 4862.683 AA)
    Ha      -> H1_6563A   (vacuum 6564.632 AA)
    NII     -> N2_6584A   (vacuum 6585.270 AA, stronger of the doublet)
    OII     -> O2_3726A, O2_3729A, blend O2_3729A_b
These are user-defined labels passed straight into spec.fit.bands, not
looked up from LiMe's own line database, so LiMe accepts them regardless.
Still worth a quick sanity check against spec.retrieve.lines_frame() on a
handful of sources before trusting the sample in bulk, in case your local
LiMe build expects air rather than vacuum wavelengths baked into the label.

Outputs, all written fresh (old files overwritten), one pair per line
-----------------------------------------------------------------------
  <name>_results_by_JELS_ID.csv   one row per source: per-grating DJA z,
                                   and for each grating actually fitted,
                                   z_<name>_<gr>, its error, and its S/N.
  <name>_summary_by_JELS_ID.csv   one row per source: per-grating success
                                   flag and a <name>_success count.
  Figures under JWST_SPECTRA_ROOT/<name>_fits/<ID>/, grouping the contsub
  and line-fit plots for every grating of one source together.

Success. A fit is successful when BOTH the detection S/N clears SNR_MIN and
the fitted centre error is finite and below CENTRE_ERR_MAX_AA. Both
defaults match the [OIII] script; they have not been separately validated
per line the way lime_diagnostics.py did for [OIII], so treat them as a
starting point.

Usage
-----
python lime_lines_jointfit.py                 # walks the default grating dirs
python lime_lines_jointfit.py <parent_dir>    # override the spectra root

Filename convention expected: <ID>_<grating>_spectra_lime.fits
"""

import os
import sys
import re
import glob

import numpy as np
import pandas as pd
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")

import lime


# ----------------------------------------------------------------------
# Fixed configuration. Paths match ways-of-working.md / lime_OIII_jointfit.py.
# ----------------------------------------------------------------------
CATALOG_DIR = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
JWST_SPECTRA_ROOT = "/ceph/cephfs/apatrick/P2/jwst_spectra"

# The four gratings and their catalogue files.
GRATINGS = ["G235M_F170LP", "G235H_F170LP", "G395M_F290LP", "G395H_F290LP"]

# DJA catalogue column names.
DJA_Z_COL = "z"
DJA_ID_COL = "ID"

# --- Band geometry, rest-frame Angstrom. Same as lime_OIII_jointfit.py. ---
LINE_MARGIN = 15.0     # w3-w4 extends this far beyond each outer line
CONT_GAP = 5.0          # gap between line region and each continuum flank
CONT_WIDTH = 20.0       # width of each continuum flank
# ---------------------------------------------------------------------

# --- Success criteria. Both must pass. Same defaults as [OIII]. ---
SNR_MIN = 13.0
CENTRE_ERR_MAX_AA = 2.5
# -------------------------------------------------------------------

# ----------------------------------------------------------------------
# Line definitions. Vacuum rest wavelengths, Angstrom.
# kind "single" fits one Gaussian. kind "doublet" fits a kinematically
# tied blend. amp_ratio_theory is only set where the ratio is fixed by
# atomic physics; leave None to fit both amplitudes free (OII).
# ----------------------------------------------------------------------
LINES = [
    {
        "name": "Hbeta",
        "kind": "single",
        "label": "H1_4861A",
        "rest_vac": 4862.683,
    },
    {
        "name": "Ha",
        "kind": "single",
        "label": "H1_6563A",
        "rest_vac": 6564.632,
    },
    {
        "name": "NII",
        "kind": "single",
        "label": "N2_6584A",
        "rest_vac": 6585.270,
    },
    {
        "name": "OII",
        "kind": "doublet",
        "label_blend": "O2_3729A_b",
        "label_ref": "O2_3729A",      # redder component, used for z
        "label_other": "O2_3726A",
        "rest_vac_ref": 3729.875,
        "rest_vac_other": 3727.092,
        "amp_ratio_theory": None,      # density-dependent, not fixed
    },
]
# ----------------------------------------------------------------------


def results_csv(line):
    return os.path.join(CATALOG_DIR, f"{line['name']}_results_by_JELS_ID.csv")


def summary_csv(line):
    return os.path.join(CATALOG_DIR, f"{line['name']}_summary_by_JELS_ID.csv")


def fig_root(line):
    return os.path.join(JWST_SPECTRA_ROOT, f"{line['name']}_fits")


def grating_short(grating):
    """G235M_F170LP -> G235M, for column and filename use."""
    return grating.split("_")[0]


def catalogue_path(grating):
    return os.path.join(CATALOG_DIR, f"JELS_F356W_DJA_{grating}_match_0p3as.fits")


def parse_id_grating(lime_path):
    """Pull the integer ID and grating string from the filename."""
    name = os.path.basename(lime_path)
    m = re.match(r"^(\d+)_([A-Z0-9]+_[A-Z0-9]+)_spectra_lime\.fits$", name)
    if m is None:
        raise ValueError(f"Cannot parse ID and grating from filename: {name}")
    src_id = int(m.group(1))
    grating = m.group(2)
    if grating not in GRATINGS:
        raise ValueError(f"Grating {grating} not one of {GRATINGS}")
    return src_id, grating


def get_dja_redshift(cat, src_id):
    """DJA redshift for this ID from an already-loaded catalogue table."""
    mask = cat[DJA_ID_COL] == src_id
    if mask.sum() == 0:
        return None
    return float(cat[DJA_Z_COL][mask][0])


def load_lime_spectrum(lime_path, redshift):
    """Load the _lime.fits SPECTRUM table into a lime.Spectrum."""
    from astropy.io import fits

    with fits.open(lime_path) as hdul:
        tab = hdul["SPECTRUM"].data
        wave = np.asarray(tab["WAVE"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["ERR"], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave, flux, err = wave[good], flux[good], err[good]

    norm_flux = np.nanmedian(np.abs(flux[flux > 0])) if np.any(flux > 0) else 1e-20

    spec = lime.Spectrum(
        input_wave=wave,
        input_flux=flux,
        input_err=err,
        redshift=redshift,
        units_wave="AA",
        units_flux="FLAM",
        norm_flux=norm_flux,
    )
    return spec, wave


def single_band_df(label, rest_vac, line_margin=LINE_MARGIN,
                    cont_gap=CONT_GAP, cont_width=CONT_WIDTH):
    w3 = rest_vac - line_margin
    w4 = rest_vac + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [rest_vac],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[label],
    )


def doublet_band_df(label, rest_vac_lo, rest_vac_hi, line_margin=LINE_MARGIN,
                     cont_gap=CONT_GAP, cont_width=CONT_WIDTH):
    """One-row bands frame whose line region spans both doublet components."""
    w3 = rest_vac_lo - line_margin
    w4 = rest_vac_hi + line_margin
    w2 = w3 - cont_gap
    w1 = w2 - cont_width
    w5 = w4 + cont_gap
    w6 = w5 + cont_width
    return pd.DataFrame(
        {"wavelength": [rest_vac_hi],
         "w1": [w1], "w2": [w2], "w3": [w3],
         "w4": [w4], "w5": [w5], "w6": [w6]},
        index=[label],
    )


def band_in_range(band_row, wave, zf):
    lo = float(band_row["w1"]) * zf
    hi = float(band_row["w6"]) * zf
    return (lo >= np.nanmin(wave)) and (hi <= np.nanmax(wave))


def centre_to_redshift(centre_obs_aa, rest_vac):
    return centre_obs_aa / rest_vac - 1.0


def centre_err_to_redshift_err(centre_err_aa, rest_vac):
    if centre_err_aa is None or not np.isfinite(centre_err_aa):
        return np.nan
    return centre_err_aa / rest_vac


def build_doublet_fit_cfg(line):
    """fit_cfg for the doublet blend. Kinematics tied, amplitude free
    unless amp_ratio_theory is set."""
    cfg = {
        line["label_blend"]: f"{line['label_other']}+{line['label_ref']}",
        f"{line['label_other']}_kinem": line["label_ref"],
    }
    if line.get("amp_ratio_theory") is not None:
        cfg[f"{line['label_other']}_amp"] = {
            "expr": f"{line['label_ref']}_amp/{line['amp_ratio_theory']}"
        }
    return cfg


def get_line_row(spec, label):
    """Return centre, centre_err, and detection S/N for one line label.

    S/N is LiMe's own snr_line (amplitude-based, Rola et al. definition).
    Falls back to profile_flux / its error only if snr_line is absent.
    """
    if label not in spec.frame.index:
        return np.nan, np.nan, np.nan
    row = spec.frame.loc[label]
    cols = spec.frame.columns

    centre = float(row["center"]) if "center" in cols else np.nan
    centre_err = (
        float(row["center_err"])
        if "center_err" in cols and np.isfinite(row["center_err"]) else np.nan
    )

    snr = np.nan
    if "snr_line" in cols and np.isfinite(row["snr_line"]):
        snr = float(row["snr_line"])
    elif "profile_flux" in cols and "profile_flux_err" in cols:
        f, fe = row["profile_flux"], row["profile_flux_err"]
        if np.isfinite(f) and np.isfinite(fe) and fe > 0:
            snr = float(f / fe)
    return centre, centre_err, snr


def is_success(snr, centre_err_aa):
    snr_ok = np.isfinite(snr) and snr >= SNR_MIN
    err_ok = np.isfinite(centre_err_aa) and centre_err_aa <= CENTRE_ERR_MAX_AA
    return bool(snr_ok and err_ok)


def fit_one_line(lime_path, z_dja, line):
    """Fit one line in one spectrum.

    Returns a dict with status 'fitted' or 'out_of_range', plus z, z_err,
    snr, success for 'fitted'. Raises on genuine fit errors.
    """
    src_id, grating = parse_id_grating(lime_path)
    name = line["name"]

    spec, wave = load_lime_spectrum(lime_path, z_dja)
    zf = 1.0 + z_dja

    if line["kind"] == "single":
        band = single_band_df(line["label"], line["rest_vac"])
        ref_label = line["label"]
        rest_vac = line["rest_vac"]
    else:
        band = doublet_band_df(line["label_blend"], line["rest_vac_other"],
                                line["rest_vac_ref"])
        ref_label = line["label_ref"]
        rest_vac = line["rest_vac_ref"]

    if not band_in_range(band.iloc[0], wave, zf):
        print(f"    {name}  out of range")
        return {"status": "out_of_range"}

    fdir = os.path.join(fig_root(line), str(src_id))
    os.makedirs(fdir, exist_ok=True)
    contsub_png = os.path.join(fdir, f"{src_id}_{grating}_{name}_contsub.png")
    fit_png = os.path.join(fdir, f"{src_id}_{grating}_{name}_fit.png")

    spec.fit.continuum(degree_list=[3, 4], emis_threshold=[3, 2])
    spec.plot.spectrum(fname=contsub_png)

    if line["kind"] == "single":
        spec.fit.bands(line["label"], bands=band)
    else:
        fit_cfg = build_doublet_fit_cfg(line)
        spec.fit.bands(line["label_blend"], bands=band, fit_cfg=fit_cfg)

    centre, centre_err, snr = get_line_row(spec, ref_label)
    z_line = centre_to_redshift(centre, rest_vac)
    z_line_err = centre_err_to_redshift_err(centre_err, rest_vac)
    success = is_success(snr, centre_err)

    print(f"    {name}  z {z_line:.6f} +/- {z_line_err:.6f}  "
          f"S/N {snr:.2f}  centre_err {centre_err} AA  success {success}")

    spec.plot.bands(ref_label, fname=fit_png)

    return {
        "status": "fitted",
        "z": z_line,
        "z_err": z_line_err,
        "snr": snr,
        "success": success,
    }


def run_all(spectra_root, line):
    """Walk all four grating dirs, fit one line for every source, build
    that line's pair of CSVs."""
    name = line["name"]

    catalogues = {}
    for grating in GRATINGS:
        cp = catalogue_path(grating)
        if os.path.exists(cp):
            catalogues[grating] = Table.read(cp)
        else:
            print(f"catalogue missing, skipping grating: {cp}")

    results = {}   # src_id -> {col: value}
    summary = {}   # src_id -> {col: value}
    gratings_seen = {}

    for grating in GRATINGS:
        if grating not in catalogues:
            continue
        gr = grating_short(grating)
        cat = catalogues[grating]
        spec_dir = os.path.join(spectra_root, grating)
        files = sorted(glob.glob(os.path.join(spec_dir, "*_lime.fits")))
        print(f"\n{'#'*60}\n# {name}  {grating}   {len(files)} files\n{'#'*60}")

        for f in files:
            try:
                src_id, _ = parse_id_grating(f)
            except ValueError as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                continue

            z_dja = get_dja_redshift(cat, src_id)
            if z_dja is None:
                print(f"SKIPPED  {os.path.basename(f)}  no DJA z in catalogue")
                continue

            results.setdefault(src_id, {"ID": src_id})
            summary.setdefault(src_id, {"ID": src_id, f"{name}_success": 0})
            gratings_seen.setdefault(src_id, [])
            gratings_seen[src_id].append(gr)
            results[src_id][f"z_{gr}"] = z_dja

            print(f"\n--- ID {src_id}  {grating} ---")
            try:
                r = fit_one_line(f, z_dja, line)
            except Exception as err:
                print(f"SKIPPED  {os.path.basename(f)}  reason: {err}")
                summary[src_id][f"{name}_{gr}_success"] = False
                continue

            if r["status"] == "out_of_range":
                summary[src_id][f"{name}_{gr}_success"] = False
                continue

            results[src_id][f"z_{name}_{gr}"] = r["z"]
            results[src_id][f"z_{name}_{gr}_err"] = r["z_err"]
            results[src_id][f"z_{name}_{gr}_snr"] = r["snr"]

            summary[src_id][f"{name}_{gr}_success"] = r["success"]
            if r["success"]:
                summary[src_id][f"{name}_success"] += 1

    for src_id, grs in gratings_seen.items():
        summary[src_id]["gratings"] = ", ".join(grs)

    res_df = pd.DataFrame(list(results.values())).sort_values("ID")
    res_cols = ["ID"]
    for gr in [grating_short(g) for g in GRATINGS]:
        if f"z_{gr}" in res_df.columns:
            res_cols.append(f"z_{gr}")
    for gr in [grating_short(g) for g in GRATINGS]:
        for suff in ("", "_err", "_snr"):
            c = f"z_{name}_{gr}{suff}"
            if c in res_df.columns:
                res_cols.append(c)
    res_df = res_df.reindex(columns=res_cols)
    res_df.to_csv(results_csv(line), index=False)

    sum_df = pd.DataFrame(list(summary.values())).sort_values("ID")
    sum_cols = ["ID", "gratings"]
    for gr in [grating_short(g) for g in GRATINGS]:
        c = f"{name}_{gr}_success"
        if c in sum_df.columns:
            sum_cols.append(c)
    sum_cols.append(f"{name}_success")
    sum_df = sum_df.reindex(columns=sum_cols)
    sum_df.to_csv(summary_csv(line), index=False)

    print("\n" + "=" * 60)
    print(f"DONE  {name}")
    print(f"  sources processed   {len(results)}")
    if len(sum_df):
        print(f"  with >=1 success    {int((sum_df[f'{name}_success'] > 0).sum())}")
    print(f"  results csv         {results_csv(line)}")
    print(f"  summary csv         {summary_csv(line)}")
    print(f"  figures root        {fig_root(line)}")
    print("=" * 60)


def main():
    spectra_root = sys.argv[1] if len(sys.argv) == 2 else JWST_SPECTRA_ROOT

    print("lime_lines_jointfit.py")
    print(f"  spectra root   {spectra_root}")
    print(f"  catalogue dir  {CATALOG_DIR}")
    print("  lines          " + ", ".join(l["name"] for l in LINES))
    print("  output files")
    for line in LINES:
        print(f"    {results_csv(line)}")
        print(f"    {summary_csv(line)}")
        print(f"    {fig_root(line)}/<ID>/")

    for line in LINES:
        os.makedirs(fig_root(line), exist_ok=True)
        run_all(spectra_root, line)


if __name__ == "__main__":
    main()