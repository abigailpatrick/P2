#!/usr/bin/env python
"""Match the JELS grating sources to the PRIMER + MINERVA COSMOS catalogue.

Writes one row per JELS ID with the matched catalogue entry, its photometry
(aperture and total), and the catalogue's own photo-z and LePhare mass, so the
Halpha corrections, the M_UV / beta fits, the SED fits and the master
catalogue all read the same photometry by JELS ID.

The catalogue (Holst et al. in prep., cosmos_primer_minerva_production.fits)
------------------------------------------------------------------------------
  19 bands: ACS F435W F606W F814W, PRIMER NIRCam F090W F115W F150W F200W
  F277W F356W F444W, MINERVA medium bands F140M F162M F182M F210M F250M
  F300M F360M F410M F460M. PSF-homogenised to F444W, Milky Way extinction
  corrected (Gordon et al. 2023), F356W detected, fluxes in uJy.
  Fluxes are in the aperture given by ap_diam (0.3 arcsec, or 0.5 arcsec
  for bright sources). total_correction = (F356W Kron / aperture) x 1.1 is
  the factor that takes every band's aperture flux to total.
  -99 means the source is not imaged in that band.
  Flags combine as digits (56 = Flags 5 and 6, 561 = 5, 6 and 1).
  Flag 0 clean. 1, 2 r_half < 3 or >= 50 px (near a star or bright
  neighbour, total_correction set to its minimum). 3 total_correction
  hit its cap. 4 no photo-z. 5 a photo-z run had chi2 > 5 sigma.
  6 photo-z runs disagree (sigma_z > 0.66). Flags 1-3 concern the
  photometry, 4-6 only the photo-z.

Matching
--------
Nearest catalogue source to each JELS position (RA, Dec from
grating_sources_by_JELS_ID.csv) within --radius arcsec (default 0.3, the
radius the JELS-DJA match used). Both catalogues are F356W detected, so the
positions should agree to well under that. The number of catalogue sources
within the radius is recorded so blends are visible.

Output columns
--------------
  ID, pm_number, pm_sep_arcsec, pm_n_within, pm_ra, pm_dec, pm_flag,
  pm_flag_phot_bad (any of Flags 1, 2 or 3 among its digits), pm_ap_diam, pm_total_correction,
  pm_n_bands, pm_r_half_f356w, pm_z_phot, pm_sigma_z, pm_zspec, pm_zspec_cat,
  pm_z_best, pm_mass_lephare (aperture), pm_mass_lephare_total
  (+ log10 total_correction, as the ReadMe says)
  pm_sep2_arcsec, pm_number2   the second-nearest catalogue object
  pm_f356w_ap_match, pm_f356w_ap_second, pm_second_flux_ratio
  pm_blend       a second object within --blend-radius with at least
                 --blend-flux-ratio of the match's F356W flux. Its light can
                 sit in the aperture and the Kron total correction, so
                 build_ha_flux_corrections.py gives these the sample-median
                 slit factor.
  jels_f356w_ujy, pm_jels_f356w_ratio, pm_jels_outlier
                 F356W of the match divided by the JELS catalogue F356W
                 (0.6 arcsec). The apertures differ, so each source is compared
                 with the median ratio for its aperture, and flagged when it
                 is more than --jels-outlier-factor away. An outlier suggests
                 the wrong object or contaminated photometry.
  and for every band <filt> (lower case, e.g. f444w):
  <filt>_ap, <filt>_ap_err     aperture flux and error, uJy (-99 -> blank)
  <filt>_tot, <filt>_tot_err   x total_correction, uJy

Usage
-----
python match_primer_minerva.py
python match_primer_minerva.py --radius 0.5
"""

import argparse
import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

import p2_common as pc

BANDS = ["f435w", "f606w", "f814w", "f090w", "f115w", "f140m", "f150w", "f162m",
         "f182m", "f200w", "f210m", "f250m", "f277w", "f300m", "f356w", "f360m",
         "f410m", "f444w", "f460m"]
CAT_COLS = {"Number": "pm_number", "RA": "pm_ra", "Dec": "pm_dec", "Flag": "pm_flag",
            "ap_diam": "pm_ap_diam", "total_correction": "pm_total_correction",
            "N_bands": "pm_n_bands", "r_half_f356w": "pm_r_half_f356w",
            "z_phot": "pm_z_phot", "sigma_z": "pm_sigma_z", "zspec": "pm_zspec",
            "zspec_cat": "pm_zspec_cat", "z_best": "pm_z_best",
            "mass_lephare": "pm_mass_lephare"}


def parse_args():
    p = argparse.ArgumentParser(description="Match JELS sources to PRIMER+MINERVA.")
    cat = f"{pc.P2_ROOT}/jwst_catalogs"
    p.add_argument("--sources-csv", default=f"{cat}/grating_sources_by_JELS_ID.csv")
    p.add_argument("--pm-fits", default=f"{cat}/cosmos_primer_minerva_production.fits")
    p.add_argument("--out-csv", default=f"{cat}/primer_minerva_by_JELS_ID.csv")
    p.add_argument("--radius", type=float, default=0.3, help="Match radius, arcsec.")
    p.add_argument("--blend-radius", type=float, default=0.3,
                   help="A second catalogue object within this distance (arcsec) "
                        "of the JELS position can count as a blend.")
    p.add_argument("--blend-flux-ratio", type=float, default=0.3,
                   help="...if its F356W aperture flux is at least this fraction "
                        "of the matched object's.")
    p.add_argument("--jels-cats", nargs="*",
                   default=[f"{cat}/JELS_F356W_DJA_{g}_match_0p3as.fits" for g in
                            ("G235H_F170LP", "G235M_F170LP", "G395H_F290LP", "G395M_F290LP")],
                   help="JELS catalogues for the F356W flux comparison. "
                        "The first holding the ID is used. Skipped if missing.")
    p.add_argument("--jels-f356w-col", default="NIRCam_F356W_APER_600_mas_flux_corr")
    p.add_argument("--jels-f356w-mag", default="NIRCam_F356W_APER_600_mas_mag_corr",
                   help="Used only to find the flux unit (zero point 23.9 uJy, 31.4 nJy).")
    p.add_argument("--jels-outlier-factor", type=float, default=2.0,
                   help="Flag sources whose PRIMER+MINERVA / JELS F356W ratio is more "
                        "than this factor from the median for their aperture.")
    return p.parse_args()


ZP_TO_UJY = {23.9: 1.0, 31.4: 1e-3, 8.9: 1e6, 16.4: 1e3}


def jels_f356w(paths, fcol, mcol):
    """{JELS ID: F356W flux in uJy} from the JELS catalogues, or {} if none found."""
    out = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] JELS catalogue not found, skipped: {path}")
            continue
        t = Table.read(path).to_pandas()
        if fcol not in t.columns:
            print(f"  [WARN] {fcol} not in {path}, skipped")
            continue
        f = pd.to_numeric(t[fcol], errors="coerce").values
        scale = 1.0
        if mcol in t.columns:
            m = pd.to_numeric(t[mcol], errors="coerce").values
            good = np.isfinite(f) & np.isfinite(m) & (f > 0) & (m > 0) & (m < 50)
            if good.any():
                zp = np.median(m[good] + 2.5 * np.log10(f[good]))
                scale = next((v for k, v in ZP_TO_UJY.items() if abs(zp - k) < 0.3), np.nan)
        print(f"  read {os.path.abspath(path)}  (F356W unit scale to uJy {scale:g})")
        for i, val in zip(t["ID"].astype(int), f * scale):
            if i not in out and np.isfinite(val):
                out[i] = float(val)
    return out


def flag_digits(f):
    """Flags are combined by writing the digits together (56 = Flags 5 and 6,
    561 = 5, 6 and 1), so return the set of individual flags."""
    if f is None or not np.isfinite(f):
        return set()
    return {int(c) for c in str(int(f))}


def clean(x):
    """-99 (not imaged) and non-finite values to NaN."""
    x = np.asarray(x, dtype=float).copy()
    x[~np.isfinite(x) | (x <= -98)] = np.nan
    return x


def main():
    a = parse_args()
    print("[CONFIG]")
    print(f"  JELS sources   {os.path.abspath(a.sources_csv)}")
    print(f"  PRIMER+MINERVA {os.path.abspath(a.pm_fits)}")
    print(f"  output         {os.path.abspath(a.out_csv)}")
    print(f"  radius         {a.radius:g} arcsec\n")

    src = pd.read_csv(a.sources_csv)
    src["ID"] = src["ID"].astype(int)
    pm = Table.read(a.pm_fits)
    print(f"  {len(src)} JELS sources, {len(pm)} catalogue sources")

    c_src = SkyCoord(src["ra"].values * u.deg, src["dec"].values * u.deg)
    c_pm = SkyCoord(np.asarray(pm["RA"]) * u.deg, np.asarray(pm["Dec"]) * u.deg)
    idx, sep, _ = c_src.match_to_catalog_sky(c_pm)
    sep = sep.to(u.arcsec).value
    ok = sep <= a.radius

    # how many catalogue sources lie within the radius (blends)
    i_src, _, _, _ = c_pm.search_around_sky(c_src, a.radius * u.arcsec)
    n_within = np.bincount(i_src, minlength=len(src))

    # second-nearest catalogue object, for the blend check
    idx2, sep2, _ = c_src.match_to_catalog_sky(c_pm, nthneighbor=2)
    sep2 = sep2.to(u.arcsec).value
    f356 = clean(np.asarray(pm["f356w"]))

    out = pd.DataFrame({"ID": src["ID"],
                        "pm_sep_arcsec": np.where(ok, sep, np.nan),
                        "pm_n_within": n_within})
    for col, new in CAT_COLS.items():
        vals = np.asarray(pm[col])[idx]
        if vals.dtype.kind == "S":
            vals = np.char.decode(vals).astype(object)
        out[new] = np.where(ok, vals, None if vals.dtype == object else np.nan)
    for c in ("pm_z_phot", "pm_zspec", "pm_z_best", "pm_mass_lephare", "pm_sigma_z"):
        out[c] = clean(out[c])
    flag = pd.to_numeric(out["pm_flag"], errors="coerce")
    out["pm_flag_phot_bad"] = [bool(flag_digits(f) & {1, 2, 3}) if o else np.nan
                               for f, o in zip(flag, ok)]
    tc = pd.to_numeric(out["pm_total_correction"], errors="coerce").values
    out["pm_mass_lephare_total"] = out["pm_mass_lephare"] + np.log10(np.where(tc > 0, tc, np.nan))

    # Check 1 and 3: second object near the JELS position, and whether it is a blend
    f1, f2 = f356[idx], f356[idx2]
    out["pm_sep2_arcsec"] = np.where(ok, sep2, np.nan)
    out["pm_number2"] = np.where(ok, np.asarray(pm["Number"])[idx2], np.nan)
    out["pm_f356w_ap_match"] = np.where(ok, f1, np.nan)
    out["pm_f356w_ap_second"] = np.where(ok, f2, np.nan)
    ratio2 = np.where(np.isfinite(f1) & (f1 > 0), f2 / f1, np.nan)
    out["pm_second_flux_ratio"] = np.where(ok, ratio2, np.nan)
    blend = ok & (sep2 <= a.blend_radius) & ~(ratio2 < a.blend_flux_ratio)
    out["pm_blend"] = np.where(ok, blend, np.nan)

    # Check 2: F356W against the JELS catalogue flux (different apertures,
    # so compare each source with the median ratio for its aperture)
    jels = jels_f356w(a.jels_cats, a.jels_f356w_col, a.jels_f356w_mag)
    out["jels_f356w_ujy"] = out["ID"].map(jels)
    out["pm_jels_f356w_ratio"] = out["pm_f356w_ap_match"] / out["jels_f356w_ujy"]
    out["pm_jels_outlier"] = pd.Series(np.nan, index=out.index, dtype=object)
    if out["pm_jels_f356w_ratio"].notna().any():
        apd = pd.to_numeric(out["pm_ap_diam"], errors="coerce")
        med = out.groupby(apd)["pm_jels_f356w_ratio"].transform("median")
        rel = out["pm_jels_f356w_ratio"] / med
        has = rel.notna()
        out.loc[has, "pm_jels_outlier"] = ((rel[has] > a.jels_outlier_factor)
                                           | (rel[has] < 1.0 / a.jels_outlier_factor))

    for b in BANDS:
        f = clean(np.asarray(pm[b])[idx])
        e = clean(np.asarray(pm[b + "_err"])[idx])
        f[~ok], e[~ok] = np.nan, np.nan
        out[b + "_ap"], out[b + "_ap_err"] = f, e
        out[b + "_tot"], out[b + "_tot_err"] = f * tc, e * tc

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    out.to_csv(a.out_csv, index=False)

    print("\n[SUMMARY]")
    print(f"  matched within {a.radius:g} arcsec   {int(ok.sum())} of {len(src)}")
    if ok.any():
        print(f"  separation median / max      {np.median(sep[ok]):.3f} / {sep[ok].max():.3f} arcsec")
    miss = src.loc[~ok, "ID"].tolist()
    print(f"  unmatched                    {len(miss)}  {miss}")
    multi = out.loc[out["pm_n_within"] > 1, "ID"].tolist()
    print(f"  >1 catalogue source in radius {len(multi)}  {multi}")
    print("  Flag counts (matched)        " +
          ", ".join(f"{int(k)}: {v}" for k, v in flag[ok].value_counts().sort_index().items()))
    bad = out.loc[out["pm_flag_phot_bad"] == True, "ID"].tolist()  # noqa: E712
    print(f"  photometry flag 1-3          {len(bad)}  {bad}")
    if ok.any():
        t = tc[ok]
        print(f"  total_correction median      {np.nanmedian(t):.2f} "
              f"(16-84: {np.nanpercentile(t, 16):.2f}-{np.nanpercentile(t, 84):.2f})")
        ap = pd.to_numeric(out.loc[ok, "pm_ap_diam"], errors="coerce").value_counts()
        print("  aperture used                " + ", ".join(f'{k:g}": {v}' for k, v in ap.items()))
    if out["pm_jels_f356w_ratio"].notna().any():
        apd = pd.to_numeric(out["pm_ap_diam"], errors="coerce")
        meds = out.groupby(apd)["pm_jels_f356w_ratio"].median()
        print("  F356W PRIMER+MINERVA / JELS  median by aperture: " +
              ", ".join(f'{k:g}": {v:.2f}' for k, v in meds.items()))
        print(f"  JELS F356W outliers (>x{a.jels_outlier_factor:g})  "
              f"{out.loc[out['pm_jels_outlier'] == True, 'ID'].tolist()}")  # noqa: E712
    nb = out.loc[out["pm_blend"] == True, "ID"].tolist()  # noqa: E712
    print(f"  blends (2nd object <= {a.blend_radius:g}\" and >= "
          f"{a.blend_flux_ratio:g} x F356W)  {len(nb)}  {nb}")

    # Details for every source with a second object within the blend radius
    near = out[(out["pm_sep2_arcsec"] <= a.blend_radius)].copy()
    if len(near):
        dup = src.set_index("ID").get("duplicate")
        near["dup"] = near["ID"].map(dup) if dup is not None else np.nan
        print("\n[SOURCES WITH A SECOND OBJECT NEARBY]  (F356W aperture fluxes in uJy)")
        print(f"  {'ID':>6} {'dup':>3} {'sep1':>6} {'F356W_1':>8} {'sep2':>6} {'F356W_2':>8} "
              f"{'2nd/1st':>7} {'PM/JELS':>7} {'blend':>5} {'JELS?':>5}")
        for _, r in near.sort_values("ID").iterrows():
            print(f"  {int(r['ID']):>6} {int(r['dup']) if np.isfinite(r['dup']) else 0:>3} "
                  f"{r['pm_sep_arcsec']:6.3f} {r['pm_f356w_ap_match']:8.3f} "
                  f"{r['pm_sep2_arcsec']:6.3f} {r['pm_f356w_ap_second']:8.3f} "
                  f"{r['pm_second_flux_ratio']:7.2f} {r['pm_jels_f356w_ratio']:7.2f} "
                  f"{str(bool(r['pm_blend'])):>5} "
                  f"{'odd' if r['pm_jels_outlier'] == True else '':>5}")  # noqa: E712
        print("  dup = JELS duplicate group (0 = none). sep in arcsec. "
              "PM/JELS = F356W flux ratio of the match to the JELS catalogue.")

    print(f"\nwritten  {os.path.abspath(a.out_csv)}")


if __name__ == "__main__":
    main()