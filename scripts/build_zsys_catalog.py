#!/usr/bin/env python
"""Build the systemic redshift catalogue of the PRIMER + MINERVA sample.

Replaces merge_systematic_redshifts.py and the old build_zsys_catalog.py.
Reads the per-line LiMe summaries from lime_fit_lines.py, gathers the best
successful fit of each line per source, and applies the z_sys rule in
p2_common.pick_zsys:

    1. [OIII] if it is detected, A/noise >= --oiii-snr-min (5)
    2. otherwise Halpha if detected, A/noise >= --ha-snr-min (5)
    3. otherwise no z_sys, and the source is dropped from the z_sys sample

A/noise is the fitted amplitude over the flux scatter in the adjacent
continuum bands, and 5 is the LiMe paper's detection boundary (Fernandez et
al. 2024, Sect. 5.3). A fit only counts if it also has FWHM >= 1 pixel and a
centre error <= 2.5 Angstrom (lime_fit_lines.py). A line's "best" fit is the
grating with the highest A/noise among those fits, so where a source has
[OIII] in more than one grating, the strongest one sets z_sys. Hbeta and [OII] are kept in systemic_redshifts.csv
for reference but are not used for z_sys.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse.csv
  jwst_catalogs/lime_OIII_summary.csv, lime_Ha_summary.csv,
  lime_Hbeta_summary.csv, lime_OII_summary.csv

Outputs
-------
  jwst_catalogs/systemic_redshifts.csv
      every source in primer_minerva_in_muse.csv, with z_<line>, z_<line>_err,
      z_<line>_snr, z_<line>_grating for OIII, Ha, Hbeta, OII, and the z_sys
      columns below (blank where there is no z_sys)
  jwst_catalogs/primer_minerva_in_muse_zsys.csv
      the rows of primer_minerva_in_muse.csv with a z_sys, plus
      z_sys, z_sys_err, z_sys_snr, z_sys_line, z_sys_grating, and
      dv_sys_dja_kms, the DJA redshift of the z_sys grating relative to z_sys

Usage
-----
python build_zsys_catalog.py --dry-run
python build_zsys_catalog.py
python build_zsys_catalog.py --oiii-snr-min 10 --ha-snr-min 10
"""

import argparse
import os

import numpy as np
import pandas as pd

import p2_common as pc

P2 = pc.P2_ROOT
CAT_DIR = f"{P2}/jwst_catalogs"
LINES = ["OIII", "Ha", "Hbeta", "OII"]
ZSYS_COLS = ["z_sys", "z_sys_err", "z_sys_snr", "z_sys_line", "z_sys_grating", "dv_sys_dja_kms"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=f"{CAT_DIR}/primer_minerva_in_muse.csv")
    p.add_argument("--lime-dir", default=CAT_DIR, help="folder of lime_<line>_summary.csv")
    p.add_argument("--out-systemic", default=f"{CAT_DIR}/systemic_redshifts.csv")
    p.add_argument("--out-zsys", default=f"{CAT_DIR}/primer_minerva_in_muse_zsys.csv")
    p.add_argument("--oiii-snr-min", type=float, default=pc.SNR_MIN["OIII"])
    p.add_argument("--ha-snr-min", type=float, default=pc.SNR_MIN["Ha"])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def why_dropped(r, cuts):
    """Short reason a source has no z_sys."""
    parts = []
    for line in ("OIII", "Ha"):
        if pd.notna(r.get(f"z_{line}")):
            parts.append(f"{line} A/noise {r[f'z_{line}_snr']:.1f} ({r[f'z_{line}_grating']}) "
                         f"below {cuts[line]:g}")
        elif r.get(f"{line}_fitted"):
            parts.append(f"{line} fitted in {r[f'{line}_fitted']} but never successful")
        else:
            parts.append(f"{line} not covered or not fitted")
    return "; ".join(parts)


def main():
    a = parse_args()
    cuts = {"OIII": a.oiii_snr_min, "Ha": a.ha_snr_min}
    print("build_zsys_catalog.py")
    print(f"  catalogue      {os.path.abspath(a.catalog)}")
    print(f"  z_sys rule     {pc.rule_text(cuts)}")

    cat = pd.read_csv(a.catalog)
    cat["ID"] = cat["ID"].astype(int)
    sysz = cat[["ID"]].copy()
    for line in LINES:
        path = os.path.abspath(os.path.join(a.lime_dir, f"lime_{line}_summary.csv"))
        if not os.path.exists(path):
            print(f"  MISSING        {path}")
            for s in ("", "_err", "_snr", "_grating"):
                sysz[f"z_{line}{s}"] = np.nan
            sysz[f"{line}_fitted"] = ""
            continue
        print(f"  reading        {path}")
        s = pd.read_csv(path)
        s["ID"] = s["ID"].astype(int)
        s = s[["ID", f"z_{line}", f"z_{line}_err", f"z_{line}_snr", f"z_{line}_grating",
               "gratings_fitted"]].rename(columns={"gratings_fitted": f"{line}_fitted"})
        sysz = sysz.merge(s, on="ID", how="left")
        sysz[f"{line}_fitted"] = sysz[f"{line}_fitted"].fillna("")

    picks = [pc.pick_zsys(r, cuts) for _, r in sysz.iterrows()]
    sysz["z_sys"] = [p[0] for p in picks]
    sysz["z_sys_err"] = [p[1] for p in picks]
    sysz["z_sys_snr"] = [p[2] for p in picks]
    sysz["z_sys_line"] = [p[3] for p in picks]
    sysz["z_sys_grating"] = [p[4] for p in picks]
    z_dja = [cat.set_index("ID").loc[i, f"z_{g}"] if isinstance(g, str) else np.nan
             for i, g in zip(sysz["ID"], sysz["z_sys_grating"])]
    sysz["dv_sys_dja_kms"] = pc.dz_to_kms(np.asarray(z_dja, float), sysz["z_sys"])

    has = sysz["z_sys"].notna()
    zsys = cat.merge(sysz.loc[has, ["ID"] + ZSYS_COLS], on="ID", how="inner")
    lead = ["ID", "ra", "dec"] + ZSYS_COLS
    zsys = zsys[lead + [c for c in zsys.columns if c not in lead]]

    print("\n--- Summary ---")
    print(f"  sources in catalogue          {len(cat)}")
    for line in LINES:
        print(f"  successful {line:<6} fit          {int(sysz[f'z_{line}'].notna().sum())}")
    print(f"  z_sys from [OIII]             {int((sysz['z_sys_line'] == 'OIII').sum())}")
    print(f"  z_sys from Halpha             {int((sysz['z_sys_line'] == 'Ha').sum())}")
    print(f"  z_sys by grating              "
          + ", ".join(f"{g} {n}" for g, n in sysz["z_sys_grating"].value_counts().items()))
    print(f"  with z_sys (kept)             {int(has.sum())}")
    print(f"  without z_sys (dropped)       {int((~has).sum())}")
    dv = sysz.loc[has, "dv_sys_dja_kms"]
    if len(dv):
        print(f"  DJA z - z_sys                 median {np.nanmedian(dv):+.0f} km/s, "
              f"|dv| > 300 km/s for {int((np.abs(dv) > 300).sum())}")

    print("\nDropped sources")
    for _, r in sysz[~has].iterrows():
        print(f"  {r['ID']:<7d} {why_dropped(r, cuts)}")

    if os.path.exists(a.out_zsys):
        old = set(pd.read_csv(a.out_zsys)["ID"].astype(int))
        new = set(zsys["ID"])
        print(f"\nCompared with the existing {a.out_zsys}")
        print(f"  gained {sorted(new - old)}")
        print(f"  lost   {sorted(old - new)}")

    if a.dry_run:
        print("\nDry run, nothing written.")
        return
    sysz.to_csv(a.out_systemic, index=False)
    zsys.to_csv(a.out_zsys, index=False)
    print("\nWrote")
    print(f"  {os.path.abspath(a.out_systemic)}")
    print(f"  {os.path.abspath(a.out_zsys)}")


if __name__ == "__main__":
    main()