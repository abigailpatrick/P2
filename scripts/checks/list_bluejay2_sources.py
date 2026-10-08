#!/usr/bin/env python
"""List every source in the JELS-DJA match catalogues that has a Blue Jay 2
(bluejay2-v4, PID 5427) spectrum, with its JELS ID, matched Isaac (PRIMER +
MINERVA) Number and the DJA URL the original download used.

Prints to the terminal. Writes a CSV only if --out-csv is given.

For each Blue Jay 2 row it gives
  JELS_ID, grating, grade, z, reviewer (from the match catalogue)
  isaac_ID, isaac_sep     nearest Isaac object to the JELS position
                          (ALPHA_J2000, DELTA_J2000), separation in arcsec
  isaac_ID_dja, sep_dja   nearest Isaac object to the DJA slit position (ra, dec)
  in_muse_old             from grating_sources_by_JELS_ID.csv
  in_new_sample           1 if isaac_ID_dja is in primer_minerva_in_muse.csv
  local_old               1 if the spectrum from the original download is on disk
  http                    status of the DJA URL (with --check-urls)
  url                     https://s3.amazonaws.com/msaexp-nirspec/extractions/<root>/<file>

Usage
-----
python list_bluejay2_sources.py
python list_bluejay2_sources.py --check-urls
python list_bluejay2_sources.py --out-csv /ceph/cephfs/apatrick/P2/jwst_catalogs/bluejay2_sources.csv
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"
OLD_SPEC_DIR = f"{P2}/jwst_spectra"
SERVER = "https://s3.amazonaws.com/msaexp-nirspec/extractions"
GRATING_FILES = {
    "G235H_F170LP": "JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    "G235M_F170LP": "JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    "G395H_F290LP": "JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    "G395M_F290LP": "JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
}
ISAAC = f"{CAT_DIR}/cosmos_primer_minerva_production.fits"
OLD_MERGED = f"{CAT_DIR}/grating_sources_by_JELS_ID.csv"
NEW_MAIN = f"{CAT_DIR}/primer_minerva_in_muse.csv"
ROOT = "bluejay2-v4"


def dec(x):
    return x.decode().strip() if isinstance(x, bytes) else str(x).strip()


def http_status(url):
    r = subprocess.run(["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code}", url],
                       capture_output=True, text=True)
    return r.stdout.strip() or "error"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-urls", action="store_true", help="curl -I every URL")
    p.add_argument("--out-csv", default=None, help="also write the table here")
    a = p.parse_args()

    print("Reading")
    rows = []
    for gf, fname in GRATING_FILES.items():
        path = f"{CAT_DIR}/{fname}"
        print(f"  {path}")
        t = Table.read(path).to_pandas()
        t["root"] = t["root"].apply(dec)
        t = t[t["root"] == ROOT]
        for _, r in t.iterrows():
            rows.append({
                "JELS_ID": int(r["ID"]), "grating": gf,
                "grade": r["grade"], "z": r["z"], "reviewer": dec(r["reviewer"]),
                "ra_jels": r["ALPHA_J2000"], "dec_jels": r["DELTA_J2000"],
                "ra_dja": r["ra"], "dec_dja": r["dec"],
                "root": r["root"], "file": dec(r["file"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No {ROOT} rows found.")
        return

    print(f"  {ISAAC}")
    isaac = Table.read(ISAAC)
    ci = SkyCoord(np.asarray(isaac["RA"], float) * u.deg, np.asarray(isaac["Dec"], float) * u.deg)
    num = np.asarray(isaac["Number"]).astype(int)
    for tag, rc, dc in (("", "ra_jels", "dec_jels"), ("_dja", "ra_dja", "dec_dja")):
        idx, sep, _ = SkyCoord(df[rc].values * u.deg, df[dc].values * u.deg).match_to_catalog_sky(ci)
        df[f"isaac_ID{tag}"] = num[idx]
        df[f"isaac_sep{tag}"] = np.round(sep.arcsec, 3)

    if os.path.exists(OLD_MERGED):
        print(f"  {OLD_MERGED}")
        old = pd.read_csv(OLD_MERGED).set_index("ID")
        df["in_muse_old"] = df["JELS_ID"].map(old["in_muse"]).fillna(-1).astype(int)
    if os.path.exists(NEW_MAIN):
        print(f"  {NEW_MAIN}")
        new_ids = set(pd.read_csv(NEW_MAIN)["ID"].astype(int))
        df["in_new_sample"] = df["isaac_ID_dja"].isin(new_ids).astype(int)

    df["url"] = SERVER + "/" + df["root"] + "/" + df["file"]
    df["local_old"] = [
        int(os.path.exists(f"{OLD_SPEC_DIR}/{g}/{i}_{g}_spectra.fits"))
        for i, g in zip(df["JELS_ID"], df["grating"])]
    if a.check_urls:
        print(f"Checking {len(df)} URLs")
        df["http"] = [http_status(u_) for u_ in df["url"]]

    df = df.sort_values(["JELS_ID", "grating"]).reset_index(drop=True)
    show = ["JELS_ID", "isaac_ID", "isaac_sep", "isaac_ID_dja", "isaac_sep_dja", "grating",
            "grade", "z", "reviewer"]
    show += [c for c in ("in_muse_old", "in_new_sample", "local_old", "http") if c in df]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"\n{len(df)} {ROOT} spectra across {df['JELS_ID'].nunique()} JELS IDs\n")
    print(df[show].to_string(index=False))
    print("\nOriginal download URLs")
    for _, r in df.iterrows():
        print(f"  {r['JELS_ID']:<6d} {r['isaac_ID']:<6d} {r['url']}")

    if "in_muse_old" in df:
        print(f"\nIn the old MUSE footprint: {int((df['in_muse_old'] == 1).sum())} spectra, "
              f"{df.loc[df['in_muse_old'] == 1, 'JELS_ID'].nunique()} JELS IDs")
    if "http" in df:
        print(f"HTTP status counts: {df['http'].value_counts().to_dict()}")

    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
        print(f"\nWrote {os.path.abspath(a.out_csv)}")


if __name__ == "__main__":
    main()