#!/usr/bin/env python
"""Download the DJA NIRSpec spectra for the PRIMER + MINERVA sample and plot each one.

Reads primer_minerva_in_muse.csv (written by build_primer_minerva_in_muse.py),
which holds, for every Isaac ID and grating, the root and file of the best DJA
spectrum. Each spectrum is fetched from the public DJA S3 bucket as

    https://s3.amazonaws.com/msaexp-nirspec/extractions/<root>/<file>

and saved with a simple 1D plot as

    <out-base>/<grating>_<filter>/<ID>_<grating>_<filter>_spectra.fits
    <out-base>/<grating>_<filter>/<ID>_<grating>_<filter>_spectra.png

A spectrum shared by two IDs (a close pair) is downloaded once and copied to
the second ID. A log of every file, its URL and what happened to it is
written to <out-base>/download_log.csv.

Existing files are skipped unless --overwrite is given, so a rerun only picks
up what failed.

Usage
-----
python extract_nirspec_spectra.py --dry-run          # list what would be fetched
python extract_nirspec_spectra.py                    # all IDs, all gratings
python extract_nirspec_spectra.py --ids 50914 57866  # selected IDs
python extract_nirspec_spectra.py --gratings G395M --overwrite
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits

P2 = "/ceph/cephfs/apatrick/P2"
CATALOG = f"{P2}/jwst_catalogs/primer_minerva_in_muse.csv"
OUT_BASE = f"{P2}/jwst_spectra_pm"
SERVER = "https://s3.amazonaws.com/msaexp-nirspec/extractions"

GRATINGS = ["G235H", "G235M", "G395H", "G395M"]
GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=CATALOG)
    p.add_argument("--out-base", default=OUT_BASE)
    p.add_argument("--server", default=SERVER)
    p.add_argument("--ids", nargs="+", type=int, default=None)
    p.add_argument("--gratings", nargs="+", default=GRATINGS, choices=GRATINGS)
    p.add_argument("--overwrite", action="store_true", help="re-download existing files")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list downloads, fetch nothing")
    return p.parse_args()


def curl_download(url, out_path):
    tmp = out_path + ".part"
    cmd = ["curl", "-fsSL", "--retry", "3", "--retry-delay", "5", url, "-o", tmp]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"curl exit {err.returncode}") from err
    os.replace(tmp, out_path)


def plot_spectrum_1d(fits_path, png_path, title=None):
    """1D flux vs wavelength. HDU 1 has wave (micron), flux and err (uJy)."""
    with fits.open(fits_path) as hdu:
        spec1d = hdu[1].data
    wave = spec1d["wave"] * u.micron
    fnu = spec1d["flux"] * u.uJy
    fnu_err = spec1d["err"] * u.uJy
    flam = fnu.to(u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(wave))
    flam_err = fnu_err / fnu * flam
    norm = 1e-20 * u.erg / u.s / u.cm ** 2 / u.AA
    f = (flam / norm).value
    e = (flam_err / norm).value

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axhline(0.0, ls="--", color="0.6", lw=1)
    ax.fill_between(wave.value, f + e, f - e, color="0.5", alpha=0.5,
                    step="mid", linewidth=0.0)
    ax.step(wave.value, f, where="mid", color="k", lw=1)
    fin = np.isfinite(f)
    if fin.any():
        ax.set_ylim(np.nanpercentile(f[fin], 1), 1.4 * np.nanmax(f[fin]))
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"$F_{\lambda}\ [10^{-20}\,{\rm erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}]$")
    if title:
        ax.set_title(title, size=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    print("NIRSpec spectra download")
    print(f"  catalogue  {os.path.abspath(args.catalog)}")
    print(f"  output     {os.path.abspath(args.out_base)}")
    print(f"  server     {args.server}")
    print(f"  overwrite  {args.overwrite}   dry run  {args.dry_run}")

    cat = pd.read_csv(args.catalog, keep_default_na=False, na_values=[""])
    if args.ids is not None:
        cat = cat[cat["ID"].isin(args.ids)]
        missing = sorted(set(args.ids) - set(cat["ID"]))
        if missing:
            print(f"  IDs not in the catalogue: {missing}")
    print(f"  {len(cat)} IDs")

    jobs = []
    for _, row in cat.iterrows():
        for g in args.gratings:
            if int(row.get(g, 0)) != 1:
                continue
            root, fname = str(row[f"root_{g}"]), str(row[f"file_{g}"])
            gf = f"{g}_{GRATING_FILTER[g]}"
            base = f"{int(row['ID'])}_{gf}_spectra"
            out_dir = os.path.join(args.out_base, gf)
            jobs.append({"ID": int(row["ID"]), "grating": gf, "root": root, "file": fname,
                         "url": f"{args.server}/{root}/{fname}",
                         "fits_path": os.path.abspath(os.path.join(out_dir, base + ".fits")),
                         "png_path": os.path.abspath(os.path.join(out_dir, base + ".png"))})
    print(f"  {len(jobs)} spectra to fetch, {len({j['url'] for j in jobs})} unique files")
    for gf in sorted({j["grating"] for j in jobs}):
        print(f"    {gf}: {sum(j['grating'] == gf for j in jobs)}")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['ID']:<7d} {j['grating']}  {j['url']}")
            print(f"           -> {j['fits_path']}")
        print("\nDry run, nothing downloaded.")
        return

    fetched = {}  # url -> local path, so shared spectra are copied rather than re-fetched
    for j in jobs:
        os.makedirs(os.path.dirname(j["fits_path"]), exist_ok=True)
        print(f"\nID {j['ID']}  {j['grating']}")
        print(f"  url   {j['url']}")
        print(f"  fits  {j['fits_path']}")
        try:
            if os.path.exists(j["fits_path"]) and not args.overwrite:
                j["status"] = "exists"
            elif j["url"] in fetched:
                shutil.copy2(fetched[j["url"]], j["fits_path"])
                j["status"] = "copied"
            else:
                curl_download(j["url"], j["fits_path"])
                j["status"] = "downloaded"
            fetched.setdefault(j["url"], j["fits_path"])
        except RuntimeError as err:
            j["status"] = f"failed ({err})"
            j["png_path"] = ""
            print(f"  ERROR {j['status']}")
            continue
        print(f"  {j['status']}")

        if args.no_plots:
            j["png_path"] = ""
            continue
        try:
            plot_spectrum_1d(j["fits_path"], j["png_path"], title=f"{j['ID']}  {j['grating']}")
            print(f"  png   {j['png_path']}")
        except Exception as err:
            print(f"  ERROR plot failed: {err}")
            j["png_path"] = ""

    log = pd.DataFrame(jobs)
    this_run = log.copy()
    log_path = os.path.abspath(os.path.join(args.out_base, "download_log.csv"))
    if os.path.exists(log_path) and args.ids is not None:
        # Partial run, update the matching rows of the existing log.
        old = pd.read_csv(log_path)
        old = old[~old.set_index(["ID", "grating"]).index.isin(
            log.set_index(["ID", "grating"]).index)]
        log = pd.concat([old, log], ignore_index=True).sort_values(["ID", "grating"])
    log.to_csv(log_path, index=False)

    counts = this_run["status"].str.split(" ").str[0].value_counts()
    print("\n--- Summary of this run ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    failed = this_run[this_run["status"].str.startswith("failed")]
    for _, r in failed.iterrows():
        print(f"  failed  {r['ID']}  {r['grating']}  {r['url']}")
    print(f"\nLog written to {log_path}")
    print(f"Spectra under  {os.path.abspath(args.out_base)}")


if __name__ == "__main__":
    main()