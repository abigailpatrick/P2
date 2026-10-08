#!/usr/bin/env python
"""Make one PDF of the LiMe fit that set z_sys for every source in
primer_minerva_in_muse_zsys.csv, six fits per page (3 rows x 2 columns),
for a visual check.

For each source it takes the line (z_sys_line, OIII or Ha) and grating
(z_sys_grating) that build_zsys_catalog.py chose, finds that fit's row in
lime_<line>_fits.csv and places its fit PNG on the page. For [OIII] this is
the 4959,5007 blend (or 5007 alone), for Halpha the single-Gaussian fit that
gave the redshift. Each panel is titled with the ID, line, grating, z_sys and
its error, A/noise, the centre error and the DJA z offset.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv
  jwst_catalogs/lime_OIII_fits.csv, lime_Ha_fits.csv
  lime_fits/<ID>/<line>_fits/<ID>_<G>_<F>_<line>_fit.png

Output
------
  lime_fits/zsys_fit_check.pdf   (change with --out)

Usage
-----
python make_zsys_fit_pdf.py
python make_zsys_fit_pdf.py --sort snr          # lowest S/N first
python make_zsys_fit_pdf.py --out /ceph/cephfs/apatrick/P2/plots/zsys_fit_check.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"
FIG_ROOT = f"{P2}/lime_fits"
GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}
NROW, NCOL = 3, 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zsys-csv", default=f"{CAT_DIR}/primer_minerva_in_muse_zsys.csv")
    p.add_argument("--lime-dir", default=CAT_DIR, help="folder of lime_<line>_fits.csv")
    p.add_argument("--fig-root", default=FIG_ROOT)
    p.add_argument("--out", default=f"{FIG_ROOT}/zsys_fit_check.pdf")
    p.add_argument("--sort", choices=["id", "snr", "line"], default="id",
                   help="id (default), snr (lowest A/noise first) or line (OIII then Ha, by ID)")
    return p.parse_args()


def find_png(fits_tab, sid, line, g, fig_root):
    """PNG path from the fits table, else the standard name."""
    if fits_tab is not None:
        m = fits_tab[(fits_tab["ID"] == sid) & (fits_tab["grating"] == g)]
        if len(m) and isinstance(m.iloc[0].get("fit_png"), str) and os.path.exists(m.iloc[0]["fit_png"]):
            return m.iloc[0]["fit_png"], m.iloc[0]
        row = m.iloc[0] if len(m) else None
    else:
        row = None
    gf = f"{g}_{GRATING_FILTER.get(g, '')}"
    return os.path.join(fig_root, str(sid), f"{line}_fits", f"{sid}_{gf}_{line}_fit.png"), row


def main():
    a = parse_args()
    print("make_zsys_fit_pdf.py")
    print(f"  z_sys catalogue  {os.path.abspath(a.zsys_csv)}")
    zs = pd.read_csv(a.zsys_csv)
    zs["ID"] = zs["ID"].astype(int)

    tabs = {}
    for line in ("OIII", "Ha"):
        path = os.path.abspath(os.path.join(a.lime_dir, f"lime_{line}_fits.csv"))
        print(f"  {line} fits         {path}")
        tabs[line] = pd.read_csv(path) if os.path.exists(path) else None
        if tabs[line] is not None:
            tabs[line]["ID"] = tabs[line]["ID"].astype(int)

    if a.sort == "snr":
        zs = zs.sort_values("z_sys_snr")
    elif a.sort == "line":
        zs = zs.assign(_o=(zs["z_sys_line"] != "OIII")).sort_values(["_o", "ID"])
    else:
        zs = zs.sort_values("ID")

    panels, missing = [], []
    for _, r in zs.iterrows():
        sid, line, g = int(r["ID"]), r["z_sys_line"], r["z_sys_grating"]
        png, frow = find_png(tabs.get(line), sid, line, g, a.fig_root)
        ce = frow["centre_err_AA"] if frow is not None and "centre_err_AA" in frow else np.nan
        meth = frow["method"] if frow is not None and isinstance(frow.get("method"), str) else ""
        title = (f"ID {sid}   {line} {g} {meth}\n"
                 f"z_sys {r['z_sys']:.5f} ± {r['z_sys_err']:.5f}   A/noise {r['z_sys_snr']:.1f}   "
                 f"centre err {ce:.2f} Å   DJA−sys {r['dv_sys_dja_kms']:+.0f} km/s")
        if not os.path.exists(png):
            missing.append((sid, png))
        panels.append((title, png))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    per_page = NROW * NCOL
    n_pages = int(np.ceil(len(panels) / per_page))
    with PdfPages(a.out) as pdf:
        for p in range(n_pages):
            fig, axes = plt.subplots(NROW, NCOL, figsize=(8.27, 11.69))
            for ax, (title, png) in zip(axes.ravel(),
                                        panels[p * per_page:(p + 1) * per_page]):
                if os.path.exists(png):
                    ax.imshow(plt.imread(png))
                else:
                    ax.text(0.5, 0.5, "fit PNG not found", ha="center", va="center",
                            transform=ax.transAxes, color="tab:red")
                ax.set_title(title, fontsize=7)
                ax.axis("off")
            for ax in axes.ravel()[len(panels[p * per_page:(p + 1) * per_page]):]:
                ax.axis("off")
            fig.suptitle(f"z_sys fits, page {p + 1} of {n_pages}", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n  sources          {len(panels)}  "
          f"([OIII] {int((zs['z_sys_line'] == 'OIII').sum())}, Ha {int((zs['z_sys_line'] == 'Ha').sum())})")
    print(f"  pages            {n_pages}")
    if missing:
        print(f"  fit PNG missing for {len(missing)}:")
        for sid, png in missing:
            print(f"    {sid}  {png}")
    print(f"\nWrote {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()