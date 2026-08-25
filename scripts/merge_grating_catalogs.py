#!/usr/bin/env python
"""Build a merged catalogue keyed on unique JELS ID from the four NIRSpec grating catalogues."""

import numpy as np
import pandas as pd
from astropy.table import Table

cat_dir = "/ceph/cephfs/apatrick/P2/jwst_catalogs"
out_path = f"{cat_dir}/grating_sources_by_JELS_ID.csv"

gratings = {
    "G235H": "JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    "G235M": "JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    "G395H": "JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    "G395M": "JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
}

# Read each catalogue, keeping only ID, z, ra, dec. Drop duplicate IDs within a catalogue.
per_grating = {}
for name, fname in gratings.items():
    df = Table.read(f"{cat_dir}/{fname}").to_pandas()
    df = df[["ID", "z", "ra", "dec"]].drop_duplicates(subset="ID")
    per_grating[name] = df.set_index("ID")

# Master list of unique IDs across all catalogues, plus one ra/dec per ID.
radec = pd.concat([g[["ra", "dec"]] for g in per_grating.values()])
radec = radec[~radec.index.duplicated(keep="first")]

out = pd.DataFrame(index=radec.index)
out.index.name = "ID"

for name, g in per_grating.items():
    out[name] = out.index.isin(g.index).astype(int)
    out[f"z_{name}"] = g["z"].reindex(out.index)

out["ra"] = radec["ra"]
out["dec"] = radec["dec"]

# Order columns as requested.
cols = (["G235H", "G235M", "G395H", "G395M"]
        + [f"z_{g}" for g in ["G235H", "G235M", "G395H", "G395M"]]
        + ["ra", "dec"])
out = out[cols].reset_index()

out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} unique sources to {out_path}")

# --- Add-on: flag sources within the real MUSE data region + sanity plot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from skimage import measure

cube_path = "/ceph/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_VAR_2.fits"
check_png = f"{cat_dir}/muse_footprint_check.png"

with fits.open(cube_path, memmap=True) as hdul:
    hdr = hdul[0].header
    nz = hdr["NAXIS3"]
    wcs = WCS(hdr).celestial
    mid = nz // 2
    plane = hdul[0].data[mid, :, :]

# Real data region: finite and non-zero pixels.
valid = np.isfinite(plane) & (plane != 0)
ny, nx = valid.shape

x, y = wcs.world_to_pixel_values(out["ra"].values, out["dec"].values)
xi = np.round(x).astype(int)
yi = np.round(y).astype(int)

in_grid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
in_muse = np.zeros(len(out), dtype=int)
in_muse[in_grid] = valid[yi[in_grid], xi[in_grid]].astype(int)
out["in_muse"] = in_muse


def plot_footprint_check(plane, valid, x, y, in_muse, mid, out_png):
    """Save a whitelight-style PNG of the slice used, with the valid-region
    outline and source positions overplotted for a visual sanity check."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: the slice itself, scaled like a whitelight image.
    finite = plane[np.isfinite(plane)]
    if finite.size:
        norm = simple_norm(plane, "sqrt", percent=99.0)
        ax.imshow(plane, origin="lower", cmap="Greys_r", norm=norm)

    # Outline of the valid region, traced the same way as the overlay figure.
    for contour in measure.find_contours(valid.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="cyan", lw=0.8)

    # Sources, coloured by their flag. Only plot ones landing on the grid.
    inside = (in_muse == 1)
    outside = (in_muse == 0)
    ax.scatter(x[inside], y[inside], s=8, c="lime", edgecolors="k",
               linewidths=0.2, label=f"in_muse=1 ({inside.sum()})")
    ax.scatter(x[outside], y[outside], s=8, c="red", edgecolors="k",
               linewidths=0.2, label=f"in_muse=0 ({outside.sum()})")

    ax.set_xlim(0, plane.shape[1])
    ax.set_ylim(0, plane.shape[0])
    ax.set_title(f"MUSE slice {mid} with valid-region outline")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved footprint check to {out_png}")


plot_footprint_check(plane, valid, x, y, in_muse, mid, check_png)

out.to_csv(out_path, index=False)
print(f"{out['in_muse'].sum()} sources fall within the MUSE data region")

# Breakdown of why each in_muse=0 source scored zero.
off_grid = ~in_grid
on_grid_invalid = in_grid & (in_muse == 0)
print(f"in_muse=0 total: {(in_muse == 0).sum()}")
print(f"  off the slice grid (not drawn): {off_grid.sum()}")
print(f"  on grid but invalid pixel: {on_grid_invalid.sum()}")
print(f"  of off-grid, NaN position: {(~np.isfinite(x) | ~np.isfinite(y)).sum()}")