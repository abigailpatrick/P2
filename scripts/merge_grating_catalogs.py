#!/usr/bin/env python
"""Build a merged catalogue keyed on unique JELS ID from the four NIRSpec grating catalogues."""

import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from skimage import measure
from scipy.ndimage import distance_transform_edt

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

# Mean redshift across the gratings a source appears in, ignoring NaNs.
z_cols = [f"z_{g}" for g in ["G235H", "G235M", "G395H", "G395M"]]
out["z_av"] = out[z_cols].mean(axis=1, skipna=True)

# Order columns as requested.
cols = (["G235H", "G235M", "G395H", "G395M"]
        + [f"z_{g}" for g in ["G235H", "G235M", "G395H", "G395M"]]
        + ["z_av", "ra", "dec"])
out = out[cols].reset_index()


# --- Add-on: flag sources within the real MUSE data region + sanity plot ---


cube_path = "/ceph/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_VAR_2.fits"
check_png = f"/ceph/cephfs/apatrick/P2/field_images/muse_footprint_check.png"

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

# --- Add-on: edge proximity flag ---

# Pad the valid mask with one ring of False so the field boundary counts as
# an edge. Distance is then measured in pixels to the nearest invalid pixel.
padded = np.pad(valid, 1, mode="constant", constant_values=False)
dist_pix_padded = distance_transform_edt(padded)
dist_pix = dist_pix_padded[1:-1, 1:-1]  # strip the padding back off

ARCSEC_PER_PIX = 0.2
EDGE_THRESH_ARCSEC = 6.0

edge = np.zeros(len(out), dtype=int)
on_valid = (in_muse == 1)  # only meaningful for sources on real data
d_arcsec = dist_pix[yi[on_valid], xi[on_valid]] * ARCSEC_PER_PIX
# 0 if >= 6 arcsec from an edge, else the rounded arcsec distance.
edge_vals = np.where(d_arcsec >= EDGE_THRESH_ARCSEC, 0,
                     np.round(d_arcsec).astype(int))
edge[np.where(on_valid)[0]] = edge_vals
out["edge"] = edge
print(f"{(out['edge'] > 0).sum()} sources lie within {EDGE_THRESH_ARCSEC:g} arcsec of a field edge")

# --- Add-on: duplicate position flag ---
grouped = out.groupby(["ra", "dec"], sort=False)
out["duplicate"] = 0
flag = 0
n_dup_rows = 0
for (ra, dec), idx in grouped.groups.items():
    if len(idx) > 1:
        flag += 1
        out.loc[idx, "duplicate"] = flag
        ids = [int(i) for i in out.loc[idx, "ID"].tolist()]
        print(f"Duplicate {flag}  ra={ra}  dec={dec}  ->  IDs {ids}")
        n_dup_rows += len(idx)
if flag == 0:
    print("No duplicate positions found.")
else:
    print(f"Found {flag} duplicated position(s) covering {n_dup_rows} rows.")

# --- Add-on: sodium AO laser gap flag ---
LYA_REST = 1215.67
AO_LO, AO_HI = 5802.0, 5967.0

lya_obs = LYA_REST * (1.0 + out["z_av"].values)
out["AO_block"] = ((lya_obs >= AO_LO) & (lya_obs <= AO_HI)).astype(int)


# --- Summary counts ---
in_field = out["in_muse"] == 1

# Collapse duplicate pairs to one representative row. Non-duplicates (0) all
# kept, each duplicate group (1, 2, ...) reduced to its first occurrence.
is_rep = (out["duplicate"] == 0) | (~out["duplicate"].duplicated(keep="first") & (out["duplicate"] > 0))

n_near_edge = int((out["edge"] > 0).sum())
n_in_field = int(in_field.sum())
n_unique_in_field = int((in_field & is_rep).sum())
n_unique_clear = int((in_field & is_rep & (out["edge"] == 0)).sum())
n_in_gap = int((out["AO_block"] == 1).sum())
n_good = int((in_field & is_rep & (out["edge"] == 0) & (out["AO_block"] == 0)).sum())

print("")
print("--- Summary ---")
print(f"Sources within {EDGE_THRESH_ARCSEC:g} arcsec of a field edge: {n_near_edge}")
print(f"Sources in the MUSE field (total): {n_in_field}")
print(f"Unique sources in the MUSE field (duplicate pair counted once): {n_unique_in_field}")
print(f"Unique sources in the MUSE field more than {EDGE_THRESH_ARCSEC:g} arcsec from an edge: {n_unique_clear}")
print(f"Sources with Lya in the sodium AO gap ({AO_LO:g}-{AO_HI:g} A): {n_in_gap}")
print(f"Unique sources in field, clear of edge, and outside the AO gap: {n_good}")

# Full catalogue with all flags.
out.to_csv(out_path, index=False)
print(f"Wrote updated catalogue with all flags to: {out_path}")

# Good sample: unique, in field, clear of edge, outside AO gap.
good = out[in_field & is_rep & (out["edge"] == 0) & (out["AO_block"] == 0)].copy()
good_path = f"{cat_dir}/grating_sources_by_JELS_ID_good.csv"
good.to_csv(good_path, index=False)
print(f"Wrote {len(good)} good sources to: {good_path}")