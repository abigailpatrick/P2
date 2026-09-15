#!/usr/bin/env python3
"""
Generate random off-source (false-positive) positions in a P2 grating-source
continuum-subtracted MUSE subcube, and plot them on the white-light image.

Adapted from the P1 version:
  - Keyed on the catalogue ID column (cast to int); cube pattern
    source_{ID}_lya_contsub_cube_velocity.fits
  - z_av is the spectroscopic redshift, carried through to the positions CSV
    as z_av (the downstream optimiser maps it to observed Lya +/- a fixed
    half-width).
  - Data/variance read by integer extension index (1, 2) to avoid the mpdaf /
    astropy name collision from the megacube's stray EXTNAME='DATA' primary.
  - Runs from the terminal per source; loops over all IDs when --id is omitted.

For each source it excludes a central box around the true position and an edge
margin, samples N random positions in the remaining area, writes them to a CSV,
and saves a white-light diagnostic plot.

Edge handling:
  - Any source whose collapsed white-light image contains one or more NaN
    pixels is skipped entirely (no CSV, no plot). This removes off-edge and
    partially-off-edge subcubes from the false-positive population. Loosen by
    changing the check in process_source if a stricter/looser rule is wanted.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u


def get_wcs2d(hdr):
    return WCS(hdr).celestial


def collapse_white_light(data, stat=None, method="median"):
    """Collapse the cube along the spectral axis (assumed axis 0 in numpy order,
    i.e. FITS NAXIS3) after masking non-finite and non-positive variance."""
    valid = np.isfinite(data)
    if stat is not None:
        valid &= np.isfinite(stat) & (stat > 0)

    data_masked = np.where(valid, data, np.nan)

    if method == "median":
        white = np.nanmedian(data_masked, axis=0)
    elif method == "sum":
        white = np.nansum(data_masked, axis=0)
    else:
        raise ValueError("method must be 'median' or 'sum'")
    return white


def generate_false_positions_outside_box(
    w2d, nx, ny, source_coord,
    n_positions=100, box_size_arcsec=8.0,
    edge_margin_arcsec=1.0, rng_seed=None, max_iters=20,
):
    """Sample uniformly outside a central box around the source and inside an
    edge margin. Box default is 8 arcsec to match the paper's 8x8 exclusion."""
    rng = np.random.default_rng(rng_seed)

    pix_scales_deg = proj_plane_pixel_scales(w2d)
    arcsec_per_pix_x = pix_scales_deg[0] * 3600.0
    arcsec_per_pix_y = pix_scales_deg[1] * 3600.0

    sx, sy = w2d.world_to_pixel(source_coord)

    half_box_arcsec = box_size_arcsec / 2.0
    hx = half_box_arcsec / arcsec_per_pix_x
    hy = half_box_arcsec / arcsec_per_pix_y

    mx = edge_margin_arcsec / arcsec_per_pix_x
    my = edge_margin_arcsec / arcsec_per_pix_y

    x_lo, x_hi = mx, nx - 1 - mx
    y_lo, y_hi = my, ny - 1 - my
    if not (x_lo < x_hi and y_lo < y_hi):
        raise RuntimeError("Edge margin too large for cube size; no allowed area remains.")

    accepted = []
    for _ in range(max_iters):
        need = n_positions - len(accepted)
        if need <= 0:
            break
        batch = max(5 * need, 200)
        xs = rng.uniform(x_lo, x_hi, size=batch)
        ys = rng.uniform(y_lo, y_hi, size=batch)
        inside = (np.abs(xs - sx) <= hx) & (np.abs(ys - sy) <= hy)
        for x, y in zip(xs[~inside], ys[~inside]):
            accepted.append((x, y))
            if len(accepted) >= n_positions:
                break

    if len(accepted) < n_positions:
        warnings.warn(
            f"Generated {len(accepted)} positions after {max_iters} iterations; "
            f"consider reducing --n or the central box size."
        )

    xs = np.array([p[0] for p in accepted], dtype=float)
    ys = np.array([p[1] for p in accepted], dtype=float)
    coords = w2d.pixel_to_world(xs, ys)
    return coords, accepted, (sx, sy), (hx, hy), (x_lo, x_hi, y_lo, y_hi)


def process_source(sid, ra, dec, z, cube_dir, outdir, args):
    cube_path = os.path.join(cube_dir, f"source_{sid}_lya_contsub_cube_velocity.fits")
    if not os.path.exists(cube_path):
        print(f"[SKIP] Src {sid}: cube not found at {cube_path}")
        return False

    source_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    with fits.open(cube_path, memmap=True) as hdul:
        # Read by integer index to avoid the DATA name collision
        data = hdul[args.data_ext].data
        hdr = hdul[args.data_ext].header
        stat = hdul[args.var_ext].data if len(hdul) > args.var_ext else None

        w2d = get_wcs2d(hdr)

        # --- Edge guard ---
        # Skip if ANY spatial pixel is all-NaN through the spectral axis. This is
        # exactly the condition that triggers numpy's "All-NaN slice encountered"
        # warning inside nanmedian, and it is what leaves NaN pixels in the
        # collapsed white-light image. Checked on the raw data before collapse so
        # it applies regardless of collapse method.
        data_arr = np.asarray(data, dtype=float)
        allnan_map = np.all(~np.isfinite(data_arr), axis=0)
        n_allnan = int(np.count_nonzero(allnan_map))
        if n_allnan > 0:
            frac = n_allnan / allnan_map.size
            print(f"[SKIP] Src {sid}: {n_allnan} all-NaN spectral column(s) "
                  f"({frac:.1%}) would trigger an All-NaN slice; treating as "
                  f"off-edge. No false positions written.")
            return False

        white = collapse_white_light(data, stat=stat, method=args.collapse)
        ny, nx = white.shape

        coords, pix_positions, (sx, sy), (hx, hy), (x_lo, x_hi, y_lo, y_hi) = \
            generate_false_positions_outside_box(
                w2d, nx, ny, source_coord,
                n_positions=args.n, box_size_arcsec=args.box,
                edge_margin_arcsec=args.edge_margin, rng_seed=args.seed,
            )

    csv_out = os.path.join(outdir, f"source_{sid}_false_positions.csv")
    png_out = os.path.join(outdir, f"source_{sid}_false_positions.png")

    pd.DataFrame({
        "ra": coords.ra.deg,
        "dec": coords.dec.deg,
        "z_av": [z] * len(coords),
    }).to_csv(csv_out, index=False)
    print(f"[OK] Src {sid}: wrote {len(coords)} positions to {csv_out}")

    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = np.nanpercentile(white, 5) if np.isfinite(white).any() else 0.0
    vmax = np.nanpercentile(white, 99.5) if np.isfinite(white).any() else 1.0
    ax.imshow(white, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="none")
    ax.set_aspect("equal")
    ax.set_title(f"Src {sid}: false positions outside {args.box}\" box (N={len(coords)})")
    ax.set_xlabel("X [pix]")
    ax.set_ylabel("Y [pix]")

    xs = np.array([p[0] for p in pix_positions])
    ys = np.array([p[1] for p in pix_positions])
    ax.scatter(xs, ys, marker="x", s=25, c="cyan", linewidths=1.0, label="False")
    ax.scatter([sx], [sy], marker="+", s=60, c="red", linewidths=1.5, label="True source")

    ax.plot([sx - hx, sx + hx, sx + hx, sx - hx, sx - hx],
            [sy - hy, sy - hy, sy + hy, sy + hy, sy - hy],
            color="yellow", lw=1.0, alpha=0.8, label=f'Excluded {args.box}"')
    ax.plot([x_lo, x_hi, x_hi, x_lo, x_lo],
            [y_lo, y_lo, y_hi, y_hi, y_lo],
            color="lime", lw=1.0, alpha=0.6, label="Edge margin")

    ax.legend(loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    fig.savefig(png_out, dpi=200)
    plt.close(fig)
    print(f"[OK] Src {sid}: wrote plot {png_out}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate false positions for P2 grating sources"
    )
    parser.add_argument("--csv", required=True,
                        help="Catalogue with ID, ra, dec, z_av columns")
    parser.add_argument("--cube-dir", required=True,
                        help="Directory of continuum-subtracted line cubes")
    parser.add_argument("--id", type=int, default=None,
                        help="Single source ID. If omitted, run all.")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--z-col", default="z_av")

    parser.add_argument("--outdir",
                        default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions",
                        help="Directory for false-position CSVs and plots")

    parser.add_argument("--n", type=int, default=500,
                        help="Number of false positions per source (paper uses 500)")
    parser.add_argument("--box", type=float, default=8.0,
                        help="Central exclusion box size in arcsec (paper uses 8)")
    parser.add_argument("--edge-margin", type=float, default=1.0,
                        help="Forbidden border near cube edges (arcsec)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--collapse", choices=["median", "sum"], default="median")

    parser.add_argument("--data-ext", type=int, default=1)
    parser.add_argument("--var-ext", type=int, default=2)

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    cube_dir = os.path.abspath(args.cube_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("[PATHS]")
    print(f"  Input catalogue     : {csv_path}")
    print(f"  Cube directory      : {cube_dir}")
    print(f"  Output directory    : {outdir}")
    print("")

    df = pd.read_csv(csv_path)
    for col in (args.id_col, "ra", "dec", args.z_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available: {list(df.columns)}"
            )

    if args.id is not None:
        df = df[df[args.id_col].astype("Int64") == args.id]
        if len(df) == 0:
            raise ValueError(f"Requested ID {args.id} not found")

    n_written, n_skipped = 0, 0
    kept_ids = []
    for _, row in df.iterrows():
        sid = int(row[args.id_col])
        wrote = process_source(
            sid, float(row["ra"]), float(row["dec"]), float(row[args.z_col]),
            cube_dir, outdir, args,
        )
        if wrote:
            n_written += 1
            kept_ids.append(sid)
        else:
            n_skipped += 1

    # Write a manifest of the sources that were kept, for the optimiser to read
    kept_path = os.path.join(outdir, "false_positions_kept.csv")
    pd.DataFrame({"ID": kept_ids}).to_csv(kept_path, index=False)

    print("")
    print(f"[DONE] {n_written} sources written, {n_skipped} skipped "
          f"(missing cube or all-NaN spectral columns)")
    print(f"[MANIFEST] Kept IDs written to {kept_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()


"""
run on all good sources
python false_pos_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID_good.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --n 500 --box 8.0
  

run on one source
python false_pos_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID_good.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --id 15479 --n 500 --box 8.0
"""