
#!/usr/bin/env python
"""
extract_subcubes.py

Cut a square spatial subcube around each source in a CSV, keeping the full
spectral range and the variance extension, and write one FITS per source.

The input CSV must have columns: ra, dec, ID  (ra/dec in degrees).

For each source the central spatial pixel of the output is placed on the
source RA/Dec. The box is square and odd-sized so a single pixel sits at the
centre. A 20 arcsec box at the MUSE 0.2 arcsec pixel scale is 100 pixels, so
the script rounds up to the nearest odd number (101 px, 20.2 arcsec) by
default. Pass --exact-even to force an even box instead.

Output files are named  subcube_JELSID_<ID>.fits.

Usage
-----

python extract_subcubes.py \
    --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
    --cube /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_WITH_VAR_2.fits \
    --out-dir /cephfs/apatrick/P2/MUSE_subcubes \
    --in_muse_check

"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
BOX_ARCSEC = 20.0      # requested square side, arcsec
PIXSCALE = 0.2         # MUSE arcsec / pixel, only used if WCS scale unreadable

# Extension names to try, in order, for the data and variance cubes.
DATA_NAMES = ["DATA", "SCI", "FLUX"]
STAT_NAMES = ["STAT", "VAR", "STATISTIC"]


# ──────────────────────────────────────────────────────────────────────────────
# Cube inspection
# ──────────────────────────────────────────────────────────────────────────────
def find_ext(hdul, names, fallback_index):
    """Return the HDU index matching one of `names`, else a fallback index."""
    upper = {}
    for i, hdu in enumerate(hdul):
        name = (hdu.name or "").upper()
        if name:
            upper[name] = i
    for n in names:
        if n.upper() in upper:
            return upper[n.upper()]
    # Fall back to a positional index if it holds 3D data.
    if fallback_index < len(hdul) and hdul[fallback_index].data is not None \
            and np.ndim(hdul[fallback_index].data) == 3:
        return fallback_index
    return None


def spatial_pixscale_arcsec(wcs2d):
    """Arcsec per pixel from the celestial WCS, averaged over both axes."""
    try:
        scales = np.abs(np.diag(wcs2d.pixel_scale_matrix)) * 3600.0
        good = scales[scales > 0]
        if good.size:
            return float(np.mean(good))
    except Exception:
        pass
    return PIXSCALE


# ──────────────────────────────────────────────────────────────────────────────
# Cropping
# ──────────────────────────────────────────────────────────────────────────────
def box_half_pixels(box_arcsec, pixscale, exact_even):
    """Return (half, size) in pixels. Odd size unless exact_even is set."""
    n = box_arcsec / pixscale
    if exact_even:
        size = int(round(n))
        if size % 2:            # force even
            size += 1
        half = size // 2        # centre sits on a pixel boundary
        return half, size
    size = int(np.ceil(n))
    if size % 2 == 0:           # force odd so one pixel is central
        size += 1
    half = size // 2
    return half, size


def crop_header(header, x0, y0):
    """Copy a cube header and shift the spatial CRPIX for the crop origin.

    x0, y0 are the pixel indices (0-based) of the lower-left corner of the
    spatial box in the parent cube. FITS CRPIX is 1-based, so a source that
    was at CRPIX1 in the parent is at CRPIX1 - x0 in the child.
    """
    h = header.copy()
    if "CRPIX1" in h:
        h["CRPIX1"] = h["CRPIX1"] - x0
    if "CRPIX2" in h:
        h["CRPIX2"] = h["CRPIX2"] - y0
    return h


def extract_one(data_hdu, stat_hdu, wcs2d, ra, dec, half, size):
    """Return (data_box, stat_box, x0, y0, ok, reason).

    The output box is always size x size with the source at the centre pixel.
    Where the box runs off the mosaic, the read is clipped to the cube and the
    overhang is filled with NaN, so a source near the edge still gets a full,
    correctly centred subcube with a blank margin. ok is False only if the
    source centre itself falls outside the cube.

    x0, y0 are the intended box origin (xc - half), which may be negative. They
    are used by crop_header to shift CRPIX, so the source stays centred in the
    output WCS regardless of any padding.
    """
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    x, y = wcs2d.world_to_pixel(coord)
    xc, yc = int(round(float(x))), int(round(float(y)))

    nz, ny, nx = data_hdu.shape
    x0, y0 = xc - half, yc - half        # intended origin, may be negative
    x1, y1 = x0 + size, y0 + size        # intended end, may exceed nx/ny

    if xc < 0 or yc < 0 or xc >= nx or yc >= ny:
        return None, None, x0, y0, False, "centre off cube"

    # Portion of the intended box that actually lies on the cube.
    rx0, rx1 = max(x0, 0), min(x1, nx)
    ry0, ry1 = max(y0, 0), min(y1, ny)

    # Where that portion lands inside the size x size output box.
    ox0, oy0 = rx0 - x0, ry0 - y0
    ox1, oy1 = ox0 + (rx1 - rx0), oy0 + (ry1 - ry0)

    data_box = np.full((nz, size, size), np.nan, dtype=np.float32)
    data_box[:, oy0:oy1, ox0:ox1] = data_hdu.data[:, ry0:ry1, rx0:rx1]

    stat_box = None
    if stat_hdu is not None:
        stat_box = np.full((nz, size, size), np.nan, dtype=np.float32)
        stat_box[:, oy0:oy1, ox0:ox1] = stat_hdu.data[:, ry0:ry1, rx0:rx1]

    padded = (rx0 > x0) or (ry0 > y0) or (rx1 < x1) or (ry1 < y1)
    reason = "edge-padded with NaN" if padded else ""
    return data_box, stat_box, x0, y0, True, reason


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="Input CSV with columns ra, dec, ID.")
    p.add_argument("--cube", required=True,
                   help="Path to the megacube FITS (data + variance).")
    p.add_argument("--out-dir", required=True,
                   help="Directory for the output subcubes.")
    p.add_argument("--box-arcsec", type=float, default=BOX_ARCSEC,
                   help="Square box side in arcsec (default 20).")
    p.add_argument("--exact-even", action="store_true",
                   help="Force an even box (centre on a pixel boundary) instead "
                        "of rounding up to an odd box with a central pixel.")
    p.add_argument("--in_muse_check", action="store_true",
                   help="Only extract sources with in_muse == 1. Requires an "
                        "'in_muse' column in the CSV. Off by default.")
    p.add_argument("--data-ext", default=None,
                   help="Override data extension name or index.")
    p.add_argument("--stat-ext", default=None,
                   help="Override variance extension name or index.")
    p.add_argument("--info", action="store_true",
                   help="Print hdul.info() for the cube and exit.")
    return p.parse_args()


def resolve_ext(arg):
    """Turn a --data-ext / --stat-ext string into an int index or a name."""
    if arg is None:
        return None
    try:
        return int(arg)
    except ValueError:
        return arg


def main():
    args = parse_args()

    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Opening cube (memory-mapped): {args.cube}", flush=True)

    with fits.open(args.cube, memmap=True, ignore_missing_simple=True) as hdul:
        if args.info:
            hdul.info()
            return

        # Resolve data extension.
        if args.data_ext is not None:
            di = resolve_ext(args.data_ext)
            data_idx = di if isinstance(di, int) else find_ext(hdul, [di], 1)
        else:
            data_idx = find_ext(hdul, DATA_NAMES, 1)

        # Resolve variance extension.
        if args.stat_ext is not None:
            si = resolve_ext(args.stat_ext)
            stat_idx = si if isinstance(si, int) else find_ext(hdul, [si], 2)
        else:
            stat_idx = find_ext(hdul, STAT_NAMES, 2)

        if data_idx is None:
            print("ERROR: could not find a 3D data extension. Re-run with "
                  "--info to see the layout, then pass --data-ext.", flush=True)
            sys.exit(1)

        data_hdu = hdul[data_idx]
        stat_hdu = hdul[stat_idx] if stat_idx is not None else None
        print(f"Data extension: [{data_idx}] "
              f"'{data_hdu.name}' shape {data_hdu.shape}", flush=True)
        if stat_hdu is not None:
            print(f"Variance extension: [{stat_idx}] "
                  f"'{stat_hdu.name}' shape {stat_hdu.shape}", flush=True)
        else:
            print("Variance extension: none found, writing data only.",
                  flush=True)

        # Build the 2D celestial WCS from the data header.
        wcs_full = WCS(data_hdu.header)
        wcs2d = wcs_full.celestial

        pixscale = spatial_pixscale_arcsec(wcs2d)
        half, size = box_half_pixels(args.box_arcsec, pixscale, args.exact_even)
        print(f"Pixel scale: {pixscale:.4f} arcsec/pix -> box {size}x{size} px "
              f"({size * pixscale:.2f} arcsec)", flush=True)

        os.makedirs(args.out_dir, exist_ok=True)

        df = pd.read_csv(args.csv)
        for col in ("ra", "dec", "ID"):
            if col not in df.columns:
                print(f"ERROR: CSV missing '{col}' column.", flush=True)
                sys.exit(1)
        print(f"Loaded {len(df)} sources from {args.csv}", flush=True)

        if args.in_muse_check:
            if "in_muse" not in df.columns:
                print("ERROR: --in_muse_check set but CSV has no 'in_muse' "
                      "column.", flush=True)
                sys.exit(1)
            before = len(df)
            df = df[df["in_muse"] == 1].copy()
            print(f"in_muse check on: keeping {len(df)} of {before} sources "
                  f"marked in_muse == 1.", flush=True)

        n_ok, n_skip = 0, 0
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            # Format the ID as an integer where it is a whole number, so a
            # float-typed column (16871.0) still names the file cleanly.
            raw_id = row["ID"]
            try:
                fval = float(raw_id)
                sid = str(int(fval)) if fval.is_integer() else str(raw_id)
            except (TypeError, ValueError):
                sid = str(raw_id)

            ra, dec = float(row["ra"]), float(row["dec"])
            if not (np.isfinite(ra) and np.isfinite(dec)):
                print(f"  [skip] {sid}: non-finite ra/dec", flush=True)
                n_skip += 1
                continue

            print(f"  [{i}/{len(df)}] {sid}: reading box ...", flush=True)
            t0 = time.time()
            data_box, stat_box, x0, y0, ok, reason = extract_one(
                data_hdu, stat_hdu, wcs2d, ra, dec, half, size)
            if not ok:
                print(f"  [skip] {sid}: {reason}", flush=True)
                n_skip += 1
                continue

            # Primary HDU carries the parent primary header for provenance.
            phdu = fits.PrimaryHDU(header=hdul[0].header)
            out = fits.HDUList([phdu])

            dh = crop_header(data_hdu.header, x0, y0)
            out.append(fits.ImageHDU(data=data_box, header=dh,
                                     name=data_hdu.name or "DATA"))
            if stat_box is not None:
                sh = crop_header(stat_hdu.header, x0, y0)
                out.append(fits.ImageHDU(data=stat_box, header=sh,
                                         name=stat_hdu.name or "STAT"))

            out_path = os.path.join(args.out_dir, f"subcube_JELSID_{sid}.fits")
            out.writeto(out_path, overwrite=True)
            n_ok += 1
            dt = time.time() - t0
            note = f", {reason}" if reason else ""
            print(f"  saved: {out_path}  ({dt:.1f}s{note})", flush=True)

    print(f"Done. {n_ok} written, {n_skip} skipped. "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()