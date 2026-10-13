#!/usr/bin/env python
"""
extract_subcubes.py

Cut a square subcube from the MUSE megacube around each source in the z_sys
sample, centred on Isaac's PRIMER + MINERVA position, keeping the full
spectral range and the variance. Each source gets its own folder in
MUSE_subcubes, named by its Isaac Number (the ID column), which the later Lya
steps also write into.

Replaces the JELS-ID version, which wrote MUSE_subcubes/subcube_JELSID_<ID>.fits
at the JELS positions.

Box
---
The box is square and odd-sized so one pixel sits at the centre. 20 arcsec at
0.2 arcsec per pixel is 100 pixels, rounded up to 101 (20.2 arcsec). The
source is placed on the nearest pixel, so it sits up to 0.1 arcsec from the
centre of that pixel. Its exact position is written to the header (SRC_RA,
SRC_DEC, SRC_X, SRC_Y), so later steps can use it rather than the centre
pixel. Where the box runs off the mosaic the overhang is filled with NaN.

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     columns ID, ra, dec
  .../big_cube/MEGA_CUBE_WITH_VAR_2.fits            flux [1], variance [2]

Outputs
-------
  MUSE_subcubes/<ID>/<ID>_subcube.fits
      [0] primary header of the megacube plus the source keywords
      [1] DATA, flux
      [2] STAT, variance
  MUSE_subcubes/<ID>/<ID>_subcube_white.png
      median image over all wavelengths, with the source and a 0.6 arcsec
      radius circle (the Lya extraction aperture)
  MUSE_subcubes/subcube_log.csv
      one row per source, with the parent pixel position, whether the box was
      padded, the valid fraction of the box and whether the source pixel has
      data

Usage
-----
python extract_subcubes.py --dry-run
python extract_subcubes.py
python extract_subcubes.py --ids 23070 50803
python extract_subcubes.py --skip-existing
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

import p2_common as pc

P2 = pc.P2_ROOT
BIG_CUBE_DIR = "/ceph/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube"
CUBE_PATH = f"{BIG_CUBE_DIR}/MEGA_CUBE_WITH_VAR_2.fits"
CATALOG = f"{P2}/jwst_catalogs/primer_minerva_in_muse_zsys.csv"
OUT_DIR = f"{P2}/MUSE_subcubes"

BOX_ARCSEC = 20.0
APERTURE_ARCSEC = 0.6   # Lya extraction aperture radius, for the PNG only

# The megacube has a stray EXTNAME='DATA' in its primary header, so the
# extensions are taken by index, never by name.
DATA_EXT, STAT_EXT = 1, 2


def subcube_path(out_dir, sid):
    return os.path.join(out_dir, str(sid), f"{sid}_subcube.fits")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=CATALOG, help="CSV with ID, ra, dec")
    p.add_argument("--cube", default=CUBE_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--box-arcsec", type=float, default=BOX_ARCSEC)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave sources that already have a subcube")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="list planned outputs, no cube data cut")
    return p.parse_args()


# ----------------------------------------------------------------------
# Cube geometry
# ----------------------------------------------------------------------
def pixscale_arcsec(wcs2d):
    return float(np.mean(np.abs(np.diag(wcs2d.pixel_scale_matrix))) * 3600.0)


def box_size(box_arcsec, pixscale):
    """Odd box side in pixels and its half width."""
    size = int(np.ceil(box_arcsec / pixscale - 1e-6))
    if size % 2 == 0:
        size += 1
    return size, size // 2


def mid_valid(data, nz):
    """Valid-pixel mask of the middle wavelength slice."""
    plane = np.array(data[nz // 2, :, :])
    return np.isfinite(plane) & (plane != 0)


# ----------------------------------------------------------------------
# Cutting
# ----------------------------------------------------------------------
def place(wcs2d, ra, dec, half, nx, ny):
    """Exact (0-based) pixel position, centre pixel and box limits."""
    x, y = (float(v) for v in wcs2d.world_to_pixel_values(ra, dec))
    xc, yc = int(round(x)), int(round(y))
    x0, y0 = xc - half, yc - half
    on_cube = 0 <= xc < nx and 0 <= yc < ny
    return x, y, xc, yc, x0, y0, on_cube


def cut(cube_data, x0, y0, size):
    """size x size box from x0, y0, NaN where it runs off the cube."""
    nz, ny, nx = cube_data.shape
    rx0, rx1 = max(x0, 0), min(x0 + size, nx)
    ry0, ry1 = max(y0, 0), min(y0 + size, ny)
    box = np.full((nz, size, size), np.nan, dtype=np.float32)
    box[:, ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0] = cube_data[:, ry0:ry1, rx0:rx1]
    padded = (rx0 > x0) or (ry0 > y0) or (rx1 < x0 + size) or (ry1 < y0 + size)
    return box, padded


def shift_header(header, x0, y0, name):
    h = header.copy()
    h["CRPIX1"] = h["CRPIX1"] - x0
    h["CRPIX2"] = h["CRPIX2"] - y0
    h["EXTNAME"] = name
    return h


def source_cards(row, sid, x, y, x0, y0, cube_path, catalog):
    """Keywords added to the primary header. Pixel positions are 1-based."""
    cards = [("P2_ID", sid, "Isaac PRIMER+MINERVA Number"),
             ("SRC_RA", float(row["ra"]), "[deg] source RA (Isaac)"),
             ("SRC_DEC", float(row["dec"]), "[deg] source Dec (Isaac)"),
             ("SRC_X", round(x - x0 + 1, 4), "source x in subcube, 1-based"),
             ("SRC_Y", round(y - y0 + 1, 4), "source y in subcube, 1-based"),
             ("PAR_X0", x0, "subcube x origin in parent, 0-based"),
             ("PAR_Y0", y0, "subcube y origin in parent, 0-based"),
             ("PARENT", os.path.basename(cube_path)),
             ("SRCCAT", os.path.basename(catalog))]
    if "z_sys" in row and pd.notna(row["z_sys"]):
        cards.append(("Z_SYS", float(row["z_sys"]), "systemic redshift"))
    return cards


def white_png(path, box, sid, src_x, src_y, pixscale, z_sys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img = np.nanmedian(box, axis=0)
    size = img.shape[0]
    # Offsets in arcsec from the source, with the pixel edges as the extent.
    ext = np.array([-0.5 - src_x, size - 0.5 - src_x,
                    -0.5 - src_y, size - 0.5 - src_y]) * pixscale
    fin = img[np.isfinite(img)]
    vmin, vmax = (np.percentile(fin, [1, 99.5]) if fin.size else (0, 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax, extent=ext)
    ax.add_patch(plt.Circle((0, 0), APERTURE_ARCSEC, fill=False, color="tab:red", lw=1.2))
    ax.plot(0, 0, "+", color="tab:red", ms=8)
    ax.set_xlabel(r"$\Delta x$ [arcsec]")
    ax.set_ylabel(r"$\Delta y$ [arcsec]")
    title = f"ID {sid}, MUSE white light"
    if np.isfinite(z_sys):
        title += f", z_sys = {z_sys:.4f}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    a = parse_args()
    print("extract_subcubes.py")
    print(f"Start {time.strftime('%Y-%m-%d %H:%M:%S')}")

    cat_path = os.path.abspath(a.catalog)
    print(f"Reading catalogue        {cat_path}")
    df = pd.read_csv(a.catalog)
    missing = [c for c in ("ID", "ra", "dec") if c not in df.columns]
    if missing:
        sys.exit(f"ERROR catalogue has no {missing} column")
    df["ID"] = df["ID"].astype(int)
    if a.ids:
        absent = sorted(set(a.ids) - set(df["ID"]))
        if absent:
            print(f"  WARNING IDs not in the catalogue: {absent}")
        df = df[df["ID"].isin(a.ids)]
    df = df.sort_values("ID").reset_index(drop=True)
    print(f"  {len(df)} sources")

    out_dir = os.path.abspath(a.out_dir)
    log_path = os.path.join(out_dir, "subcube_log.csv")
    print(f"Opening cube             {os.path.abspath(a.cube)}")

    with fits.open(a.cube, memmap=True, ignore_missing_simple=True) as hdul:
        dhdu, shdu = hdul[DATA_EXT], hdul[STAT_EXT]
        if dhdu.data.ndim != 3 or shdu.data.shape != dhdu.data.shape:
            sys.exit(f"ERROR expected matching 3D cubes in extensions {DATA_EXT} "
                     f"and {STAT_EXT}, got {dhdu.data.shape} and {shdu.data.shape}")
        nz, ny, nx = dhdu.data.shape
        wcs2d = WCS(dhdu.header).celestial
        pixscale = pixscale_arcsec(wcs2d)
        size, half = box_size(a.box_arcsec, pixscale)
        print(f"  data [{DATA_EXT}] and variance [{STAT_EXT}], shape {dhdu.data.shape}")
        print(f"  {pixscale:.4f} arcsec/pix, box {size} x {size} px "
              f"({size * pixscale:.1f} arcsec)")

        # Valid pixels of the middle slice, to flag sources with no data.
        valid = mid_valid(dhdu.data, nz)

        primary = hdul[0].header.copy()
        primary.pop("EXTNAME", None)

        rows = []
        n_new = n_skip = 0
        print(f"\nCutting subcubes into    {out_dir}/<ID>/")
        for i, row in df.iterrows():
            sid = int(row["ID"])
            ra, dec = float(row["ra"]), float(row["dec"])
            z_sys = float(row["z_sys"]) if "z_sys" in row and pd.notna(row["z_sys"]) else np.nan
            fits_path = subcube_path(out_dir, sid)
            png_path = fits_path.replace(".fits", "_white.png")
            tag = f"  [{i + 1}/{len(df)}] {sid}"

            x, y, xc, yc, x0, y0, on_cube = place(wcs2d, ra, dec, half, nx, ny)
            centre_valid = bool(on_cube and valid[yc, xc])
            rec = {"ID": sid, "ra": ra, "dec": dec, "x_parent": round(x, 3),
                   "y_parent": round(y, 3), "centre_valid": int(centre_valid),
                   "padded": np.nan, "frac_valid_box": np.nan, "subcube": fits_path}

            if not on_cube:
                print(f"{tag}  SKIP centre off the cube")
                rec["subcube"] = ""
                rows.append(rec)
                continue
            flag = "" if centre_valid else "  WARNING no data at source pixel (middle slice)"

            if a.skip_existing and os.path.exists(fits_path):
                print(f"{tag}  exists, skipped")
                n_skip += 1
                continue
            if a.dry_run:
                print(f"{tag}  would write {fits_path}{flag}")
                rows.append(rec)
                continue

            t0 = time.time()
            data_box, padded = cut(dhdu.data, x0, y0, size)
            stat_box, _ = cut(shdu.data, x0, y0, size)
            vmid = data_box[nz // 2]
            rec["padded"] = int(padded)
            rec["frac_valid_box"] = round(float((np.isfinite(vmid) & (vmid != 0)).mean()), 3)

            ph = primary.copy()
            for card in source_cards(row, sid, x, y, x0, y0, a.cube, a.catalog):
                ph[card[0]] = card[1:] if len(card) > 2 else card[1]
            out = fits.HDUList([
                fits.PrimaryHDU(header=ph),
                fits.ImageHDU(data_box, header=shift_header(dhdu.header, x0, y0, "DATA")),
                fits.ImageHDU(stat_box, header=shift_header(shdu.header, x0, y0, "STAT")),
            ])
            os.makedirs(os.path.dirname(fits_path), exist_ok=True)
            out.writeto(fits_path, overwrite=True)
            msg = f"{tag}  {fits_path}"
            if not a.no_png:
                white_png(png_path, data_box, sid, x - x0, y - y0, pixscale, z_sys)
                msg += f"\n             {png_path}"
            pad = ", NaN-padded" if padded else ""
            print(f"{msg}\n             ({time.time() - t0:.0f} s{pad}){flag}", flush=True)
            rows.append(rec)
            n_new += 1

    log = pd.DataFrame(rows)
    if len(log):
        log["padded"] = log["padded"].astype("Int64")
    print("\n--- Summary ---")
    print(f"  sources                   {len(df)}")
    print(f"  {'to write' if a.dry_run else 'written':<25} {len(log[log['subcube'] != ''])}")
    print(f"  skipped (existing)        {n_skip}")
    print(f"  off the cube              {int((log['subcube'] == '').sum()) if len(log) else 0}")
    print(f"  no data at source pixel   {int((log['centre_valid'] == 0).sum()) if len(log) else 0}")
    if not a.dry_run and len(log):
        print(f"  NaN-padded boxes          {int((log['padded'] == 1).sum())}")

    # Numbered folders in MUSE_subcubes that are not in the catalogue are
    # left over from an earlier sample.
    if os.path.isdir(out_dir) and not a.ids:
        ids = set(pd.read_csv(a.catalog)["ID"].astype(int))
        stale = sorted(int(d) for d in os.listdir(out_dir)
                       if d.isdigit() and os.path.isdir(os.path.join(out_dir, d))
                       and int(d) not in ids)
        if stale:
            print(f"\n  Folders not in the catalogue (stale): {stale}")

    if a.dry_run:
        print("\nDry run, nothing written.")
        return

    # A full run replaces the log. --ids and --skip-existing patch it.
    if (a.ids or a.skip_existing) and os.path.exists(log_path) and len(log):
        old = pd.read_csv(log_path)
        log = pd.concat([old[~old["ID"].isin(log["ID"])], log]).sort_values("ID")
    if len(log):
        os.makedirs(out_dir, exist_ok=True)
        log.to_csv(log_path, index=False)
        print(f"\nWrote\n  {log_path}")
    print(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()