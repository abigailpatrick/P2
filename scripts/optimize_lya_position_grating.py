#!/usr/bin/env python
"""
optimize_lya_position_grating.py

Optimise the spatial position of the Lya extraction aperture in the
continuum-subtracted MUSE subcubes, for the PRIMER + MINERVA z_sys sample.

The method is unchanged from the JELS-ID version. The 0.6 arcsec radius
aperture is moved over a grid of offsets (default +/-0.4 arcsec in 0.1 arcsec
steps) about Isaac's position. At each offset the aperture spectrum is
measured and its peak sliding-window S/N (10 AA window, 1 AA step, smoothed
with a sigma = 2 step Gaussian, p2_common.best_centre) is found inside the Lya band, -dv_blue to
+dv_red km/s about Lya at z_sys (default -300/+1200). The score is that S/N
times a Gaussian prior on the offset (sigma --prior-scale, default 1 arcsec),
and the offset with the highest score is kept. Offsets snap to the nearest
pixel (0.2 arcsec), so several grid points share one aperture position.

Nearest-object rule (new, not in Paper 1)
-----------------------------------------
A trial aperture centre (the centre of the pixel it snaps to) is allowed only
if it is at least as close to the target as to any other object in Isaac's
catalogue (p2_common.closer_to_target). The shift can then move the aperture
towards Lya offset from the galaxy, but not onto a catalogued neighbour, so
close pairs no longer converge on one peak. The zero offset is always
allowed. The false positives apply the same rule with the random position as
the target. --no-object-rule turns it off.

at_cap is 1 when the chosen offset lies on the edge of the grid (|dx| or |dy|
= dx_max), meaning the S/N was still rising at the limit. Check those by eye.

What changed
  - reads jwst_catalogs/primer_minerva_in_muse_zsys.csv, keyed on Isaac's ID,
    with z_sys for every source (no z_dja fallback)
  - reads MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits
  - Lya at its air wavelength, as in lya_local_contsub.py
  - the S/N at the original position is kept alongside the optimised one
  - a figure per source compares the original and moved positions
  - the nearest-object rule and the at_cap flag
  - a full run lists the sources whose offset changed since the last run,
    which are the ones needing new MC errors

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits     lya_local_contsub.py
  jwst_catalogs/cosmos_primer_minerva_production.fits   Isaac, for the rule

Outputs
-------
  MUSE_catalogs/lya_optimal_offsets.csv
      ID, dx_arcsec, dy_arcsec, shift_arcsec, peak_snr (moved),
      peak_snr_orig (original position), snr_gain, at_cap, n_blocked (grid
      offsets excluded by the rule), nearest_id, nearest_sep_arcsec,
      lya_wave_air, band edges
  MUSE_subcubes/<ID>/<ID>_lya_position.png
      Lya narrowband image (the band summed) with the original and moved
      apertures and nearby Isaac objects, the S/N over the offset grid with
      the excluded offsets greyed out, and the two aperture spectra

A full run replaces the CSV. --ids patches only those rows.

Usage
-----
python optimize_lya_position_grating.py
python optimize_lya_position_grating.py --ids 23070 50803
python optimize_lya_position_grating.py --dx-max 0.4 --dx-step 0.1 --prior-scale 1.0
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p2_common as pc

OUTFILE = f"{pc.MUSE_CAT_DIR}/lya_optimal_offsets.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outfile", default=OUTFILE)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--aperture", type=float, default=0.6, help="radius, arcsec")
    p.add_argument("--dx-max", type=float, default=0.4, help="arcsec")
    p.add_argument("--dx-step", type=float, default=0.1, help="arcsec")
    p.add_argument("--prior-scale", type=float, default=1.0,
                   help="sigma of the Gaussian prior on the offset, arcsec "
                        "(0 for no prior)")
    p.add_argument("--dv-blue", type=float, default=300.0)
    p.add_argument("--dv-red", type=float, default=1200.0)
    p.add_argument("--isaac", default=pc.ISAAC_CATALOG)
    p.add_argument("--no-object-rule", action="store_true",
                   help="allow the shift onto other Isaac objects (Paper 1)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# Geometry and S/N, as in the JELS-ID version
# ----------------------------------------------------------------------
def shift_radec(ra, dec, dx_arcsec, dy_arcsec):
    return ra + dx_arcsec / 3600.0 / np.cos(np.deg2rad(dec)), dec + dy_arcsec / 3600.0


def coords_to_pixel(cube, ra, dec):
    y, x = cube.wcs.sky2pix([[dec, ra]], nearest=True, unit="deg")[0]
    return int(x), int(y)


def aperture_mask(ny, nx, x0, y0, radius_pix):
    yy, xx = np.indices((ny, nx))
    return np.hypot(xx - x0, yy - y0) <= radius_pix


def aperture_spectrum(flux, var, ap):
    """NaN-safe aperture sum of flux and variance cubes."""
    return np.nansum(flux[:, ap], axis=1), np.nansum(var[:, ap], axis=1)


def search_offsets(cube, w, f, v, ra0, dec0, band, grid, radius_pix, prior_scale,
                   dx_max, allowed=None, check_zero=False):
    """The aperture-shift search, shared by the real sources and the false
    positives (optimize_lya_positions_f_grating.py) so both are identical.

    cube supplies the WCS. w, f, v are the wavelength, flux and variance cubes
    cut to band +/- 50 AA. allowed(ra, dec) says whether an aperture centred
    on the centre of the pixel it snaps to may be used (None allows all). The
    real sources pass the nearest-object rule and never test the zero offset.
    The false positions pass the clear-of-objects test with check_zero, so
    their starting aperture is tested too. If no offset is allowed, snr and
    the offsets are NaN.

    Returns (best, snr_grid). best holds dx, dy, snr, centre (best_center at
    the chosen offset), pix, snr_orig, pix0, shift, at_cap and n_blocked.
    """
    ny, nx = f.shape[1:]
    snr_grid = np.full((grid.size, grid.size), np.nan)
    cache = {}

    def measure(pix):
        if pix not in cache:
            fl, vr = aperture_spectrum(f, v, aperture_mask(ny, nx, *pix, radius_pix))
            cache[pix] = pc.best_centre(w, fl, vr, band)
        return cache[pix]

    best = dict(score=-np.inf)
    n_blocked = 0
    for j, dy in enumerate(grid):
        for i, dx in enumerate(grid):
            pix = coords_to_pixel(cube, *shift_radec(ra0, dec0, dx, dy))
            if allowed is not None and (check_zero or (dx, dy) != (0, 0)):
                dec_c, ra_c = cube.wcs.pix2sky([[pix[1], pix[0]]], unit="deg")[0]
                if not allowed(ra_c, dec_c):
                    n_blocked += 1
                    continue
            centre, snr = measure(pix)
            snr_grid[j, i] = snr
            prior = np.exp(-(dx ** 2 + dy ** 2) / (2 * prior_scale ** 2)) \
                if prior_scale > 0 else 1.0
            if snr * prior > best["score"]:
                best = dict(score=snr * prior, dx=float(dx) + 0.0, dy=float(dy) + 0.0,
                            snr=snr, centre=centre, pix=pix)

    if not np.isfinite(best["score"]):
        best = dict(dx=np.nan, dy=np.nan, snr=np.nan, centre=np.nan, pix=None)
    pix0 = coords_to_pixel(cube, ra0, dec0)
    best["pix0"] = pix0
    best["snr_orig"] = measure(pix0)[1]
    best["shift"] = float(np.hypot(best["dx"], best["dy"]))
    best["at_cap"] = int(np.nan_to_num(max(abs(best["dx"]), abs(best["dy"]))) >= dx_max - 1e-6)
    best["n_blocked"] = n_blocked
    return best, snr_grid


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot_position(path, sid, z, wave, flux, band, lya_wave, src_xy, pix0, pix1,
                  radius_pix, pixscale, grid, snr_grid, best, spec0, spec1, flip,
                  nb_pix=(), nb_ids=()):
    dw = np.median(np.diff(wave))
    inb = (wave >= band[0]) & (wave <= band[1])
    nb = np.nansum(flux[inb], axis=0) * dw     # 1e-20 erg/s/cm2 per spaxel

    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.85, 1.5])

    # 1. Narrowband image, 3 arcsec either side of the source.
    ax = fig.add_subplot(gs[0])
    half = 3.0 / pixscale
    xs, ys = src_xy
    x0, x1 = int(max(0, np.floor(xs - half))), int(min(nb.shape[1], np.ceil(xs + half) + 1))
    y0, y1 = int(max(0, np.floor(ys - half))), int(min(nb.shape[0], np.ceil(ys + half) + 1))
    cut = nb[y0:y1, x0:x1]
    fin = cut[np.isfinite(cut)]
    lim = np.percentile(np.abs(fin), 99) if fin.size else 1.0
    ext = (np.array([x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5])
           - np.array([xs, xs, ys, ys])) * pixscale
    ax.imshow(cut, origin="lower", cmap="RdGy_r", vmin=-lim, vmax=lim, extent=ext)
    for (px, py), col, lab in ((pix0, "tab:blue", "original"), (pix1, "tab:red", "moved")):
        ax.add_patch(plt.Circle(((px - xs) * pixscale, (py - ys) * pixscale),
                                radius_pix * pixscale, fill=False, color=col, lw=1.6,
                                ls="--" if lab == "original" else "-", label=lab))
    ax.plot(0, 0, "x", color="k", ms=6, label="Isaac position")
    for (px, py), nid in zip(nb_pix, nb_ids):
        ox, oy = (px - xs) * pixscale, (py - ys) * pixscale
        if abs(ox) < 3 and abs(oy) < 3:
            ax.plot(ox, oy, "o", mfc="none", mec="tab:purple", ms=7, mew=1.3)
            ax.annotate(str(nid), (ox, oy), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="tab:purple")
    ax.set_xlabel(r"$\Delta x$ [arcsec] (cube pixel axes)")
    ax.set_ylabel(r"$\Delta y$ [arcsec]")
    ax.set_title(f"Lya NB, {band[0]:.0f}-{band[1]:.0f} " + r"$\AA$", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    # 2. S/N over the offset grid.
    ax = fig.add_subplot(gs[1])
    step = grid[1] - grid[0] if grid.size > 1 else 0.1
    e = [grid[0] - step / 2, grid[-1] + step / 2] * 2
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")
    im = ax.imshow(np.ma.masked_invalid(snr_grid), origin="lower", cmap=cmap, extent=e)
    ax.plot(0, 0, "o", mfc="none", mec="tab:blue", ms=9, mew=1.5)
    ax.plot(best["dx"], best["dy"], "+", color="tab:red", ms=12, mew=2)
    fig.colorbar(im, ax=ax, fraction=0.046, label="peak S/N in band")
    # Orient the grid like the image, so a shift to the right in the image is
    # to the right here too. dx is the RA offset (east positive).
    if flip[0]:
        ax.invert_xaxis()
    if flip[1]:
        ax.invert_yaxis()
    ax.set_xlabel("dx, RA offset (east +) [arcsec]")
    ax.set_ylabel("dy, Dec offset (north +) [arcsec]")
    ax.set_title("S/N over the offset grid" + (" (grey = nearer a neighbour)"
                 if np.isnan(snr_grid).any() else ""), fontsize=9)

    # 3. Aperture spectra at the two positions.
    ax = fig.add_subplot(gs[2])
    zoom = (wave > band[0] - 60) & (wave < band[1] + 60)
    ax.step(wave[zoom], spec0[zoom], where="mid", color="tab:blue", lw=0.9,
            label=f"original, S/N {best['snr_orig']:.2f}")
    ax.step(wave[zoom], spec1[zoom], where="mid", color="tab:red", lw=0.9,
            label=f"moved, S/N {best['snr']:.2f}")
    ax.axvspan(*band, color="0.5", alpha=0.12)
    ax.axvline(lya_wave, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel(r"Wavelength (air) [$\AA$]")
    ax.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.legend(fontsize=8)

    fig.suptitle(f"ID {sid}, z_sys = {z:.4f}, shift dx = {best['dx']:+.1f}\", "
                 f"dy = {best['dy']:+.1f}\" ({best['shift']:.2f}\"), "
                 f"S/N {best['snr_orig']:.2f} to {best['snr']:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def optimise(row, args, grid, isaac):
    sid, z = int(row["ID"]), float(row["z_sys"])
    ra0, dec0 = float(row["ra"]), float(row["dec"])
    cube_path = pc.muse_path(sid, "contsub", args.subcube_dir)
    if not os.path.exists(cube_path):
        print(f"  {sid}  SKIP no contsub cube at {cube_path}")
        return None

    cube = pc.load_muse_cube(cube_path)
    wave = cube.wave.coord()
    flux = cube.data.filled(np.nan)
    var = cube.var.filled(np.nan)
    ny, nx = flux.shape[1:]
    pixscale = float(np.mean(cube.wcs.get_step(unit="arcsec")))
    radius_pix = args.aperture / pixscale

    lya_wave = pc.lya_obs_air(z)
    band = pc.velocity_band(lya_wave, args.dv_blue, args.dv_red)
    win = (wave >= band[0] - 50) & (wave <= band[1] + 50)   # as before
    w, f, v = wave[win], flux[win], var[win]

    # Isaac objects that could compete for the aperture
    if isaac is not None:
        nb_ids, nb_ra, nb_dec, nb_sep = pc.neighbours(ra0, dec0, isaac, exclude_id=sid)
    else:
        nb_ids, nb_ra, nb_dec, nb_sep = (np.array([], int),) + (np.array([]),) * 3

    def nearest_object_rule(ra_c, dec_c):
        return pc.closer_to_target(ra_c, dec_c, ra0, dec0, nb_ra, nb_dec)
    allowed = nearest_object_rule if isaac is not None else None
    best, snr_grid = search_offsets(
        cube, w, f, v, ra0, dec0, band, grid, radius_pix, args.prior_scale, args.dx_max,
        allowed)
    pix0, at_cap, n_blocked = best["pix0"], best["at_cap"], best["n_blocked"]

    out = dict(ID=sid, dx_arcsec=round(best["dx"], 3), dy_arcsec=round(best["dy"], 3),
               shift_arcsec=round(best["shift"], 3),
               peak_snr=round(best["snr"], 3), peak_snr_orig=round(best["snr_orig"], 3),
               snr_gain=round(best["snr"] - best["snr_orig"], 3),
               at_cap=at_cap, n_blocked=n_blocked,
               nearest_id=int(nb_ids[0]) if len(nb_ids) else -1,
               nearest_sep_arcsec=round(float(nb_sep[0]), 3) if len(nb_ids) else np.nan,
               x_pix_orig=pix0[0], y_pix_orig=pix0[1],
               x_pix=best["pix"][0], y_pix=best["pix"][1],
               lya_wave_air=round(lya_wave, 3),
               band_min=round(band[0], 3), band_max=round(band[1], 3), z_sys=z)
    msg = (f"  {sid}  dx {best['dx']:+.1f}\" dy {best['dy']:+.1f}\"  "
           f"S/N {best['snr_orig']:.2f} (original) to {best['snr']:.2f} (moved)"
           + (", at cap" if at_cap else "")
           + (f", {n_blocked} offsets blocked by {', '.join(map(str, nb_ids))}"
              if n_blocked else ""))

    if not args.no_plots:
        hdr = cube.primary_header
        src_xy = ((hdr["SRC_X"] - 1, hdr["SRC_Y"] - 1) if "SRC_X" in hdr
                  else tuple(cube.wcs.sky2pix([[dec0, ra0]], unit="deg")[0][::-1]))
        spec0, _ = aperture_spectrum(flux, var, aperture_mask(ny, nx, *pix0, radius_pix))
        spec1, _ = aperture_spectrum(flux, var, aperture_mask(ny, nx, *best["pix"], radius_pix))
        # Does +dx (east) or +dy (north) run towards lower pixel numbers?
        ref = np.array(cube.wcs.sky2pix([[dec0, ra0]], unit="deg")[0])
        e = np.array(cube.wcs.sky2pix([[dec0, shift_radec(ra0, dec0, 1, 0)[0]]], unit="deg")[0])
        n = np.array(cube.wcs.sky2pix([[dec0 + 1 / 3600, ra0]], unit="deg")[0])
        flip = ((e - ref)[1] < 0, (n - ref)[0] < 0)
        nb_pix = [tuple(cube.wcs.sky2pix([[d_, r_]], unit="deg")[0][::-1])
                  for r_, d_ in zip(nb_ra, nb_dec)]
        png = pc.muse_path(sid, "position_png", args.subcube_dir)
        plot_position(png, sid, z, wave, flux, band, lya_wave, src_xy, pix0, best["pix"],
                      radius_pix, pixscale, grid, snr_grid, best, spec0, spec1, flip,
                      nb_pix, nb_ids)
        msg += f"\n         {png}"
    print(msg, flush=True)
    return out


def main():
    args = parse_args()
    print("optimize_lya_position_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")
    grid = np.round(np.arange(-args.dx_max, args.dx_max + args.dx_step / 2, args.dx_step), 6)
    print(f"Aperture {args.aperture}\" radius, offsets +/-{args.dx_max}\" step "
          f"{args.dx_step}\", prior sigma {args.prior_scale}\", "
          f"band -{args.dv_blue:g}/+{args.dv_red:g} km/s about Lya (air)")
    isaac = None
    if not args.no_object_rule:
        print(f"Nearest-object rule on, reading {os.path.abspath(args.isaac)}")
        isaac = pc.load_isaac_positions(args.isaac)
    else:
        print("Nearest-object rule off")
    print("")

    rows = [r for r in (optimise(row, args, grid, isaac) for _, row in df.iterrows()) if r]
    new = pd.DataFrame(rows)

    out_path = os.path.abspath(args.outfile)
    changed = []
    had_prev = os.path.exists(out_path)
    if had_prev and len(new):
        prev = pd.read_csv(out_path).set_index("ID")
        for r in rows:
            if r["ID"] not in prev.index or \
                    (prev.loc[r["ID"], "x_pix"], prev.loc[r["ID"], "y_pix"]) != (r["x_pix"], r["y_pix"]):
                changed.append(r["ID"])
    if args.ids and os.path.exists(out_path) and len(new):
        old = pd.read_csv(out_path)
        new = pd.concat([old[~old["ID"].isin(new["ID"])], new]).sort_values("ID")
    if len(new):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        new.to_csv(out_path, index=False)

    print("\n--- Summary ---")
    print(f"  optimised          {len(rows)}")
    print(f"  no contsub cube    {len(df) - len(rows)}")
    if rows:
        r = pd.DataFrame(rows)
        print(f"  moved at all       {int((r['shift_arcsec'] > 0).sum())}")
        print(f"  at the cap         {int(r['at_cap'].sum())}  {r.loc[r['at_cap'] == 1, 'ID'].tolist()}")
        print(f"  offsets blocked    {int((r['n_blocked'] > 0).sum())} sources")
        print(f"  median S/N gain    {r['snr_gain'].median():.2f}")
        if had_prev:
            print(f"  aperture changed since the last run: {len(changed)}")
            if changed:
                print(f"    {' '.join(map(str, changed))}")
        print(f"\nWrote\n  {out_path}")


if __name__ == "__main__":
    main()