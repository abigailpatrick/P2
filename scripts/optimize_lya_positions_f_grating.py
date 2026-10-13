#!/usr/bin/env python
"""
optimize_lya_positions_f_grating.py

Run every false position from false_pos_grating.py through exactly the same
search as the real sources, to build the distribution of the peak Lya S/N
that noise alone produces (Patrick et al., Paper 1, Sect. 3.3.3).

Each position is treated as a source at the real source's redshift. The
aperture shift (+/-0.4 arcsec in 0.1 arcsec steps, Gaussian prior sigma
1 arcsec), the Lya band (-dv_blue/+dv_red km/s about Lya at z_sys, in air)
and the sliding 10 AA S/N are those of optimize_lya_position_grating.py, by
calling its search_offsets function, so the two cannot drift apart.

Noise only
----------
false_pos_grating.py only places positions whose starting aperture contains
no object from Isaac's catalogue (none within --clear, 1.0 arcsec). Here any
trial aperture of the shift that would contain one is blocked, the
counterpart of the nearest-object rule for the real sources: the search keeps
its freedom over blank sky but cannot move onto a galaxy, and near galaxies
it loses offsets as the real sources do near their neighbours. n_blocked is
recorded so the freedom of the two can be compared. --no-object-rule turns
the blocking off and --clear must match false_pos_grating.py.

What changed from the JELS-ID version
  - search_offsets from the real optimiser, so the band (it was +/-35 AA
    about vacuum Lya at z_av), the S/N (its own trapz copy) and the shift
    are identical to the real sources
  - noise only: no aperture, starting or shifted, may contain an Isaac object
  - reads the cube once and caches by aperture pixel (much faster)
  - --ids and --task-index (slurm array, one source per task) replace --all
  - outputs in each source's folder

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, ra, dec, z_sys
  MUSE_catalogs/lya_false_pos_sources.csv           sources with positions
  MUSE_subcubes/<ID>/<ID>_false_positions.csv
  MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits
  jwst_catalogs/cosmos_primer_minerva_production.fits

Output
------
  MUSE_subcubes/<ID>/<ID>_false_optimal_snr.csv
      index, ra, dec (the position), dx_arcsec, dy_arcsec, peak_snr,
      best_center, peak_snr_orig, at_cap, n_blocked

Usage
-----
python optimize_lya_positions_f_grating.py --ids 23070 --max-positions 50   # quick test
python optimize_lya_positions_f_grating.py                                  # all, one node
sbatch --array=0-<N-1> slurm/run_optimize_lya_positions_f_grating.slurm     # all, slurm
"""

import argparse
import os
import time
from functools import partial

import numpy as np
import pandas as pd

import p2_common as pc
from optimize_lya_position_grating import search_offsets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--manifest", default=pc.FALSE_POS_SOURCES)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--task-index", type=int, default=None,
                   help="slurm array index, the Nth ID (0-based) of the sorted sample")
    p.add_argument("--max-positions", type=int, default=None,
                   help="only the first N positions, for testing")
    # These must match optimize_lya_position_grating.py
    p.add_argument("--aperture", type=float, default=0.6)
    p.add_argument("--dx-max", type=float, default=0.4)
    p.add_argument("--dx-step", type=float, default=0.1)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--dv-blue", type=float, default=300.0)
    p.add_argument("--dv-red", type=float, default=1200.0)
    p.add_argument("--isaac", default=pc.ISAAC_CATALOG)
    p.add_argument("--clear", type=float, default=pc.FALSE_POS_CLEAR,
                   help="arcsec, must match false_pos_grating.py")
    p.add_argument("--no-object-rule", action="store_true")
    return p.parse_args()


def clear_test(ra_c, dec_c, nb_ra, nb_dec, clear):
    """An aperture centred at (ra_c, dec_c) contains no Isaac object."""
    return pc.clear_of_objects(ra_c, dec_c, nb_ra, nb_dec, clear)


def false_neighbours(ra, dec, isaac):
    """Isaac objects near a false position (any could enter a shifted aperture)."""
    _, nra, ndec, _ = pc.neighbours(ra, dec, isaac)
    return nra, ndec


def run_source(row, args, grid, isaac):
    sid, z = int(row["ID"]), float(row["z_sys"])
    pos_path = pc.muse_path(sid, "false_pos", args.subcube_dir)
    cube_path = pc.muse_path(sid, "contsub", args.subcube_dir)
    for path in (pos_path, cube_path):
        if not os.path.exists(path):
            print(f"  {sid}  SKIP missing {path}")
            return False
    pos = pd.read_csv(pos_path)
    if args.max_positions:
        pos = pos.iloc[:args.max_positions]

    t0 = time.time()
    cube = pc.load_muse_cube(cube_path)
    wave = cube.wave.coord()
    pixscale = float(np.mean(cube.wcs.get_step(unit="arcsec")))
    radius_pix = args.aperture / pixscale
    band = pc.velocity_band(pc.lya_obs_air(z), args.dv_blue, args.dv_red)
    win = (wave >= band[0] - 50) & (wave <= band[1] + 50)
    w = wave[win]
    f = cube.data.filled(np.nan)[win]
    v = cube.var.filled(np.nan)[win]

    rows = []
    for _, p in pos.iterrows():
        ra, dec = float(p["ra"]), float(p["dec"])
        test = None
        if isaac is not None:
            nb_ra, nb_dec = false_neighbours(ra, dec, isaac)
            test = partial(clear_test, nb_ra=nb_ra, nb_dec=nb_dec, clear=args.clear)
        best, _ = search_offsets(cube, w, f, v, ra, dec, band, grid, radius_pix,
                                 args.prior_scale, args.dx_max, test, check_zero=True)
        rows.append(dict(index=int(p["index"]), ra=ra, dec=dec,
                         dx_arcsec=best["dx"], dy_arcsec=best["dy"],
                         peak_snr=round(best["snr"], 4) if np.isfinite(best["snr"]) else np.nan,
                         best_center=round(best["centre"], 3) if np.isfinite(best["centre"]) else np.nan,
                         peak_snr_orig=round(best["snr_orig"], 4),
                         at_cap=best["at_cap"], n_blocked=best["n_blocked"]))

    out = pc.muse_path(sid, "false_snr", args.subcube_dir)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    snr = d["peak_snr"].dropna()
    print(f"  {sid}  {len(d)} positions, median S/N {snr.median():.2f}, "
          f"98th percentile {np.percentile(snr, 98):.2f}, median blocked offsets "
          f"{d['n_blocked'].median():.0f} of {grid.size ** 2} ({time.time() - t0:.0f} s)"
          f"\n         {out}", flush=True)
    return True


def main():
    args = parse_args()
    print("optimize_lya_positions_f_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    sample = pc.load_sample(args.catalog)
    if args.task_index is not None:
        if not 0 <= args.task_index < len(sample):
            print(f"Task index {args.task_index} is past the {len(sample)} sources, nothing to do")
            return
        sample = sample.iloc[[args.task_index]]
        print(f"Task {args.task_index}, ID {int(sample.iloc[0]['ID'])}")
    elif args.ids:
        sample = pc.load_sample(args.catalog, args.ids)

    man_path = os.path.abspath(args.manifest)
    print(f"Reading manifest    {man_path}")
    man = pd.read_csv(man_path)
    used = set(man.loc[man["used"] == 1, "ID"].astype(int))
    grid = np.round(np.arange(-args.dx_max, args.dx_max + args.dx_step / 2, args.dx_step), 6)
    print(f"Aperture {args.aperture}\" radius, offsets +/-{args.dx_max}\" step {args.dx_step}\", "
          f"prior sigma {args.prior_scale}\", band -{args.dv_blue:g}/+{args.dv_red:g} km/s "
          f"about Lya (air)")
    isaac = None
    if not args.no_object_rule:
        print(f"Nearest-object rule on, reading {os.path.abspath(args.isaac)}")
        isaac = pc.load_isaac_positions(args.isaac)
    else:
        print("Nearest-object rule off")
    print("")

    n_done = 0
    for _, row in sample.iterrows():
        sid = int(row["ID"])
        if sid not in used:
            reason = man.loc[man["ID"] == sid, "reason"]
            print(f"  {sid}  SKIP not used for false positives"
                  + (f" ({reason.iloc[0]})" if len(reason) else ""))
            continue
        n_done += run_source(row, args, grid, isaac)
    print(f"\n{n_done} sources done")


if __name__ == "__main__":
    main()