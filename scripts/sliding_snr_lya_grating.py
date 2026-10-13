#!/usr/bin/env python
"""
sliding_snr_lya_grating.py

Measure the Lya detection S/N of each aperture spectrum, Sigma_Lya of
Patrick et al. (Paper 1, Sect. 3.3.2 and 3.3.3). A 10 AA window is slid along
the spectrum in 1 AA steps and the integrated flux in the window is divided by
its propagated error. The S/N curve is smoothed with a Gaussian of sigma = 2
steps, and the peak inside the Lya band (-dv_blue to +dv_red km/s about Lya at
z_sys, stored in the spectrum file) gives best_center and peak_snr_slide.
fit_lya_properties_grating.py copies peak_snr_slide into lya_snr, the S/N that
is compared with the false-positive thresholds.

The step and smoothing (p2_common.SLIDE_STEP, SLIDE_SMOOTH) are those of the
JELS-ID scripts. Paper 1 describes steps of one spectral pixel (1.25 AA) and
does not mention smoothing, so check which the Paper 1 numbers used.

What changed from the JELS-ID version
  - reads MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz, keyed on Isaac's ID, with
    z_sys for every source (no z_dja fallback)
  - the S/N is p2_common.sliding_snr, shared with the position optimiser and
    the false positives. The flux is now a plain sum over the window, to
    match its error (it was trapz before, which drops half of each end
    channel)
  - the pseudo-narrowband image has moved to the fit step

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, z_sys
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz          ap_extract_specs_grating.py

Outputs
-------
  MUSE_catalogs/lya_sliding_snr.csv
      ID, best_center (air AA), peak_snr_slide, lya_obs (air AA), band_min,
      band_max, slide_window
  MUSE_subcubes/<ID>/<ID>_lya_sliding_snr.png
      the S/N curve about Lya, as in the JELS-ID version (the pseudo-NB is
      now made by fit_lya_properties_grating.py)

A full run replaces the CSV. --ids patches only those rows.

Usage
-----
python sliding_snr_lya_grating.py
python sliding_snr_lya_grating.py --ids 23070 50803
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p2_common as pc

OUTFILE = f"{pc.MUSE_CAT_DIR}/lya_sliding_snr.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--outfile", default=OUTFILE)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--window", type=float, default=pc.SLIDE_WINDOW, help="AA")
    p.add_argument("--step", type=float, default=pc.SLIDE_STEP, help="AA")
    p.add_argument("--plot-halfwidth", type=float, default=250.0,
                   help="plotted half-width about Lya at z_sys, AA")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def plot_sliding(path, sid, ra, dec, aperture, centres, snr, sm, best, peak,
                 lya_wave, band, window, halfwidth):
    """The JELS-ID sliding S/N figure, without the pseudo-NB (that is now made
    over the fitted line window by fit_lya_properties_grating.py)."""
    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.step(centres, snr, where="mid", color="black", lw=1.0, alpha=0.6)
    ax.plot(centres, sm, color="crimson", lw=1.8)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    if np.isfinite(best):
        ax.axvspan(best - window / 2, best + window / 2, color="lightblue", alpha=0.3,
                   label=f"peak window ({window:g} " + r"$\AA$)")
    ax.axvline(band[0], color="royalblue", ls=":", lw=1.5)
    ax.axvline(band[1], color="royalblue", ls=":", lw=1.5)
    ax.axvline(lya_wave, color="orange", ls="--", lw=2.0, label=r"Ly$\alpha$ ($z_{\rm sys}$)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(lya_wave - halfwidth, lya_wave + halfwidth)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("S/N")
    ax.text(0.05, 0.95, f"Peak S/N = {peak:.2f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"Source {sid} | RA={ra:.5f} | Dec={dec:.5f} | Aperture {aperture:g}\"")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    print("sliding_snr_lya_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    df = pc.load_sample(args.catalog, args.ids)
    print(f"  {len(df)} sources")
    print(f"Window {args.window:g} AA, step {args.step:g} AA, "
          f"smoothing sigma {pc.SLIDE_SMOOTH:g} steps\n")

    rows, n_missing = [], 0
    for _, row in df.iterrows():
        sid = int(row["ID"])
        spec_path = pc.muse_path(sid, "spectrum", args.subcube_dir)
        if not os.path.exists(spec_path):
            print(f"  {sid}  SKIP no spectrum at {spec_path}")
            n_missing += 1
            continue
        d = np.load(spec_path)
        wave, flux, var = d["wave"], d["flux"], d["var"]
        lya_wave = float(d["lya_obs"])
        band = (float(d["band_min"]), float(d["band_max"]))

        best, peak = pc.best_centre(wave, flux, var, band, args.window, args.step)
        rows.append(dict(ID=sid, best_center=round(best, 3), peak_snr_slide=round(peak, 3),
                         lya_obs=round(lya_wave, 3), band_min=round(band[0], 3),
                         band_max=round(band[1], 3), slide_window=args.window))
        msg = f"  {sid}  peak sliding S/N {peak:.2f} at {best:.1f} AA"
        if not args.no_plots:
            centres, snr, sm = pc.sliding_snr(
                wave, flux, var, args.window, args.step,
                lo=lya_wave - args.plot_halfwidth - 20, hi=lya_wave + args.plot_halfwidth + 20)
            png = pc.muse_path(sid, "sliding_png", args.subcube_dir)
            plot_sliding(png, sid, float(d["ra"]), float(d["dec"]), float(d["aperture"]),
                         centres, snr, sm, best, peak, lya_wave, band, args.window,
                         args.plot_halfwidth)
            msg += f"\n         {png}"
        print(msg, flush=True)

    new = pd.DataFrame(rows)
    out_path = os.path.abspath(args.outfile)
    if args.ids and os.path.exists(out_path) and len(new):
        old = pd.read_csv(out_path)
        new = pd.concat([old[~old["ID"].isin(new["ID"])], new]).sort_values("ID")
    if len(new):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        new.to_csv(out_path, index=False)

    print("\n--- Summary ---")
    print(f"  measured     {len(rows)}")
    print(f"  no spectrum  {n_missing}")
    if len(rows):
        print(f"\nWrote\n  {out_path}")


if __name__ == "__main__":
    main()