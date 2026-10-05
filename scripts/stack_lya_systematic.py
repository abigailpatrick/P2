#!/usr/bin/env python3
"""
Systemic-anchored composite Lya spectra for the P2 grating sample, split into
two redshift bins and two M_UV bins, each shown as a two-panel figure (one bin
above the other) in the style of Tang et al. (2024).

Method (following Tang et al. 2024)
-----------------------------------
For every source:
  1. Read the MUSE aperture spectrum {ID}_spectrum.npz written by
     ap_extract_specs_grating.py (full spectral axis, air wavelengths).
  2. Blank the sodium AO laser gap (observed air 5780-5810 AA by default).
  3. Convert air to vacuum wavelengths (Ciddor 1996, same function as
     fit_lya_properties_grating.py).
  4. Shift to the rest frame with the systemic redshift,
         lambda_rest = lambda_vac / (1 + z_sys)
         f_rest      = f_obs * (1 + z_sys)
  5. Interpolate onto a common rest-frame grid (default 0.2 AA bins).
Spectra are NOT normalised by their Lya flux (unlike Tang et al. 2024), so
that sources without an individual Lya detection still contribute to the stack.
Then, per bin:
  6. Median-combine the rest-frame flux densities in each wavelength bin.
  7. Convert to velocity, Delta_v = c (lambda_rest - 1215.67) / 1215.67.
The shaded region is the 16th to 84th percentile of a bootstrap over sources
(resampling sources with replacement and re-taking the median). For display,
each composite is scaled so its peak within --peak-window km/s is 1.

z_sys is NOT taken from the .npz or from grating_sources_with_zsys.csv (both
can be stale). It is recomputed here from systemic_redshifts_by_JELS_ID.csv
with the priority used by find_delta_v_from_best_zsys_line.py:
    OIII (z_OIII_snr >= --oiii-snr-min), then Ha, Hbeta, NII, OII.

Sample
------
Every source in systemic_redshifts_by_JELS_ID.csv with a usable line and
in_muse == 1 in grating_sources_by_JELS_ID.csv, minus --exclude. The default
exclude list holds the duplicate members dropped by group_lya_sample.py
(42990, 43604, 49296, 49694) and 48086, whose Lya products were made with a
redshift 48 AA off (its continuum subtraction mask was centred in the wrong
place), so it needs rerunning before it is stacked.
The M_UV split uses only sources with good_muv == True.

Outputs (all to --outdir, full paths printed)
---------------------------------------------
  lya_stack_zsplit.png              two panels, z < split above, z >= split below
  lya_stack_muvsplit.png            two panels, bright above, faint below
  lya_stack_<bin>.csv               velocity, rest wavelength, composite,
                                    bootstrap 16/84 percentiles, N per pixel
  lya_stack_members.csv             one row per source with its bin labels

Usage
-----
python stack_lya_systemic.py
python stack_lya_systemic.py --z-split 3.5 --muv-split -19.0 --nboot 1000
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LYA_REST = 1215.67          # AA, vacuum
C_KMS = 299792.458
LINE_PRIORITY = ["OIII", "Ha", "Hbeta", "NII", "OII"]

DEFAULT_EXCLUDE = [42990, 43604, 49296, 49694, 48086]


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Systemic-anchored Lya stacks in redshift and M_UV bins.")
    p.add_argument("--systemic-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/systemic_redshifts_by_JELS_ID.csv")
    p.add_argument("--grating-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv",
                   help="Used only for the in_muse flag.")
    p.add_argument("--lya-csv",
                   default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                   help="Source of lya_snr, used only by --min-snr.")
    p.add_argument("--muv-csv",
                   default="/ceph/cephfs/apatrick/P2/jwst_catalogs/muv_beta_by_JELS_ID.csv")
    p.add_argument("--spec-dir",
                   default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts",
                   help="Directory of {ID}_spectrum.npz files.")
    p.add_argument("--outdir",
                   default="/ceph/cephfs/apatrick/P2/plots/stacks")

    p.add_argument("--oiii-snr-min", type=float, default=13.0)
    p.add_argument("--z-split", type=float, default=3.5)
    p.add_argument("--muv-split", type=float, default=-19.0)
    p.add_argument("--exclude", type=int, nargs="*", default=DEFAULT_EXCLUDE)
    p.add_argument("--min-snr", type=float, default=0.0,
                   help="Only stack sources with lya_snr >= this (default 0, all).")

    p.add_argument("--bin", type=float, default=0.2,
                   help="Rest-frame bin size in AA.")
    p.add_argument("--vmin", type=float, default=-2000.0)
    p.add_argument("--vmax", type=float, default=2000.0)
    p.add_argument("--peak-window", type=float, default=1000.0,
                   help="Composite is scaled to its peak within +/- this (km/s).")
    p.add_argument("--ao-gap", type=float, nargs=2, default=[5780.0, 5810.0],
                   help="Observed air wavelength range to blank (AA).")
    p.add_argument("--nboot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def air_to_vac(wave_air):
    """Air to vacuum, Ciddor (1996). Copied from fit_lya_properties_grating.py."""
    wave_air = np.asarray(wave_air, dtype=float)
    s2 = (1e4 / wave_air) ** 2
    n = (1.0 + 0.05792105 / (238.0185 - s2)
         + 0.00167917 / (57.362 - s2))
    return wave_air * n


def best_zsys(row, oiii_snr_min):
    """(z, z_err, line) by the find_delta_v_from_best_zsys_line.py priority."""
    for line in LINE_PRIORITY:
        z = row.get(f"z_{line}")
        if not np.isfinite(z):
            continue
        if line == "OIII" and not (row.get("z_OIII_snr", np.nan) >= oiii_snr_min):
            continue
        return z, row.get(f"z_{line}_err", np.nan), line
    return np.nan, np.nan, None


def rest_frame_spectrum(path, z, grid, ao_gap):
    """Load one npz, shift to the rest frame and put it on the grid."""
    d = np.load(path)
    wave_air = np.asarray(d["wave"], dtype=float)
    flux = np.asarray(d["flux"], dtype=float)

    flux = flux.copy()
    flux[(wave_air >= ao_gap[0]) & (wave_air <= ao_gap[1])] = np.nan

    wave_rest = air_to_vac(wave_air) / (1.0 + z)
    f_rest = flux * (1.0 + z)

    good = np.isfinite(f_rest)
    out = np.full(grid.size, np.nan)
    if good.sum() < 2:
        return out
    out = np.interp(grid, wave_rest[good], f_rest[good], left=np.nan, right=np.nan)

    # Re-blank grid points that fall inside a NaN stretch of the input
    bad_rest = wave_rest[~good]
    if bad_rest.size:
        step = np.nanmedian(np.diff(wave_rest))
        idx = np.searchsorted(bad_rest, grid)
        lo = np.abs(grid - bad_rest[np.clip(idx - 1, 0, bad_rest.size - 1)])
        hi = np.abs(grid - bad_rest[np.clip(idx, 0, bad_rest.size - 1)])
        out[np.minimum(lo, hi) < step] = np.nan
    return out


def stack(specs, nboot, rng):
    """Median composite plus bootstrap 16/84 percentiles over sources."""
    specs = np.asarray(specs)
    comp = np.nanmedian(specs, axis=0)
    npix = np.sum(np.isfinite(specs), axis=0)
    boots = np.empty((nboot, specs.shape[1]))
    n = specs.shape[0]
    for i in range(nboot):
        boots[i] = np.nanmedian(specs[rng.integers(0, n, n)], axis=0)
    lo, hi = np.nanpercentile(boots, [16, 84], axis=0)
    return comp, lo, hi, npix


def plot_pair(v, results, labels, colours, outpath, vmin, vmax, peak_window):
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 7.0), sharex=True)
    for ax, key, label, col in zip(axes, results, labels, colours):
        comp, lo, hi, _, n = results[key]
        win = np.abs(v) <= peak_window
        scale = np.nanmax(comp[win]) if np.any(np.isfinite(comp[win])) else 1.0
        ax.fill_between(v, lo / scale, hi / scale, step="mid",
                        color=col, alpha=0.3, lw=0)
        ax.step(v, comp / scale, where="mid", color=col, lw=1.2)
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel(r"$f_\lambda$ (arbitrary units)")
        ax.text(0.04, 0.93, f"{label}\nN = {n}", transform=ax.transAxes,
                va="top", ha="left")
        ax.set_xlim(vmin, vmax)
    axes[-1].set_xlabel(r"$\Delta v$ (km s$^{-1}$)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Inputs")
    for name, path in [("systemic", args.systemic_csv), ("grating", args.grating_csv),
                       ("lya", args.lya_csv), ("muv", args.muv_csv),
                       ("spectra", args.spec_dir)]:
        print(f"  {name:9s}: {os.path.abspath(path)}")
    print(f"  outdir   : {outdir}\n")

    sysz = pd.read_csv(args.systemic_csv)
    grat = pd.read_csv(args.grating_csv)[["ID", "in_muse"]]
    lya = pd.read_csv(args.lya_csv)[["ID", "lya_snr"]]
    muv = pd.read_csv(args.muv_csv)[["ID", "M_UV", "good_muv"]]
    for t in (sysz, grat, lya, muv):
        t["ID"] = t["ID"].astype(int)

    z = sysz.apply(lambda r: pd.Series(best_zsys(r, args.oiii_snr_min),
                                       index=["z_sys", "z_sys_err", "z_sys_line"]),
                   axis=1)
    cat = pd.concat([sysz[["ID"]], z], axis=1)
    cat = cat.merge(grat, on="ID", how="left")
    cat = cat[(cat["in_muse"] == 1) & cat["z_sys"].notna()]
    print(f"Sources with usable z_sys in MUSE: {len(cat)}")

    cat = cat[~cat["ID"].isin(args.exclude)]
    print(f"After excluding {sorted(args.exclude)}: {len(cat)}")

    cat = cat.merge(lya, on="ID", how="left").merge(muv, on="ID", how="left")

    # Build the rest-frame grid wide enough for the velocity range
    wmin = LYA_REST * (1 + args.vmin / C_KMS) - 2 * args.bin
    wmax = LYA_REST * (1 + args.vmax / C_KMS) + 2 * args.bin
    grid = np.arange(wmin, wmax + args.bin, args.bin)
    v = C_KMS * (grid - LYA_REST) / LYA_REST

    specs, kept = [], []
    for _, r in cat.iterrows():
        sid = int(r["ID"])
        if args.min_snr > 0 and not (r["lya_snr"] >= args.min_snr):
            print(f"[SKIP] {sid}: lya_snr {r['lya_snr']:.2f} < {args.min_snr}")
            continue
        path = os.path.join(args.spec_dir, f"{sid}_spectrum.npz")
        if not os.path.exists(path):
            print(f"[SKIP] {sid}: spectrum not found at {path}")
            continue
        s = rest_frame_spectrum(path, r["z_sys"], grid, args.ao_gap)
        if not np.any(np.isfinite(s)):
            print(f"[SKIP] {sid}: no valid pixels in the velocity window")
            continue
        specs.append(s)
        kept.append(r)
    kept = pd.DataFrame(kept).reset_index(drop=True)
    specs = np.array(specs)
    print(f"\nStacked sample: {len(kept)} sources\n")

    kept["z_bin"] = np.where(kept["z_sys"] < args.z_split, "zlow", "zhigh")
    bright = kept["M_UV"] < args.muv_split
    kept["muv_bin"] = np.where(kept["good_muv"] == True,
                               np.where(bright, "bright", "faint"), "")

    splits = {
        "zsplit": ("z_bin", ["zlow", "zhigh"],
                   [f"$z < {args.z_split}$", rf"$z \geq {args.z_split}$"],
                   ["0.25", "tab:blue"]),
        "muvsplit": ("muv_bin", ["bright", "faint"],
                     [rf"$M_{{\rm UV}} < {args.muv_split}$",
                      rf"$M_{{\rm UV}} \geq {args.muv_split}$"],
                     ["tab:purple", "tab:orange"]),
    }

    for split, (col, keys, labels, colours) in splits.items():
        results = {}
        for key in keys:
            sel = (kept[col] == key).values
            n = int(sel.sum())
            if n == 0:
                print(f"[WARN] {split}: bin '{key}' is empty")
                continue
            comp, lo, hi, npix = stack(specs[sel], args.nboot, rng)
            results[key] = (comp, lo, hi, npix, n)
            ids = kept.loc[sel, "ID"].astype(int).tolist()
            print(f"{split} {key}: N = {n}  IDs = {ids}")

            out_csv = os.path.join(outdir, f"lya_stack_{key}.csv")
            pd.DataFrame({"dv_kms": v, "wave_rest": grid, "flux": comp,
                          "flux_p16": lo, "flux_p84": hi, "n_spec": npix}
                         ).to_csv(out_csv, index=False)
            print(f"  saved {out_csv}")

        if len(results) == 2:
            out_png = os.path.join(outdir, f"lya_stack_{split}.png")
            plot_pair(v, results, labels, colours, out_png,
                      args.vmin, args.vmax, args.peak_window)
            print(f"  saved {out_png}\n")

    members = os.path.join(outdir, "lya_stack_members.csv")
    kept[["ID", "z_sys", "z_sys_err", "z_sys_line", "M_UV", "good_muv",
          "lya_snr", "z_bin", "muv_bin"]].to_csv(members, index=False)
    print(f"saved {members}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()