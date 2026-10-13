#!/usr/bin/env python3
"""
mc_lya_errors_grating.py

Monte Carlo uncertainties on the Lya fits from fit_lya_properties_grating.py,
for the PRIMER + MINERVA z_sys sample.

The method is unchanged from the JELS-ID version. Each successfully fitted
spectrum is refitted N times (default 500) under noise realisations, with the
fitter imported from fit_lya_properties_grating.py and centred on the same
best_center, so the model, bounds and weighting match the science run. The
scatter gives errors on the Lya peak, z_lya, flux and FWHM, and the z_lya
error is combined with z_sys_err for the Delta_v error,
    delta_v_err_kms = c / (1 + z_sys) x sqrt(sigma_z_lya^2 + z_sys_err^2)
split into delta_v_err_lya_kms and delta_v_err_sys_kms.

  --mode model (default)  realisations are the best-fit model + N(0, sqrt(var)),
                          a parametric bootstrap
  --mode data             realisations are the data + N(0, sqrt(var)), which
                          adds noise to noise and inflates the scatter by
                          about sqrt(2)

What changed
  - keyed on Isaac's ID, reads MUSE_catalogs/lya_fits.csv and the spectra in
    MUSE_subcubes/<ID>/
  - each source writes its own result, MUSE_subcubes/<ID>/<ID>_lya_mc.csv,
    then merge_mc_partials.py joins them onto lya_fits.csv
  - the slurm array indexes the sorted ID list of the sample (--task-index),
    not the row of the successful fits, so the numbering only changes when
    the sample does. Tasks whose source has no successful fit exit at once
  - run without --task-index (all sources, or --ids) it merges at the end

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv     ID, z_sys, z_sys_err
  MUSE_catalogs/lya_fits.csv                        fit_success, best_center
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz

Outputs
-------
  MUSE_subcubes/<ID>/<ID>_lya_mc.csv     one row of MC columns per source
  MUSE_catalogs/lya_fits_mc.csv          lya_fits.csv plus the MC columns
                                         (written by the merge)

Usage
-----
python mc_lya_errors_grating.py --ids 23070 50803 --n-mc 100    # quick test
python mc_lya_errors_grating.py                                  # all, one node
sbatch --array=0-<N-1> slurm/run_mc_lya_errors.slurm             # all, slurm
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as pc                      # noqa: E402
import fit_lya_properties_grating as flp    # noqa: E402
import merge_mc_partials                    # noqa: E402

FITS_CSV = f"{pc.MUSE_CAT_DIR}/lya_fits.csv"


def at_bound(value, low, high, rtol=1e-3):
    """True if a fitted parameter sits within rtol of either bound."""
    if not np.isfinite(value):
        return False
    span = abs(high - low)
    if not np.isfinite(span) or span == 0:
        return False
    tol = rtol * span
    return (abs(value - low) <= tol) or (abs(value - high) <= tol)


def run_mc_for_source(flp, npz_path, centre, z_sys, z_sys_err,
                      n_mc, fit_window, sigma_min, alpha_max, mode, rng,
                      resolve_frac, peak):
    """Refit one source under N noise realisations and derive error columns.

    Returns a dict of Monte Carlo results. The velocity-offset error combines
    the bootstrap scatter on z_lya with the catalogue z_sys_err.
    """
    out = {
        "n_mc_attempted": n_mc,
        "n_mc_success": 0,
        "lya_peak_repeat": np.nan,
        "lya_peak_err_mc": np.nan,
        "z_lya_err_mc": np.nan,
        "delta_v_err_lya_kms": np.nan,
        "delta_v_err_sys_kms": np.nan,
        "delta_v_err_kms": np.nan,
        "flux_fit_err_mc": np.nan,
        "fwhm_int_kms_err_mc": np.nan,
        "fwhm_obs_kms_err_mc": np.nan,
        "frac_unresolved_mc": np.nan,
        "frac_alpha_at_bound": np.nan,
        "frac_sigma_at_bound": np.nan,
        "frac_mu_at_bound": np.nan,
    }

    npz = np.load(npz_path, allow_pickle=True)
    wave = np.asarray(npz["wave"], dtype=float)
    flux = np.asarray(npz["flux"], dtype=float)
    var = np.asarray(npz["var"], dtype=float)

    if not np.isfinite(centre) or wave.size == 0:
        return out

    fit_kw = dict(fit_window=fit_window, sigma_min=sigma_min,
                  alpha_max=alpha_max, resolve_frac=resolve_frac, peak=peak)

    # Unperturbed refit, reproduces the science run and checks settings match.
    base_fit = flp.fit_skewed_gaussian(wave, flux, var, centre, **fit_kw)
    if not base_fit["fit_success"]:
        return out
    out["lya_peak_repeat"] = float(base_fit["lya_peak"])

    noise_sigma = np.sqrt(np.clip(var, 0.0, np.inf))

    if mode == "model":
        # The LSF-convolved model, since that is what the data are drawn from.
        base_spectrum = flp.skew_model_lsf(
            wave, base_fit["flux_fit"], base_fit["mu"],
            base_fit["sigma"], base_fit["alpha_skew"], base_fit["sigma_lsf_ang"],
        )
        base_spectrum = np.where(np.isfinite(base_spectrum), base_spectrum, 0.0)
    else:
        base_spectrum = flux

    peak_draws, flux_draws = [], []
    fwhm_int_draws, fwhm_obs_draws, resolved_draws = [], [], []
    alpha_draws, sigma_draws, mu_draws = [], [], []

    for _ in range(n_mc):
        realisation = base_spectrum + rng.normal(loc=0.0, scale=noise_sigma)
        fit = flp.fit_skewed_gaussian(wave, realisation, var, centre, **fit_kw)
        if not fit["fit_success"] or not np.isfinite(fit["lya_peak"]):
            continue
        peak_draws.append(float(fit["lya_peak"]))
        flux_draws.append(float(fit["flux_fit"]))
        fwhm_int_draws.append(float(fit["fwhm_int_kms"]))
        fwhm_obs_draws.append(float(fit["fwhm_obs_kms"]))
        resolved_draws.append(bool(fit["resolved"]))
        alpha_draws.append(float(fit["alpha_skew"]))
        sigma_draws.append(float(fit["sigma"]))
        mu_draws.append(float(fit["mu"]))

    peak_draws = np.asarray(peak_draws, dtype=float)
    out["n_mc_success"] = int(peak_draws.size)
    if peak_draws.size < 10:
        return out

    # Bootstrap error on the peak wavelength -> z_lya -> velocity.
    # Convert the peak draws to vacuum first, matching the science path where
    # z_lya is formed from the vacuum peak. The air/vacuum scaling is ~constant
    # so this barely changes the error, but keeps error and value consistent.
    peak_err = float(np.std(peak_draws, ddof=1))
    out["lya_peak_err_mc"] = peak_err
    peak_draws_vac = flp.air_to_vac(peak_draws)
    z_lya_draws = peak_draws_vac / flp.LYA_REST - 1.0
    z_lya_err = float(np.std(z_lya_draws, ddof=1))
    out["z_lya_err_mc"] = float(z_lya_err)

    if np.isfinite(z_sys):
        conv = flp.C_KMS / (1.0 + z_sys)
        dv_err_lya = conv * z_lya_err
        out["delta_v_err_lya_kms"] = float(dv_err_lya)
        if np.isfinite(z_sys_err):
            dv_err_sys = conv * z_sys_err
            out["delta_v_err_sys_kms"] = float(dv_err_sys)
            out["delta_v_err_kms"] = float(
                np.sqrt(dv_err_lya ** 2 + dv_err_sys ** 2))
        else:
            # No systemic error available, report the Lya term alone.
            out["delta_v_err_kms"] = float(dv_err_lya)

    flux_draws = np.asarray(flux_draws, dtype=float)
    flux_draws = flux_draws[np.isfinite(flux_draws)]
    if flux_draws.size >= 10:
        out["flux_fit_err_mc"] = float(np.std(flux_draws, ddof=1))

    for key, draws in (("fwhm_int_kms_err_mc", fwhm_int_draws),
                       ("fwhm_obs_kms_err_mc", fwhm_obs_draws)):
        d = np.asarray(draws, dtype=float)
        d = d[np.isfinite(d)]
        if d.size >= 10:
            out[key] = float(np.std(d, ddof=1))
    out["frac_unresolved_mc"] = float(1.0 - np.mean(resolved_draws))

    out["frac_alpha_at_bound"] = float(
        np.mean([at_bound(a, 0.0, alpha_max) for a in alpha_draws]))
    out["frac_sigma_at_bound"] = float(
        np.mean([at_bound(s, sigma_min, flp.SIGMA_MAX) for s in sigma_draws]))
    out["frac_mu_at_bound"] = float(
        np.mean([at_bound(m, centre - 10.0, centre + 10.0) for m in mu_draws]))
    return out



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=FITS_CSV)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs")
    p.add_argument("--task-index", type=int, default=None,
                   help="slurm array index, the Nth ID (0-based) of the sorted "
                        "sample. Writes that source's file only, no merge")
    p.add_argument("--n-mc", type=int, default=500)
    p.add_argument("--mode", choices=["model", "data"], default="model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="must match the science run")
    p.add_argument("--sigma-min", type=float, default=flp.SIGMA_MIN_DEFAULT)
    p.add_argument("--alpha-max", type=float, default=flp.ALPHA_MAX_DEFAULT)
    p.add_argument("--resolve-frac", type=float, default=flp.RESOLVE_FRAC_DEFAULT)
    p.add_argument("--peak", choices=["obs", "int"], default=flp.PEAK_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    print("mc_lya_errors_grating.py")
    print(f"Reading catalogue   {os.path.abspath(args.catalog)}")
    sample = pc.load_sample(args.catalog, args.ids)
    fits_path = os.path.abspath(args.fits_csv)
    print(f"Reading fits        {fits_path}")
    props = pd.read_csv(fits_path)
    props["ID"] = props["ID"].astype(int)
    props = props.set_index("ID")
    print(f"mode {args.mode}, N = {args.n_mc}, seed {args.seed}, fit +/-{args.fit_window:g} AA, "
          f"peak {args.peak}")

    if args.task_index is not None:
        if not 0 <= args.task_index < len(sample):
            print(f"Task index {args.task_index} is past the {len(sample)} sources, nothing to do")
            return
        sample = sample.iloc[[args.task_index]]
        print(f"Task {args.task_index}, ID {int(sample.iloc[0]['ID'])}")
    print("")

    n_done = 0
    for _, row in sample.iterrows():
        sid = int(row["ID"])
        if sid not in props.index:
            print(f"  {sid}  SKIP not in {os.path.basename(fits_path)}")
            continue
        prow = props.loc[sid]
        if str(prow["fit_success"]).lower() not in ("true", "1"):
            print(f"  {sid}  SKIP fit failed")
            continue
        npz_path = pc.muse_path(sid, "spectrum", args.subcube_dir)
        if not os.path.exists(npz_path):
            print(f"  {sid}  SKIP no spectrum at {npz_path}")
            continue

        z_sys = float(row["z_sys"])
        z_sys_err = float(row["z_sys_err"]) if "z_sys_err" in row and pd.notna(row["z_sys_err"]) else np.nan
        rng = np.random.default_rng(args.seed + sid)
        res = run_mc_for_source(flp, npz_path, float(prow["best_center"]), z_sys, z_sys_err,
                                args.n_mc, args.fit_window, args.sigma_min, args.alpha_max,
                                args.mode, rng, args.resolve_frac, args.peak)
        res = {"ID": sid, **res, "mc_mode": args.mode}

        rep, orig = res["lya_peak_repeat"], float(prow["lya_peak_wave"])
        if np.isfinite(rep) and np.isfinite(orig) and abs(rep - orig) > 1e-3:
            print(f"  {sid}  WARNING refit peak {rep:.3f} differs from lya_fits.csv "
                  f"{orig:.3f}, check the fit settings match the science run")

        out = pc.muse_path(sid, "mc", args.subcube_dir)
        pd.DataFrame([res]).to_csv(out, index=False)
        n_done += 1
        print(f"  {sid}  dv_err {res['delta_v_err_kms']:.1f} km/s (Lya "
              f"{res['delta_v_err_lya_kms']:.1f}, sys {res['delta_v_err_sys_kms']:.1f}), "
              f"{res['n_mc_success']}/{args.n_mc} refits converged\n         {out}", flush=True)

    print(f"\n{n_done} sources done")
    if args.task_index is None:
        print("")
        merge_mc_partials.merge(fits_path, args.subcube_dir)


if __name__ == "__main__":
    main()