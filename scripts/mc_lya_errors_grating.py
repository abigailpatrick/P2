#!/usr/bin/env python3
"""
Monte Carlo uncertainties for the skewed-Gaussian Ly-alpha fits (P2).

Standalone add-on to fit_lya_properties_grating.py. Reads the saved NPZ
extractions written by ap_extract_specs_grating.py, refits each source N times
under noise realisations, and records the scatter in the recovered Lya peak
wavelength, z_lya, line flux and FWHM. It then combines the bootstrap error on
z_lya with the systemic-redshift error z_sys_err (taken from the catalogue) to
give a velocity-offset uncertainty, delta_v_err_kms, split into its Lya and
systemic contributions so it is clear which dominates.

The fitting itself is imported directly from fit_lya_properties_grating so the
model, bounds, initial guesses, weighting and fit centre are identical to the
science run. The fit is centred on best_center from the sliding-S/N CSV, exactly
as in the science run, so pass the same --snr-csv and --fit-window here.

Two noise modes, as in the Paper 1 flux-error add-on, and the distinction
matters:

   --mode model  (default, parametric bootstrap)
      Realisations are best_fit_model + N(0, sqrt(var)). The scatter in the
      refits estimates the uncertainty on the measured parameters. This is the
      statistically correct construction.

   --mode data
      Realisations are observed_flux + N(0, sqrt(var)). Because the observed
      spectrum already contains one noise realisation this adds noise on top of
      noise, inflating the scatter by roughly sqrt(2) in the linear regime. It
      is conservative rather than wrong, but understand the factor before
      quoting it.

Running both is cheap and the comparison is a useful check.

delta_v_err_kms = (c / (1 + z_sys)) * sqrt( sigma_z_lya^2 + z_sys_err^2 )
  delta_v_err_lya_kms = (c / (1 + z_sys)) * sigma_z_lya   (bootstrap)
  delta_v_err_sys_kms = (c / (1 + z_sys)) * z_sys_err     (catalogue)

Example
-------
python mc_lya_errors_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --snr-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --properties-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv \
  --out-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv \
  --n-mc 500 --mode model
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def load_fitter(fitter_dir):
    """Import the fitting machinery from fit_lya_properties_grating.py."""
    fitter_dir = os.path.abspath(fitter_dir)
    if fitter_dir not in sys.path:
        sys.path.insert(0, fitter_dir)
    import fit_lya_properties_grating as flp
    return flp


def npz_scalar(npz, key, default=np.nan):
    """Read a scalar stored in an NPZ as a 0-d array."""
    if key not in npz.files:
        return default
    try:
        return float(np.asarray(npz[key]).item())
    except (ValueError, TypeError):
        return default


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
                      n_mc, fit_window, sigma_min, alpha_max, mode, rng):
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
        "fwhm_kms_err_mc": np.nan,
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

    # Unperturbed refit, reproduces the science run and checks settings match.
    base_fit = flp.fit_skewed_gaussian(
        wave, flux, var, centre,
        fit_window=fit_window, sigma_min=sigma_min, alpha_max=alpha_max,
    )
    if not base_fit["fit_success"]:
        return out
    out["lya_peak_repeat"] = float(base_fit["lya_peak"])

    noise_sigma = np.sqrt(np.clip(var, 0.0, np.inf))

    if mode == "model":
        base_spectrum = flp.skew_model(
            wave, base_fit["flux_fit"], base_fit["mu"],
            base_fit["sigma"], base_fit["alpha_skew"],
        )
        base_spectrum = np.where(np.isfinite(base_spectrum), base_spectrum, 0.0)
    else:
        base_spectrum = flux

    peak_draws, flux_draws, fwhm_draws = [], [], []
    alpha_draws, sigma_draws, mu_draws = [], [], []

    for _ in range(n_mc):
        realisation = base_spectrum + rng.normal(loc=0.0, scale=noise_sigma)
        fit = flp.fit_skewed_gaussian(
            wave, realisation, var, centre,
            fit_window=fit_window, sigma_min=sigma_min, alpha_max=alpha_max,
        )
        if not fit["fit_success"] or not np.isfinite(fit["lya_peak"]):
            continue
        peak_draws.append(float(fit["lya_peak"]))
        flux_draws.append(float(fit["flux_fit"]))
        fwhm_draws.append(float(fit["fwhm_kms"]))
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

    fwhm_draws = np.asarray(fwhm_draws, dtype=float)
    fwhm_draws = fwhm_draws[np.isfinite(fwhm_draws)]
    if fwhm_draws.size >= 10:
        out["fwhm_kms_err_mc"] = float(np.std(fwhm_draws, ddof=1))

    out["frac_alpha_at_bound"] = float(
        np.mean([at_bound(a, 0.0, alpha_max) for a in alpha_draws]))
    out["frac_sigma_at_bound"] = float(
        np.mean([at_bound(s, sigma_min, 50.0) for s in sigma_draws]))
    out["frac_mu_at_bound"] = float(
        np.mean([at_bound(m, centre - 10.0, centre + 10.0) for m in mu_draws]))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo uncertainties for P2 Lya skewed-Gaussian fits.")
    parser.add_argument("--indir",
                        default="/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts",
                        help="Directory of {ID}_spectrum.npz files.")
    parser.add_argument("--snr-csv",
                        default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv",
                        help="Sliding-S/N CSV providing best_center per ID.")
    parser.add_argument("--catalog",
                        default="/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv",
                        help="Catalogue with ID, z_sys, z_sys_err columns.")
    parser.add_argument("--properties-csv",
                        default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv",
                        help="lya_properties.csv from the science fit run.")
    parser.add_argument("--out-csv",
                        default="/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv",
                        help="Output CSV with the Monte Carlo columns merged in.")

    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--z-col", default="z_sys")
    parser.add_argument("--z-col-fallback", default="z_dja")
    parser.add_argument("--z-err-col", default="z_sys_err",
                        help="Catalogue column with the systemic-redshift error.")

    parser.add_argument("--n-mc", type=int, default=500,
                        help="Number of noise realisations per source.")
    parser.add_argument("--mode", choices=["model", "data"], default="model",
                        help="Perturb the best-fit model (default) or the "
                             "observed spectrum.")
    parser.add_argument("--fit-window", type=float, default=25.0,
                        help="Must match the science run (default 25).")
    parser.add_argument("--sigma-min", type=float, default=1.0,
                        help="Must match the science run.")
    parser.add_argument("--alpha-max", type=float, default=15.0,
                        help="Must match the science run (default 15).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--row-index", type=int, default=None,
                        help="Process only the Nth successfully-fitted source "
                             "(0-based). Used by the SLURM array so each task "
                             "does one source. In this mode the script writes a "
                             "one-row partial CSV to --out-csv and does not "
                             "merge; a separate merge step combines the partials. "
                             "Omit to process all sources and write the merged "
                             "CSV as usual.")
    parser.add_argument("--fitter-dir", default=None,
                        help="Directory holding fit_lya_properties_grating.py. "
                             "Defaults to this script's directory.")
    args = parser.parse_args()

    fitter_dir = args.fitter_dir or os.path.dirname(os.path.abspath(__file__))
    flp = load_fitter(fitter_dir)

    print("[CONFIG]")
    print(f"  Spectra directory : {os.path.abspath(args.indir)}")
    print(f"  Sliding-S/N CSV   : {os.path.abspath(args.snr_csv)}")
    print(f"  Catalogue         : {os.path.abspath(args.catalog)}")
    print(f"  Properties CSV    : {os.path.abspath(args.properties_csv)}")
    print(f"  Output CSV        : {os.path.abspath(args.out_csv)}")
    print(f"  mode = {args.mode}, N = {args.n_mc}, seed = {args.seed}")
    print("")

    # Catalogue: z_sys (with fallback) and z_sys_err
    catalog = pd.read_csv(args.catalog)
    for col in (args.id_col, args.z_col):
        if col not in catalog.columns:
            raise KeyError(
                f"Column '{col}' not found in catalogue. Available: {list(catalog.columns)}")
    use_fallback = bool(args.z_col_fallback) and args.z_col_fallback in catalog.columns
    has_zerr = args.z_err_col in catalog.columns
    if not has_zerr:
        print(f"[WARN] catalogue has no '{args.z_err_col}' column, the systemic "
              f"error term will be NaN and delta_v_err_kms will hold the Lya "
              f"term alone.")
    catalog[args.id_col] = catalog[args.id_col].astype("Int64")
    catalog = catalog.set_index(args.id_col)

    # Sliding-S/N best_center per ID
    snr = pd.read_csv(args.snr_csv)
    if "ID" not in snr.columns or "best_center" not in snr.columns:
        raise KeyError("Sliding-S/N CSV must have ID and best_center columns.")
    snr["ID"] = snr["ID"].astype("Int64")
    best_center_by_id = snr.set_index("ID")["best_center"].to_dict()

    # Which sources to bootstrap: those in the properties CSV
    props = pd.read_csv(args.properties_csv)
    success = props["fit_success"].astype(str).str.lower().isin(["true", "1"])
    todo = props[success].reset_index(drop=True)
    n_success = len(todo)
    print(f"[INFO] {n_success} successfully-fitted sources to bootstrap")

    single_mode = args.row_index is not None
    if single_mode:
        if args.row_index < 0 or args.row_index >= n_success:
            print(f"[INFO] row-index {args.row_index} out of range "
                  f"(0..{n_success - 1}), nothing to do for this task.")
            return
        todo = todo.iloc[[args.row_index]]
        print(f"[INFO] SLURM array mode: processing row {args.row_index} "
              f"(ID {int(todo.iloc[0]['ID'])}) only")
    print("")

    records = []

    for _, prow in todo.iterrows():
        idx = int(prow["ID"])
        # Per-source seed so array tasks are independent yet reproducible
        rng = np.random.default_rng(args.seed + idx)
        npz_path = os.path.join(args.indir, f"{idx}_spectrum.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] missing extraction for ID {idx}, skipping")
            continue

        if idx not in catalog.index:
            print(f"[WARN] ID {idx} not in catalogue, skipping")
            continue

        z_sys, _ = flp.resolve_redshift(
            catalog.loc[idx], args.z_col,
            args.z_col_fallback if use_fallback else "")
        z_sys_err = (pd.to_numeric(catalog.loc[idx].get(args.z_err_col),
                                   errors="coerce") if has_zerr else np.nan)

        centre = best_center_by_id.get(idx, np.nan)
        if not np.isfinite(centre):
            centre = float(prow.get("best_center", np.nan))

        res = run_mc_for_source(
            flp, npz_path, centre, z_sys, z_sys_err,
            args.n_mc, args.fit_window, args.sigma_min, args.alpha_max,
            args.mode, rng,
        )
        res["ID"] = idx

        # Check the unperturbed refit reproduces the science peak
        repeat = res["lya_peak_repeat"]
        original = float(prow.get("lya_peak_wave", np.nan))
        if np.isfinite(repeat) and np.isfinite(original) and original != 0:
            drift = abs(repeat - original) / abs(original)
            if drift > 1e-4:
                print(f"[WARN] ID {idx}: refit peak differs from science run by "
                      f"{1e4 * drift:.1f}e-4 frac, check fit-window/sigma/alpha")

        records.append(res)
        print(f"  ID {idx}: dv_err = {res['delta_v_err_kms']:.1f} km/s "
              f"(Lya {res['delta_v_err_lya_kms']:.1f}, "
              f"sys {res['delta_v_err_sys_kms']:.1f}), "
              f"{res['n_mc_success']}/{args.n_mc} fits converged")

    mc = pd.DataFrame(records)

    if single_mode:
        # Write a one-row partial with the MC columns (plus ID). The merge step
        # combines all partials and joins them onto the properties CSV.
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        mc.to_csv(args.out_csv, index=False)
        print("")
        print(f"[INFO] written partial {args.out_csv}")
        return

    merged = props.merge(mc, on="ID", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print("")
    print(f"[INFO] written {args.out_csv}")

    # Summary of which term dominates the velocity error
    valid = merged[np.isfinite(merged["delta_v_err_kms"])]
    if len(valid):
        med = float(np.nanmedian(valid["delta_v_err_kms"]))
        med_lya = float(np.nanmedian(valid["delta_v_err_lya_kms"]))
        med_sys = float(np.nanmedian(valid["delta_v_err_sys_kms"]))
        print("")
        print("=" * 68)
        print(f"N sources with delta_v_err : {len(valid)}")
        print(f"median delta_v_err_kms     : {med:.1f} km/s")
        print(f"median Lya term            : {med_lya:.1f} km/s")
        print(f"median systemic term       : {med_sys:.1f} km/s")
        print("=" * 68)


if __name__ == "__main__":
    main()


"""
python mc_lya_errors_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --snr-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --properties-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv \
  --out-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv \
  --n-mc 500 --mode model
"""