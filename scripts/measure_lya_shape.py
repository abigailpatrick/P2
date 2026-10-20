#!/usr/bin/env python3
"""
measure_lya_shape.py

The fraction of Lya flux blueward of systemic, f_blue, for every source with
a Lya fit. Run it after fit_lya_properties_grating.py (step 13) and
fit_lya_double_peak.py (step 14b). build_lya_catalog.py joins the result.

    f_blue = F(v < 0) / F(vmin < v < vmax)

with v the velocity relative to Lya at z_sys (air wavelength on the cube, as
pc.lya_obs_air), and the window vmin to vmax (--vmin -1000, --vmax +1000
km/s) the same for every measurement below. A purely red line gives 0, a line
symmetric about systemic 0.5. The same quantity as the v < 0 flux fraction of
Trainor et al. (2015). The blue/red ratios of Erb et al. (2014, A_Lya) and
Hayes et al. (2021, L_B/R) convert as f_blue = R / (1 + R).

Three versions
--------------
  f_blue_int    from the preferred model before the LSF (the intrinsic
                profile). Not smeared by the ~150 km/s MUSE resolution, but
                for a single skewed Gaussian the blue fraction is set mostly
                by the model shape (alpha >= 0 puts the tail on the red side).
  f_blue_conv   the same model convolved with the MUSE LSF, so it can be
                compared like for like with f_blue_data.
  f_blue_data   straight from the aperture spectrum, the flux summed either
                side of systemic over the window, with fractional pixels at
                the edges. Model independent, but includes the LSF smearing
                and the noise, so it can be negative or above 1 for faint
                lines.
The model is always the catalogue's lya_model. Both scripts take it from
pc.preferred_lya_model with the same inputs (z_lya in lya_fits_mc.csv and
double_peak in lya_double_fits.csv), so the double fit where double_peak = 1
and the single skewed Gaussian otherwise. There is no fallback: a source
whose preferred model has missing parameters gets no f_blue
(shape_status no_double_params or no_fit_params). build_lya_catalog.py also
blanks f_blue for any source whose shape_model differs from its lya_model,
so a stale lya_shape.csv can never put another model's value into the
catalogue. f_blue_int_single is the single fit's value for every source with
one, kept separately, to see how much the model choice matters.

The model fractions are exact, from the skew-normal and Gaussian cumulative
distributions (scipy.stats), not a numerical integral.

Errors
------
--n-mc (200) draws. Each draw takes z_sys from N(z_sys, z_sys_err), so the
dividing line moves with the systemic error, and
  model   refits the preferred model to the best-fit model plus the
          spectrum's noise (a parametric bootstrap, as the MC steps 14 and
          14b), on the same pixels (best_center +/- 25 AA) and with the same
          limits as the original fit
  data    adds the spectrum's noise to the measured flux over the window.
Each value is the best fit, with _err_lo = value - p16, _err_hi = p84 - value
(both clipped at 0) and _err = their mean, from the draws.

Columns of the output CSV
-------------------------
  ID, z_sys, lya_snr, shape_model (single or double)
  f_blue_int, f_blue_conv, f_blue_data       each + _err_lo, _err_hi, _err
  f_blue_int_single (+ _err)                 the single skewed Gaussian
  flux_blue_data, flux_red_data (+ _err)     1e-20 erg/s/cm2, the data sums
  vmin_kms, vmax_kms, n_mc_success,
  shape_status     ok, no_spectrum, no_fit, no_fit_params or no_double_params

Inputs
------
  jwst_catalogs/primer_minerva_in_muse_zsys.csv   z_sys, z_sys_err
  MUSE_catalogs/lya_fits_mc.csv                   single-fit parameters and z_lya
  MUSE_catalogs/lya_double_fits.csv               double-fit parameters (optional)
  MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz

Outputs
-------
  MUSE_catalogs/lya_shape.csv
  MUSE_subcubes/<ID>/<ID>_lya_fblue.png    velocity profile with the blue side
                                           shaded, model and data (--no-plots)

Usage
-----
python measure_lya_shape.py
python measure_lya_shape.py --ids 31364 26917          # replaces only these rows
python measure_lya_shape.py --n-mc 50 --no-plots       # quick
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm

import p2_common as pc
import fit_lya_properties_grating as flp
import fit_lya_double_peak as fdp

C_KMS = pc.C_KMS
OUT_CSV = f"{pc.MUSE_CAT_DIR}/lya_shape.csv"
FIG_KIND = "lya_fblue.png"
SINGLE_PARS = ("flux_fit", "mu", "sigma", "alpha_skew", "sigma_lsf_ang")
KINDS = ("int", "conv", "data")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zsys", default=pc.ZSYS_CATALOG)
    p.add_argument("--fits-csv", default=pc.LYA_FITS_MC,
                   help="the single fits, the same file build_lya_catalog.py reads")
    p.add_argument("--double-csv", default=pc.LYA_DOUBLE_FITS)
    p.add_argument("--subcube-dir", default=pc.MUSE_SUBCUBE_DIR)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--ids", type=int, nargs="+", help="only these IDs, rows replaced in --out-csv")
    p.add_argument("--vmin", type=float, default=-1000.0, help="km/s, blue edge of the window")
    p.add_argument("--vmax", type=float, default=1000.0, help="km/s, red edge of the window")
    p.add_argument("--fit-window", type=float, default=25.0,
                   help="AA about best_center for the MC refits, as the fits (default 25)")
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fractions
# ---------------------------------------------------------------------------

def cdf_single(x, flux, mu, sig, alpha):
    return flux * skewnorm.cdf(x, alpha, loc=mu, scale=sig)


def model_cdf(x, model, pars, s_lsf, convolved):
    """Cumulative flux of the model at wavelength(s) x (air AA)."""
    if model == "single":
        F, mu, sig, alpha = pars
        if convolved:
            sig, alpha = flp.convolved_skew_params(sig, alpha, s_lsf)
        return cdf_single(x, F, mu, sig, alpha)
    fb, mub, sb, fr, mur, sr, ar = pars
    if convolved:
        sb = np.hypot(sb, s_lsf)
        sr, ar = flp.convolved_skew_params(sr, ar, s_lsf)
    return fb * norm.cdf(x, loc=mub, scale=sb) + cdf_single(x, fr, mur, sr, ar)


def model_fblue(model, pars, s_lsf, lam_sys, lo, hi, convolved):
    c = model_cdf(np.array([lo, lam_sys, hi]), model, pars, s_lsf, convolved)
    tot = c[2] - c[0]
    return float((c[1] - c[0]) / tot) if tot > 0 else np.nan


def cum_flux(wave, flux):
    """Cumulative flux at the pixel edges, for sums with fractional pixels."""
    dw = np.gradient(wave)
    edges = np.concatenate([[wave[0] - dw[0] / 2], wave + dw / 2])
    return edges, np.concatenate([[0.0], np.cumsum(flux * dw)])


def data_sums(wave, flux, lam_sys, lo, hi):
    """Flux from lo to lam_sys and from lam_sys to hi, fractional pixels at the
    ends (the cumulative flux is linear across each pixel)."""
    edges, cum = cum_flux(wave, flux)
    a, s, b = np.interp([lo, lam_sys, hi], edges, cum)
    return float(s - a), float(b - s)


def data_sum_errors(wave, var, lam_sys, lo, hi):
    """1 sigma errors on the two data sums (pixels partly inside counted by
    their fraction squared, a small effect)."""
    dw = np.gradient(wave)
    left, right = wave - dw / 2, wave + dw / 2

    def frac(a, b):
        return np.clip((np.minimum(right, b) - np.maximum(left, a)) / dw, 0, 1)
    fb, fr = frac(lo, lam_sys), frac(lam_sys, hi)
    return (float(np.sqrt(np.nansum((fb * dw) ** 2 * var))),
            float(np.sqrt(np.nansum((fr * dw) ** 2 * var))))


def summarise(best, draws):
    v = np.asarray(draws, float)
    v = v[np.isfinite(v)]
    if v.size < 10 or not np.isfinite(best):
        return best, np.nan, np.nan, np.nan
    p16, p84 = np.percentile(v, [16, 84])
    lo, hi = max(best - p16, 0.0), max(p84 - best, 0.0)
    return best, lo, hi, 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Refits for the MC
# ---------------------------------------------------------------------------

def refit_single(w, f, e, s_lsf, centre, p0):
    lower = [0.0, centre - fdp.RED_MU_WINDOW, flp.SIGMA_MIN_DEFAULT, 0.0]
    upper = [np.inf, centre + fdp.RED_MU_WINDOW, flp.SIGMA_MAX, flp.ALPHA_MAX_DEFAULT]
    lo, hi = np.array(lower), np.array(upper)
    span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    p0 = np.clip(p0, lo + 1e-6 * span, hi - 1e-6 * span)

    def model(x, F, mu, sig, alpha):
        return fdp.skew_lsf(x, F, mu, sig, alpha, s_lsf)
    try:
        popt, _ = curve_fit(model, w, f, p0=p0, sigma=e, absolute_sigma=True,
                            bounds=(lower, upper), maxfev=20000)
    except Exception:
        return None
    return popt


def refit(model, w, f, e, s_lsf, centre, lam_sys, pars):
    if model == "single":
        return refit_single(w, f, e, s_lsf, centre, pars)
    bnds = fdp.bounds_for(lam_sys, centre, flp.SIGMA_MIN_DEFAULT, flp.ALPHA_MAX_DEFAULT)
    b = fdp.fit_double(w, f, e, s_lsf, bnds, [np.asarray(pars, float)])
    return None if b is None else b[0]


def model_flux(model, w, pars, s_lsf):
    if model == "single":
        return fdp.skew_lsf(w, *pars, s_lsf)
    return fdp.double_model(w, *pars, s_lsf)


# ---------------------------------------------------------------------------
# Per source
# ---------------------------------------------------------------------------

def window(lam_sys, vmin, vmax):
    return lam_sys * (1 + vmin / C_KMS), lam_sys * (1 + vmax / C_KMS)


def run_source(sid, z_sys, z_err, frow, drow, a, rng):
    res = dict(ID=sid, z_sys=z_sys, lya_snr=float(frow.get("lya_snr", np.nan)),
               vmin_kms=a.vmin, vmax_kms=a.vmax, shape_model="", n_mc_success=0,
               shape_status="ok")
    path = pc.muse_path(sid, "spectrum", a.subcube_dir)
    if not os.path.exists(path):
        res["shape_status"] = "no_spectrum"
        return res, None
    # the catalogue's rule, never anything else
    model = pc.preferred_lya_model(frow.get("z_lya", np.nan),
                                   drow.get("double_peak", np.nan) if drow is not None else np.nan)
    res["shape_model"] = model
    if model == "":
        res["shape_status"] = "no_fit"
        return res, None
    ok_single = all(np.isfinite(float(frow.get(k, np.nan))) for k in SINGLE_PARS)
    if not ok_single:
        res["shape_status"] = "no_fit_params"
        return res, None
    d = np.load(path, allow_pickle=True)
    wave, flux, var = (np.asarray(d[k], float) for k in ("wave", "flux", "var"))
    good = np.isfinite(flux) & np.isfinite(var) & (var > 0)
    wave, flux, var = wave[good], flux[good], var[good]

    single = [float(frow[k]) for k in SINGLE_PARS[:4]]
    s_lsf = float(frow["sigma_lsf_ang"])
    pars = single
    if model == "double":
        dp = [float(drow.get(c, np.nan)) for c in fdp.PAR_COLS]
        s_d = float(drow.get("sigma_lsf_ang", np.nan))
        if not (np.all(np.isfinite(dp)) and np.isfinite(s_d)):
            res["shape_status"] = "no_double_params"
            return res, None
        pars, s_lsf = dp, s_d

    lam_sys = float(pc.lya_obs_air(z_sys))
    lo, hi = window(lam_sys, a.vmin, a.vmax)
    best = dict(
        int=model_fblue(model, pars, s_lsf, lam_sys, lo, hi, False),
        conv=model_fblue(model, pars, s_lsf, lam_sys, lo, hi, True),
        int_single=model_fblue("single", single, float(frow["sigma_lsf_ang"]),
                               lam_sys, lo, hi, False))
    fb_d, fr_d = data_sums(wave, flux, lam_sys, lo, hi)
    eb_d, er_d = data_sum_errors(wave, var, lam_sys, lo, hi)
    tot = fb_d + fr_d
    best["data"] = fb_d / tot if tot > 0 else np.nan
    res.update(flux_blue_data=fb_d, flux_blue_data_err=eb_d,
               flux_red_data=fr_d, flux_red_data_err=er_d)

    # MC
    draws = {k: [] for k in best}
    if a.n_mc > 0:
        centre = float(frow.get("best_center", np.nan))
        if not np.isfinite(centre):
            centre = lam_sys
        sel = (wave >= centre - a.fit_window) & (wave <= centre + a.fit_window)
        w, e = wave[sel], np.sqrt(var[sel])
        base = model_flux(model, w, pars, s_lsf)
        base_s = fdp.skew_lsf(w, *single, float(frow["sigma_lsf_ang"]))
        sig_pix = np.sqrt(var)
        zs = rng.normal(z_sys, z_err, a.n_mc) if np.isfinite(z_err) and z_err > 0 \
            else np.full(a.n_mc, z_sys)
        for zd in zs:
            ls = float(pc.lya_obs_air(zd))
            l0, l1 = window(ls, a.vmin, a.vmax)
            p = refit(model, w, base + rng.normal(0, e), e, s_lsf, centre, lam_sys, pars)
            if p is not None:
                draws["int"].append(model_fblue(model, p, s_lsf, ls, l0, l1, False))
                draws["conv"].append(model_fblue(model, p, s_lsf, ls, l0, l1, True))
            if model == "single":
                if p is not None:
                    draws["int_single"].append(draws["int"][-1])
            else:
                ps = refit_single(w, base_s + rng.normal(0, e), e,
                                  float(frow["sigma_lsf_ang"]), centre, single)
                if ps is not None:
                    draws["int_single"].append(model_fblue(
                        "single", ps, float(frow["sigma_lsf_ang"]), ls, l0, l1, False))
            b, r = data_sums(wave, flux + rng.normal(0, sig_pix), ls, l0, l1)
            draws["data"].append(b / (b + r) if (b + r) > 0 else np.nan)
        res["n_mc_success"] = len(draws["int"])

    for k in KINDS:
        v, elo, ehi, e = summarise(best[k], draws[k])
        res.update({f"f_blue_{k}": v, f"f_blue_{k}_err_lo": elo, f"f_blue_{k}_err_hi": ehi,
                    f"f_blue_{k}_err": e})
    v, _, _, e = summarise(best["int_single"], draws["int_single"])
    res.update(f_blue_int_single=v, f_blue_int_single_err=e)
    plot = dict(wave=wave, flux=flux, var=var, lam_sys=lam_sys, lo=lo, hi=hi,
                model=model, pars=pars, s_lsf=s_lsf)
    return res, plot


def plot_source(path, sid, res, p):
    lam = p["lam_sys"]
    v = C_KMS * (p["wave"] / lam - 1)
    m = (v > res["vmin_kms"] - 500) & (v < res["vmax_kms"] + 500)
    vv = np.linspace(v[m].min(), v[m].max(), 1500)
    xx = lam * (1 + vv / C_KMS)
    f_conv = model_flux(p["model"], xx, p["pars"], p["s_lsf"])
    f_int = model_flux(p["model"], xx, p["pars"], 1e-6)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axvspan(res["vmin_kms"], 0, color="royalblue", alpha=0.08, lw=0)
    ax.axvspan(0, res["vmax_kms"], color="firebrick", alpha=0.05, lw=0)
    ax.step(v[m], p["flux"][m], where="mid", color="k", lw=0.8, alpha=0.7, label="data")
    sd = np.sqrt(p["var"][m])
    ax.fill_between(v[m], -sd, sd, step="mid", color="0.6", alpha=0.3, lw=0)
    ax.plot(vv, f_conv, color="crimson", lw=1.5, label=f"{p['model']} model, with LSF")
    ax.plot(vv, f_int, color="crimson", lw=1.0, ls=":", label="intrinsic")
    ax.axvline(0, color="orange", lw=1.5, ls="--", label=r"$z_{\rm sys}$")
    ax.axhline(0, color="k", lw=0.6, ls="--")

    def fmt(k):
        val, e = res[f"f_blue_{k}"], res.get(f"f_blue_{k}_err", np.nan)
        return f"{val:.2f}" + (rf" $\pm$ {e:.2f}" if np.isfinite(e) else "")
    ax.text(0.02, 0.96, "\n".join([rf"$f_{{\rm blue}}$ int {fmt('int')}",
                                   rf"$f_{{\rm blue}}$ conv {fmt('conv')}",
                                   rf"$f_{{\rm blue}}$ data {fmt('data')}"]),
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_xlim(vv.min(), vv.max())
    ax.set_xlabel(r"Velocity from systemic [km s$^{-1}$]")
    ax.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.set_title(f"{sid}  S/N {res['lya_snr']:.1f}  window {res['vmin_kms']:.0f} to "
                 f"{res['vmax_kms']:+.0f} km/s", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    a = parse_args()
    print("measure_lya_shape.py")
    for p_ in (a.zsys, a.fits_csv):
        print(f"  reading  {os.path.abspath(p_)}")
    has_dbl = os.path.exists(a.double_csv)
    print(f"  reading  {os.path.abspath(a.double_csv)}"
          + ("" if has_dbl else "  (not found, every source uses the single fit)"))
    print(f"  reading  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_lya_spectrum.npz")
    print(f"  writing  {os.path.abspath(a.out_csv)}")
    if not a.no_plots:
        print(f"  writing  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_KIND}")
    print(f"  f_blue over {a.vmin:+.0f} to {a.vmax:+.0f} km/s about systemic, {a.n_mc} MC draws\n")

    sample = pc.load_sample(a.zsys, ids=a.ids, need=("ID", "z_sys"))
    fits_df = pd.read_csv(a.fits_csv)
    fits_df["ID"] = fits_df["ID"].astype(int)
    fits_df = fits_df.set_index("ID")
    dbl_df = None
    if has_dbl:
        dbl_df = pd.read_csv(a.double_csv)
        dbl_df["ID"] = dbl_df["ID"].astype(int)
        dbl_df = dbl_df.set_index("ID")
    rng = np.random.default_rng(a.seed)

    rows = []
    for _, s in sample.iterrows():
        sid = int(s["ID"])
        if sid not in fits_df.index:
            print(f"  {sid:>6d}  not in {os.path.basename(a.fits_csv)}, skipped")
            continue
        zerr = float(s["z_sys_err"]) if "z_sys_err" in s and pd.notna(s["z_sys_err"]) else np.nan
        drow = dbl_df.loc[sid] if dbl_df is not None and sid in dbl_df.index else None
        res, plot = run_source(sid, float(s["z_sys"]), zerr, fits_df.loc[sid], drow, a, rng)
        rows.append(res)
        if res["shape_status"] != "ok":
            print(f"  {sid:>6d}  {res['shape_status']}")
            continue
        print(f"  {sid:>6d}  S/N {res['lya_snr']:5.1f}  {res['shape_model']:<6s}  f_blue int "
              f"{res['f_blue_int']:.3f} +/- {res['f_blue_int_err']:.3f}  conv "
              f"{res['f_blue_conv']:.3f}  data {res['f_blue_data']:+.3f} +/- "
              f"{res['f_blue_data_err']:.3f}", flush=True)
        if plot is not None and not a.no_plots:
            png = f"{os.path.abspath(a.subcube_dir)}/{sid}/{sid}_{FIG_KIND}"
            os.makedirs(os.path.dirname(png), exist_ok=True)
            plot_source(png, sid, res, plot)

    cols = (["ID", "z_sys", "lya_snr", "shape_model"]
            + [f"f_blue_{k}{s}" for k in KINDS for s in ("", "_err_lo", "_err_hi", "_err")]
            + ["f_blue_int_single", "f_blue_int_single_err", "flux_blue_data",
               "flux_blue_data_err", "flux_red_data", "flux_red_data_err",
               "vmin_kms", "vmax_kms", "n_mc_success", "shape_status"])
    df = pd.DataFrame(rows).reindex(columns=cols)
    out = os.path.abspath(a.out_csv)
    if a.ids and os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old[~old["ID"].astype(int).isin(df["ID"])], df], ignore_index=True)
    df = df.sort_values("ID").reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    ok = df["shape_status"] == "ok"
    print(f"\n  measured {int(ok.sum())} of {len(df)}")
    for m in ("single", "double"):
        sub = df[ok & (df["shape_model"] == m)]
        if len(sub):
            print(f"    {m:<6s} {len(sub):>3d}  median f_blue int {sub['f_blue_int'].median():.3f}, "
                  f"conv {sub['f_blue_conv'].median():.3f}, data {sub['f_blue_data'].median():.3f}")
    print(f"\nWrote  {out}")
    if not a.no_plots:
        print(f"Wrote  {os.path.abspath(a.subcube_dir)}/<ID>/<ID>_{FIG_KIND}")


if __name__ == "__main__":
    main()