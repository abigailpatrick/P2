#!/usr/bin/env python3
"""
Sample A checks for the JELS-MUSE NIRSpec proposal (Ken's email, 30 Sep 2026).

Sample A is the manual group a from group_by_manual.py (lya_group_a.csv),
the sources with a clear Lya detection and a good systemic line. Redshifts
are the best z_sys from delta_v_from_best_zsys_line.csv.

Printed, in order
-----------------
  [0] Sample funnel, from the grating catalogue down to sample A
  [a] Counts per grating and per proposal redshift bin
  [b] Systemic redshift precision, medium vs high
  [c] Hbeta counts
  [d] Systemic error vs systemic line S/N, scaled to fainter lines
  [SUMMARY] every number needed for the reply, in one block

a) Redshift distribution
     fig_a_zbins_proposal.png   counts in the proposal redshift bins
     fig_a_zhist_overlaid.png   fine z histogram, four gratings overlaid
     fig_a_zhist_M_vs_H.png     medium and high gratings in separate panels

b) Systemic redshift precision
   delta_v_err_sys_kms = c z_sys_err / (1 + z_sys), the LiMe line-fit error
   on the z_sys used for each source's Delta_v. Split by whether that z_sys
   came from a medium (G235M, G395M) or high (G235H, G395H) grating, with
   the median of each marked.
     fig_b_zsys_err_M_vs_H.png

   These are formal least-squares errors from lmfit (statistical only).

   As a like-for-like check the script also measures every source in every
   grating it has (same line priority as find_delta_v_from_best_zsys_line.py)
   and prints the medians. This is written to sampleA_zsys_err_by_grating.csv
   but not plotted.

c) Hbeta, printed only
   Counts with LiMe snr_line >= 3 and >= 5 per grating.

d) Systemic error against systemic line S/N
   Checks that the LiMe error scales as k/(S/N), fits k separately for
   medium and high, and extrapolates to fainter lines (S/N 3-10 shaded).
     fig_d_zsys_err_vs_snr.png

Outputs (all to --outdir)
-------------------------
  sampleA_sources.csv               one row per source
  sampleA_by_grating.csv            counts per grating
  sampleA_by_zbin.csv               counts per proposal z bin per grating
  sampleA_zsys_err_by_grating.csv   like-for-like per-grating errors
  plus the four figures above

Usage
-----
python proposal_sample_checks.py
python proposal_sample_checks.py --outdir /some/other/dir
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cmasher as cmr
    CMAP = cmr.torch
except ImportError:
    CMAP = plt.get_cmap("viridis")

C_KMS = 299792.458
GRATINGS = ["G235M", "G235H", "G395M", "G395H"]
MEDIUM = ["G235M", "G395M"]
HIGH = ["G235H", "G395H"]
COLOURS = {g: CMAP(x) for g, x in zip(GRATINGS, (0.15, 0.4, 0.62, 0.85))}
COL_M, COL_H = CMAP(0.2), CMAP(0.7)
LINE_PRIORITY = ["OIII", "Ha", "Hbeta", "NII", "OII"]
TIERS = ["gold", "silver", "bronze", "stone", "bad"]

# Proposal bins plus a low-z bin below the G395 Halpha blue edge
Z_EDGES = [2.9, 3.373, 4.0, 4.5, 5.0, 5.5, 6.0, 6.6]
Z_LOW = 4.0                               # "low redshift" for the summary
Z_HA_EDGE = 3.373                         # Halpha enters G395
Z_HB_EDGE = 4.90                          # Hbeta enters G395
Z_HIST_BINS = np.arange(2.8, 6.01, 0.1)
ERR_BINS = np.arange(0, 21, 1.0)          # km/s


def zbin_labels():
    return [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:])]


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else np.nan


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save(fig, path, written):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(path)


def write_csv(df, path, written):
    df.to_csv(path, index=False)
    written.append(path)


# ----------------------------------------------------------------------
# [0] Funnel
# ----------------------------------------------------------------------
def funnel(gcat, musedir, n_a):
    header("[0] SAMPLE FUNNEL")
    n_cat = len(gcat)
    n_muse = int((gcat["in_muse"] == 1).sum())
    n_unique = 0
    for t in TIERS:
        p = os.path.join(musedir, f"lya_group_{t}.csv")
        if os.path.exists(p):
            n_unique += len(pd.read_csv(p))
    labels = {}
    for lab in "abcd":
        p = os.path.join(musedir, f"lya_group_{lab}.csv")
        labels[lab] = len(pd.read_csv(p)) if os.path.exists(p) else 0
    print(f"  JELS sources with a DJA grating spectrum     {n_cat}")
    print(f"  of which inside the MUSE footprint           {n_muse}")
    print(f"  unique after collapsing duplicates           {n_unique}  (sum of tier CSVs)")
    print(f"  manual groups a / b / c / d                  "
          + " / ".join(str(labels[l]) for l in "abcd"))
    print(f"  sample A (clear Lya + good systemic line)    {n_a}")
    return {"n_cat": n_cat, "n_muse": n_muse, "n_unique": n_unique}


# ----------------------------------------------------------------------
# [a] Sample by grating and redshift
# ----------------------------------------------------------------------
def task_a(src, outdir, written):
    header("[a] SAMPLE A BY GRATING AND REDSHIFT")
    n = len(src)
    has_m = src[MEDIUM].any(axis=1)
    has_h = src[HIGH].any(axis=1)
    out = {
        "n": n, "any_M": int(has_m.sum()), "any_H": int(has_h.sum()),
        "M_only": int((has_m & ~has_h).sum()), "H_only": int((~has_m & has_h).sum()),
        "both": int((has_m & has_h).sum()),
        "z_low": int((src.z_sys < Z_LOW).sum()), "z_high": int((src.z_sys >= Z_LOW).sum()),
        "z_below_Ha": int((src.z_sys < Z_HA_EDGE).sum()),
    }
    print(f"  N sources                     {n}")
    print(f"  with a medium / high spectrum {out['any_M']} / {out['any_H']}")
    print(f"  M only / H only / both        {out['M_only']} / {out['H_only']} / {out['both']}")
    print(f"  z_sys range                   {src.z_sys.min():.2f} - {src.z_sys.max():.2f}, "
          f"median {src.z_sys.median():.2f}")
    print(f"  z < {Z_HA_EDGE} (no G395 Halpha)  {out['z_below_Ha']}")
    print(f"  z < {Z_LOW:g} / z >= {Z_LOW:g}              {out['z_low']} / {out['z_high']}")
    print(f"  z_sys line used               "
          + ", ".join(f"{k} {v}" for k, v in src.z_sys_line.value_counts().items()))

    rows = []
    for g in GRATINGS + ["any M", "any H"]:
        m = has_m if g == "any M" else has_h if g == "any H" else (src[g] == 1)
        s = src[m]
        rows.append({
            "grating": g, "N": int(m.sum()),
            "z_min": round(s.z_sys.min(), 2), "z_median": round(s.z_sys.median(), 2),
            "z_max": round(s.z_sys.max(), 2),
            f"N_z<{Z_LOW:g}": int((s.z_sys < Z_LOW).sum()),
            f"N_z>={Z_LOW:g}": int((s.z_sys >= Z_LOW).sum()),
        })
    grat = pd.DataFrame(rows)
    write_csv(grat, os.path.join(outdir, "sampleA_by_grating.csv"), written)
    print("\n  Per grating (a source can appear in more than one):")
    print(grat.to_string(index=False))

    zb = pd.cut(src.z_sys, bins=Z_EDGES, labels=zbin_labels(), right=False)
    tab = pd.DataFrame({"zbin": zbin_labels()})
    tab["N_all"] = [int((zb == l).sum()) for l in zbin_labels()]
    for g in GRATINGS:
        tab[f"N_{g}"] = [int(((zb == l) & (src[g] == 1)).sum()) for l in zbin_labels()]
    tab["N_any_M"] = [int(((zb == l) & has_m).sum()) for l in zbin_labels()]
    tab["N_any_H"] = [int(((zb == l) & has_h).sum()) for l in zbin_labels()]
    write_csv(tab, os.path.join(outdir, "sampleA_by_zbin.csv"), written)
    print("\n  Per proposal redshift bin:")
    print(tab.to_string(index=False))
    out["zbins"] = tab

    # Proposal bins, all sources
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(tab))
    ax.bar(x, tab.N_all, width=0.8, color=CMAP(0.35))
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("-", "–") for l in zbin_labels()], fontsize=8)
    ax.set_xlabel(r"$z_{\rm sys}$ bin")
    ax.set_ylabel("N")
    ax.set_ylim(0, tab.N_all.max() * 1.1 + 1)
    ax.yaxis.get_major_locator().set_params(integer=True)
    save(fig, os.path.join(outdir, "fig_a_zbins_proposal.png"), written)

    # Fine histogram, gratings overlaid
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(src.z_sys, bins=Z_HIST_BINS, histtype="stepfilled", color="0.85",
            zorder=0, label=f"sample A (N={n})")
    for g in GRATINGS:
        z = src.loc[src[g] == 1, "z_sys"]
        ax.hist(z, bins=Z_HIST_BINS, histtype="step", lw=1.8, color=COLOURS[g],
                label=f"{g} (N={len(z)})")
    ax.axvline(Z_HA_EDGE, color="k", ls="--", lw=1,
               label=rf"G395 H$\alpha$ edge ($z={Z_HA_EDGE}$)")
    ax.set_xlabel(r"$z_{\rm sys}$")
    ax.set_ylabel("N")
    ax.legend(frameon=False, fontsize=8)
    save(fig, os.path.join(outdir, "fig_a_zhist_overlaid.png"), written)

    # Medium vs high panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True, sharey=True)
    for ax, grp, name, nn in zip(axes, (MEDIUM, HIGH),
                                 ("Medium (R~1000)", "High (R~2700)"),
                                 (out["any_M"], out["any_H"])):
        zs = [src.loc[src[g] == 1, "z_sys"] for g in grp]
        ax.hist(zs, bins=Z_HIST_BINS, stacked=True, color=[COLOURS[g] for g in grp],
                label=[f"{g} (N={len(z)})" for g, z in zip(grp, zs)])
        ax.axvline(Z_HA_EDGE, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$z_{\rm sys}$")
        ax.legend(frameon=False, fontsize=8, title=f"{name}, {nn} sources",
                  title_fontsize=8)
    axes[0].set_ylabel("N")
    fig.tight_layout()
    save(fig, os.path.join(outdir, "fig_a_zhist_M_vs_H.png"), written)
    return out


# ----------------------------------------------------------------------
# [b] Systemic precision, medium vs high
# ----------------------------------------------------------------------
def load_line_tables(catdir):
    tabs = {}
    for line in LINE_PRIORITY:
        res = pd.read_csv(os.path.join(catdir, f"{line}_results_by_JELS_ID.csv"))
        summ = pd.read_csv(os.path.join(catdir, f"{line}_summary_by_JELS_ID.csv"))
        df = res.merge(summ, on="ID", how="left")
        df["ID"] = df["ID"].astype(int)
        tabs[line] = df.set_index("ID")
    return tabs


def zsys_in_grating(tabs, sid, g):
    """First successful line in grating g, in LINE_PRIORITY order."""
    for line in LINE_PRIORITY:
        t = tabs[line]
        if sid not in t.index or t.loc[sid].get(f"{line}_{g}_success") != True:
            continue
        r = t.loc[sid]
        z, ze = r.get(f"z_{line}_{g}"), r.get(f"z_{line}_{g}_err")
        if pd.notna(z) and pd.notna(ze) and np.isfinite(ze):
            return line, float(z), float(ze)
    return None


def task_b(catdir, systemic_path, src, outdir, written):
    header("[b] SYSTEMIC REDSHIFT PRECISION, MEDIUM VS HIGH")

    # Grating of the z_sys used in the Delta_v
    sysd = pd.read_csv(systemic_path)
    sysd["ID"] = sysd["ID"].astype(int)
    sysd = sysd.set_index("ID")
    used = []
    for _, s in src.iterrows():
        col = f"z_{s.z_sys_line}_grating"
        used.append(sysd.loc[int(s.ID), col]
                    if int(s.ID) in sysd.index and col in sysd.columns else np.nan)
    src = src.assign(zsys_grating=used)
    src["zsys_res"] = np.where(src.zsys_grating.isin(MEDIUM), "M",
                               np.where(src.zsys_grating.isin(HIGH), "H", ""))

    err_m = src.loc[src.zsys_res == "M", "delta_v_err_sys_kms"].dropna()
    err_h = src.loc[src.zsys_res == "H", "delta_v_err_sys_kms"].dropna()
    out = {
        "N_M": len(err_m), "N_H": len(err_h),
        "med_M": pct(err_m, 50), "med_H": pct(err_h, 50),
        "mean_M": float(err_m.mean()), "mean_H": float(err_h.mean()),
    }
    print("  delta_v_err_sys_kms for the z_sys used in Delta_v (LiMe fit error)")
    print(f"  {'':<8}{'N':>4}{'median':>9}{'mean':>8}{'p16':>7}{'p84':>7}{'max':>7}  km/s")
    for lab, v in (("medium", err_m), ("high", err_h),
                   ("all", src.delta_v_err_sys_kms.dropna())):
        print(f"  {lab:<8}{len(v):>4}{pct(v, 50):>9.1f}{v.mean():>8.1f}"
              f"{pct(v, 16):>7.1f}{pct(v, 84):>7.1f}{v.max():>7.1f}")
    print("  z_sys grating used: " + ", ".join(
        f"{g} {int((src.zsys_grating == g).sum())}" for g in GRATINGS))
    print(f"\n  For scale: Delta_v median {src.delta_v_kms.median():.0f} km/s "
          f"(16-84th {pct(src.delta_v_kms, 16):.0f}-{pct(src.delta_v_kms, 84):.0f}), "
          f"total Delta_v error median {src.delta_v_err_kms.median():.0f} km/s, "
          f"Lya term {src.delta_v_err_lya_kms.median():.0f} km/s")
    out.update({"dv_med": float(src.delta_v_kms.median()),
                "dv_err_med": float(src.delta_v_err_kms.median()),
                "dv_err_lya_med": float(src.delta_v_err_lya_kms.median())})

    # Like-for-like: every source in every grating it has
    tabs = load_line_tables(catdir)
    rows = []
    for _, s in src.iterrows():
        for g in GRATINGS:
            if s[g] != 1:
                continue
            hit = zsys_in_grating(tabs, int(s.ID), g)
            row = {"ID": int(s.ID), "grating": g}
            if hit:
                line, z, ze = hit
                row.update({"line": line, "z": z, "z_err": ze,
                            "sigma_v_sys_kms": C_KMS * ze / (1 + z)})
            rows.append(row)
    per = pd.DataFrame(rows)
    write_csv(per, os.path.join(outdir, "sampleA_zsys_err_by_grating.csv"), written)
    print("\n  Like-for-like check, each source measured in every grating it has:")
    for name, grp in (("medium", MEDIUM), ("high", HIGH)):
        v = per.loc[per.grating.isin(grp), "sigma_v_sys_kms"].dropna()
        print(f"  {name:<8} N={len(v):>3}  median {pct(v, 50):.1f} km/s")

    # Figure
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    top = ERR_BINS[-1] - 1e-6
    for v, col, lab, med in ((err_m, COL_M, "Medium", out["med_M"]),
                             (err_h, COL_H, "High", out["med_H"])):
        ax.hist(np.clip(v, 0, top), bins=ERR_BINS, histtype="stepfilled",
                color=col, alpha=0.35)
        ax.hist(np.clip(v, 0, top), bins=ERR_BINS, histtype="step", lw=2, color=col,
                label=f"{lab} (N={len(v)}), median {med:.1f} km/s")
        ax.axvline(med, color=col, ls="--", lw=1.8)
    ax.set_xlabel(r"Systemic redshift error (km s$^{-1}$)")
    ax.set_ylabel("N")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(outdir, "fig_b_zsys_err_M_vs_H.png"), written)
    return out, src


# ----------------------------------------------------------------------
# [c] Hbeta
# ----------------------------------------------------------------------
def task_c(catdir, src):
    header("[c] HBETA IN SAMPLE A (LiMe snr_line, amplitude-based)")
    hb = pd.read_csv(os.path.join(catdir, "Hbeta_results_by_JELS_ID.csv"))
    ha = pd.read_csv(os.path.join(catdir, "Ha_results_by_JELS_ID.csv"))
    hb["ID"] = hb["ID"].astype(int)
    ha["ID"] = ha["ID"].astype(int)
    ids = src.ID.tolist()
    s_hb = hb.set_index("ID").reindex(ids)[[f"z_Hbeta_{g}_snr" for g in GRATINGS]]
    s_ha = ha.set_index("ID").reindex(ids)[[f"z_Ha_{g}_snr" for g in GRATINGS]]
    s_hb.columns = s_ha.columns = GRATINGS

    print(f"  {'grating':<8}{'has grating':>12}{'Hb covered':>12}{'snr>=3':>8}{'snr>=5':>8}")
    for g in GRATINGS:
        print(f"  {g:<8}{int(src[g].sum()):>12}{int(s_hb[g].notna().sum()):>12}"
              f"{int((s_hb[g] >= 3).sum()):>8}{int((s_hb[g] >= 5).sum()):>8}")
    best = s_hb.max(axis=1).values
    z = src.z_sys.values
    out = {
        "hb3": int((best >= 3).sum()), "hb5": int((best >= 5).sum()),
        "hb3_M": int((s_hb[MEDIUM].max(axis=1) >= 3).sum()),
        "hb3_H": int((s_hb[HIGH].max(axis=1) >= 3).sum()),
        "hb3_G395": int((s_hb[["G395M", "G395H"]].max(axis=1) >= 3).sum()),
        "n_above_hb_edge": int((z >= Z_HB_EDGE).sum()),
        "balmer": int(((s_hb.max(axis=1) >= 3) & (s_ha.max(axis=1) >= 3)).sum()),
    }
    print(f"  any grating, snr>=3 / >=5      {out['hb3']} / {out['hb5']} of {len(ids)}")
    print(f"  any M / any H, snr>=3          {out['hb3_M']} / {out['hb3_H']}")
    print(f"  in G395 (M or H), snr>=3       {out['hb3_G395']}")
    print(f"  sources at z >= {Z_HB_EDGE} (Hbeta in G395)  {out['n_above_hb_edge']}")
    print(f"  Halpha and Hbeta both snr>=3   {out['balmer']} of {len(ids)}")
    return out


# ----------------------------------------------------------------------
# [d] Systemic error vs systemic line S/N, and the scaling to fainter lines
# ----------------------------------------------------------------------
def task_d(src, outdir, written):
    header("[d] SYSTEMIC ERROR VS SYSTEMIC LINE S/N")
    d = src.dropna(subset=["z_sys_snr", "delta_v_err_sys_kms"]).copy()
    d["k"] = d.delta_v_err_sys_kms * d.z_sys_snr
    out = {}
    print("  Error scales as k / (S/N). k = median(error x S/N) per resolution.")
    print(f"  {'':<8}{'N':>4}{'k':>8}{'rho':>7}{'p':>7}   predicted error at S/N = 3, 5, 10, 20")
    for res, name in (("M", "medium"), ("H", "high")):
        g = d[d.zsys_res == res]
        k = float(g.k.median())
        rho, pval = spearmanr(g.z_sys_snr, g.delta_v_err_sys_kms)
        pred = [k / x for x in (3, 5, 10, 20)]
        out[res] = {"k": k, "N": len(g), "pred": pred,
                    "snr_med": float(g.z_sys_snr.median())}
        print(f"  {name:<8}{len(g):>4}{k:>8.0f}{rho:>7.2f}{pval:>7.3f}   "
              + ", ".join(f"{v:.0f}" for v in pred) + " km/s")
    print(f"  (rho, p: Spearman between systemic S/N and error; negative = error falls with S/N)")
    for line in ("OIII", "Ha"):
        g = d[d.z_sys_line == line]
        if len(g):
            print(f"  {line:<5} N={len(g):>2}  median systemic S/N {g.z_sys_snr.median():.0f}")
    out["snr_min"] = float(d.z_sys_snr.min())
    out["dv_med"] = float(src.delta_v_kms.median())

    # For reference: Lya S/N does not predict the systemic line S/N
    rho_l, p_l = spearmanr(d.lya_snr, d.z_sys_snr)
    out["rho_lya"], out["p_lya"] = float(rho_l), float(p_l)
    print(f"  For reference, Spearman(Lya S/N, systemic S/N) = {rho_l:.2f}, p = {p_l:.2f}")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    marks = {"OIII": "o", "Ha": "^"}
    names = {"OIII": "[OIII]", "Ha": r"H$\alpha$"}
    snr_grid = np.logspace(np.log10(2), np.log10(d.z_sys_snr.max() * 1.3), 200)
    ax.axvspan(3, 10, color="0.9", zorder=0)
    for res, col, name in (("M", COL_M, "Medium"), ("H", COL_H, "High")):
        for line, mk in marks.items():
            g = d[(d.zsys_res == res) & (d.z_sys_line == line)]
            if g.empty:
                continue
            ax.scatter(g.z_sys_snr, g.delta_v_err_sys_kms, marker=mk, s=30, color=col,
                       edgecolor="k", lw=0.3, zorder=3,
                       label=f"{names[line]}, {name.lower()} (N={len(g)})")
        k = out[res]["k"]
        ax.plot(snr_grid, k / snr_grid, color=col, lw=1.5, ls="--",
                label=f"{name}: {k:.0f} / (S/N)")
    ax.axhline(out["dv_med"], color="k", lw=1, ls=":",
               label=rf"median Ly$\alpha$ $\Delta v$ ({out['dv_med']:.0f} km s$^{{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, snr_grid[-1])
    ax.set_xlabel("Systemic line S/N")
    ax.set_ylabel(r"Systemic redshift error (km s$^{-1}$)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, os.path.join(outdir, "fig_d_zsys_err_vs_snr.png"), written)
    return out


# ----------------------------------------------------------------------
def summary(f, a, b, c, d):
    header("[SUMMARY] numbers for the reply")
    print(f"  Of {f['n_unique']} unique grating sources in the MUSE field "
          f"({f['n_cat']} in the catalogue, {f['n_muse']} in the footprint), "
          f"{a['n']} are clear Lya detections with a good systemic line.")
    print(f"  Redshift: {a['z_low']} at z < {Z_LOW:g} and {a['z_high']} at z >= {Z_LOW:g} "
          f"({a['z_below_Ha']} below the G395 Halpha edge at z = {Z_HA_EDGE}).")
    print("  Proposal bins: " + ", ".join(
        f"{r.zbin}: {r.N_all}" for r in a["zbins"].itertuples()))
    print(f"  Systemic precision (median LiMe fit error): {b['med_M']:.1f} km/s medium "
          f"(N={b['N_M']}), {b['med_H']:.1f} km/s high (N={b['N_H']}).")
    print(f"  Sources with any medium / high spectrum: {a['any_M']} / {a['any_H']} "
          f"({a['both']} have both).")
    print(f"  For scale: median Delta_v {b['dv_med']:.0f} km/s, median total Delta_v error "
          f"{b['dv_err_med']:.0f} km/s, dominated by Lya ({b['dv_err_lya_med']:.0f} km/s).")
    print(f"  Hbeta snr>=3 in {c['hb3']} of {a['n']} (snr>=5 in {c['hb5']}), "
          f"{c['hb3_G395']} of them in G395. Only {c['n_above_hb_edge']} sources are at "
          f"z >= {Z_HB_EDGE} where Hbeta falls in G395.")
    print(f"  Halpha and Hbeta both snr>=3 in {c['balmer']} of {a['n']}.")
    for res, name in (("M", "medium"), ("H", "high")):
        o = d[res]
        print(f"  Systemic error scaling, {name}: {o['k']:.0f}/(S/N) km/s from N={o['N']} "
              f"(median S/N {o['snr_med']:.0f}). At S/N 3/5/10/20: "
              + " / ".join(f"{v:.0f}" for v in o["pred"]) + " km/s.")
    print(f"  Lowest systemic S/N in sample A: {d['snr_min']:.0f}, so S/N < ~15 is extrapolation.")


def main():
    p = argparse.ArgumentParser(description="Sample A checks for the NIRSpec proposal.")
    p.add_argument("--base", default="/ceph/cephfs/apatrick/P2")
    p.add_argument("--outdir", default=None, help="Default <base>/proposal_checks")
    args = p.parse_args()

    base = os.path.abspath(args.base)
    catdir = os.path.join(base, "jwst_catalogs")
    musedir = os.path.join(base, "MUSE_catalogs")
    outdir = os.path.abspath(args.outdir or os.path.join(base, "proposal_checks"))
    os.makedirs(outdir, exist_ok=True)

    path_a = os.path.join(musedir, "lya_group_a.csv")
    path_dv = os.path.join(musedir, "delta_v_from_best_zsys_line.csv")
    path_g = os.path.join(catdir, "grating_sources_by_JELS_ID.csv")
    path_sys = os.path.join(catdir, "systemic_redshifts_by_JELS_ID.csv")
    print("[CONFIG]")
    for k, v in (("sample A", path_a), ("delta_v", path_dv), ("gratings", path_g),
                 ("systemic", path_sys),
                 ("line csvs", f"{catdir}/<line>_results|summary_by_JELS_ID.csv"),
                 ("output dir", outdir)):
        print(f"  {k:<11} {v}")

    ids_a = set(pd.read_csv(path_a)["ID"].astype(int))
    dv = pd.read_csv(path_dv)
    dv["ID"] = dv["ID"].astype(int)
    gcat = pd.read_csv(path_g)
    gcat["ID"] = gcat["ID"].astype(int)

    src = (dv[dv.ID.isin(ids_a)]
           .merge(gcat[["ID"] + GRATINGS], on="ID", how="left")
           .sort_values("z_sys").reset_index(drop=True))
    src[GRATINGS] = src[GRATINGS].fillna(0).astype(int)
    missing = ids_a - set(src.ID)
    if missing:
        print(f"[WARN] sample A IDs missing from delta_v csv: {sorted(missing)}")

    written = []
    f = funnel(gcat, musedir, len(src))
    a = task_a(src, outdir, written)
    b, src = task_b(catdir, path_sys, src, outdir, written)
    c = task_c(catdir, src)
    d = task_d(src, outdir, written)
    write_csv(src, os.path.join(outdir, "sampleA_sources.csv"), written)
    summary(f, a, b, c, d)

    print("\n[DONE] files written:")
    for path in written:
        print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()