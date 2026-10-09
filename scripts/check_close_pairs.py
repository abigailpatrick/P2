#!/usr/bin/env python
"""Decide which Isaac object a shared NIRSpec spectrum belongs to.

For every group of Isaac objects that share one DJA spectrum (from
primer_minerva_close_pairs.csv) and have a z_sys, this runs four checks on
the spectrum that set z_sys and on Isaac's photometry, then writes one row
per object to a CSV and one PNG per group.

Checks
------
1. Position along the slit. The slit's long axis on the sky is at the
   position angle PA_APER in the spectrum header (confirmed by the nod
   offsets in the SLITS table: the telescope nods along it). Each object's
   offset from the MSA target is projected onto that axis and converted to a
   row of the 2D spectrum (target at row YTRACE, about 0.10 arcsec per row).
   The [OIII] or Halpha emission is then collapsed along wavelength and its
   centroid row measured. The object whose predicted row is closest to the
   line centroid is where the line comes from.
     row_pred       predicted row of the object
     drow_line      predicted row minus the line centroid row
     d_along, d_perp  offset from the target along and across the slit,
                    arcsec. The open shutter is 0.20 arcsec wide across the
                    slit, and the target itself sits 0.20 x |SRCXPOS| from
                    the shutter centre, so |d_perp| above about 0.2 arcsec
                    is outside the shutter.
     in_shutter     1 if |d_perp| <= 0.10 + 0.20 |SRCXPOS| arcsec, so the
                    object can lie inside the open shutter across the slit
                    (generous, as the side of the target is not known)
     in_extraction  1 if the object's row falls inside msaexp's 1D
                    extraction rows (YMIN1D to YMAX1D around the trace)
2. Medium-band line excess. The MINERVA medium band containing [OIII] 5007
   or Halpha at z_sys (the broad band if no medium band holds it), compared
   with a continuum interpolated from the nearest bands on either side that
   contain no strong line. Done for the z_sys line (excess_*) and the other
   line (excess2_*). The emitting
   object shows an excess, an interloper does not.
     excess_band, excess_ratio (band / continuum), excess_snr
3. Spectrum against photometry. Mean f_nu of the spectrum through a
   line-free band it fully covers, divided by each object's aperture and
   total flux in that band. The spectrum (path-loss corrected by msaexp for
   a point source) should be roughly 1x the object in the shutter. A value
   far above 1 means that object is too faint to be the source on its own.
     flux_band, spec_flux_snr, spec_over_ap, spec_over_tot
                    (median of the spectrum in a line-free band inside the
                    grating's nominal range, not trusted if spec_flux_snr < 3)
4. Photo-z. dz_phot = (z_phot - z_sys) / (1 + z_sys) and sigma_z. An
   object with |dz_phot| well above its sigma_z is an interloper.

Assumption to check by eye: rows of the msaexp 2D spectrum increase in the
same sense as in the detector cutouts. The sign is fitted per spectrum from
how the trace moves between nods. The target's own trace must sit at
YTRACE in the PNG; if a second trace appears on the opposite side from
where the neighbour is drawn, rerun with --flip-slit.

Band edges are the half-power ranges of the filters, from the JWST and HST
documentation (top hats are enough to decide which band holds a line).

Inputs
------
  jwst_catalogs/primer_minerva_close_pairs.csv
  jwst_catalogs/primer_minerva_in_muse_zsys.csv
  jwst_catalogs/cosmos_primer_minerva_production.fits
  jwst_spectra_pm/<G>_<F>/<ID>_<G>_<F>_spectra.fits

Outputs
-------
  jwst_catalogs/close_pairs_checks.csv
  plots/close_pairs/group_<n>_<IDs>.png

Usage
-----
python check_close_pairs.py
python check_close_pairs.py --flip-slit
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"
GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}

# Half-power band edges, micron.
BANDS = {
    "f435w": (0.363, 0.486), "f606w": (0.472, 0.718), "f814w": (0.687, 0.957),
    "f090w": (0.795, 1.005), "f115w": (1.013, 1.282), "f150w": (1.331, 1.668),
    "f200w": (1.755, 2.226), "f277w": (2.416, 3.127), "f356w": (3.140, 3.980),
    "f444w": (3.880, 4.986),
    "f140m": (1.331, 1.479), "f162m": (1.542, 1.713), "f182m": (1.722, 1.968),
    "f210m": (1.992, 2.201), "f250m": (2.412, 2.595), "f300m": (2.831, 3.157),
    "f360m": (3.426, 3.814), "f410m": (3.866, 4.302), "f460m": (4.515, 4.747),
}
MEDIUM = [b for b in BANDS if b.endswith("m")]
# Strong lines that contaminate a band, rest frame micron.
STRONG = {"[OII]": 0.3728, "Hb": 0.4863, "[OIII]4959": 0.4960, "[OIII]5007": 0.5008,
          "Ha": 0.6565, "[NII]": 0.6585, "[SII]": 0.6725}
MAIN_LINE = {"OIII": 0.5008, "Ha": 0.6565}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default=f"{CAT_DIR}/primer_minerva_close_pairs.csv")
    p.add_argument("--zsys", default=f"{CAT_DIR}/primer_minerva_in_muse_zsys.csv")
    p.add_argument("--isaac", default=f"{CAT_DIR}/cosmos_primer_minerva_production.fits")
    p.add_argument("--spectra-root", default=f"{P2}/jwst_spectra_pm")
    p.add_argument("--out-csv", default=f"{CAT_DIR}/close_pairs_checks.csv")
    p.add_argument("--out-dir", default=f"{P2}/plots/close_pairs")
    p.add_argument("--flip-slit", action="store_true",
                   help="reverse the sense of the slit axis in the 2D spectrum")
    return p.parse_args()


def groups_from_pairs(pairs):
    """Connected groups of IDs that share a spectrum."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for a, b in zip(pairs["ID"], pairs["near_ID"]):
        parent[find(int(a))] = find(int(b))
    out = {}
    for x in list(parent):
        out.setdefault(find(x), set()).add(x)
    return [sorted(g) for g in sorted(out.values(), key=min)]


def contaminated(band, z):
    lo, hi = BANDS[band]
    return any(lo <= l * (1 + z) <= hi for l in STRONG.values())


def line_excess(row, z, line):
    """Band holding the line, its flux ratio to the continuum and the S/N.

    A medium band is used when one holds the line, otherwise the broad band
    (a smaller excess for the same line). The continuum is interpolated from
    the nearest bands on each side that hold no strong line at z.
    """
    lam = MAIN_LINE[line] * (1 + z)

    def ok(k):
        return (np.isfinite(row.get(k, np.nan)) and row[k] > -90
                and np.isfinite(row.get(f"{k}_err", np.nan)) and row[f"{k}_err"] > 0)
    hold = [b for b in MEDIUM if BANDS[b][0] <= lam <= BANDS[b][1] and ok(b)]
    if not hold:
        hold = [b for b in BANDS if not b.endswith("m") and BANDS[b][0] <= lam <= BANDS[b][1] and ok(b)]
    if not hold:
        return None, np.nan, np.nan
    b = hold[0]
    pivot = {k: np.mean(v) for k, v in BANDS.items()}
    free = [k for k in BANDS if not contaminated(k, z) and k != b and ok(k)
            and not (BANDS[k][0] < BANDS[b][1] and BANDS[b][0] < BANDS[k][1])]
    blue = sorted([k for k in free if pivot[k] < pivot[b]], key=lambda k: pivot[b] - pivot[k])
    red = sorted([k for k in free if pivot[k] > pivot[b]], key=lambda k: pivot[k] - pivot[b])
    if blue and red:
        k1, k2 = blue[0], red[0]
        w = (pivot[b] - pivot[k1]) / (pivot[k2] - pivot[k1])
        cont = (1 - w) * row[k1] + w * row[k2]
        cerr = np.hypot((1 - w) * row[f"{k1}_err"], w * row[f"{k2}_err"])
    elif blue or red:
        k1 = (blue or red)[0]
        cont, cerr = row[k1], row[f"{k1}_err"]
    else:
        return b, np.nan, np.nan
    ratio = row[b] / cont if cont > 0 else np.nan
    snr = (row[b] - cont) / np.hypot(row[f"{b}_err"], cerr)
    return b, ratio, snr


def slit_geometry(path, flip):
    """PA of the slit axis, rows per arcsec with sign, trace row, target, shutter info."""
    h = fits.getheader(path, "SCI")
    h1 = fits.getheader(path, "SPEC1D")
    s = Table.read(path, "SLITS").to_pandas()
    pa = float(h["PA_APER"])
    scale = float(np.nanmedian(s["slit_pixel_scale"]))
    kappa = 1.0 / scale
    # sign from how the trace moves when the telescope nods along the slit
    sign = 1.0
    try:
        u = np.array([np.sin(np.radians(pa)), np.cos(np.radians(pa))])
        det = s["detector"].astype(str).str.strip()
        d0 = det.iloc[0]
        ss = s[det == d0]
        cosd = np.cos(np.radians(ss["dec_v1"].iloc[0]))
        p = (((ss["ra_v1"] - ss["ra_v1"].iloc[0]) * cosd * u[0]
              + (ss["dec_v1"] - ss["dec_v1"].iloc[0]) * u[1]) * 3600).values
        if np.ptp(p) > 0.1:
            m = np.polyfit(p, ss["trace_c2"].values, 1)[0]
            sign = -np.sign(m)
    except Exception as e:
        print(f"    could not fit the slit sign ({e}), assuming +1")
    if flip:
        sign = -sign
    return {"pa": pa, "kappa": sign * kappa, "ytrace": float(h1.get("YTRACE", 15)),
            "ymin": float(h1.get("YMIN1D", -3)), "ymax": float(h1.get("YMAX1D", 3)),
            "ra0": float(h["SRCRA"]), "dec0": float(h["SRCDEC"]),
            "srcx": float(h.get("SRCXPOS", 0.0)), "srcy": float(h.get("SRCYPOS", 0.0)),
            "nshut": int(h.get("NUM_SHUTTERS", 1)), "scale": scale}


def offsets(ra, dec, g):
    """Offset of a sky position from the target along and across the slit, arcsec."""
    dx = (ra - g["ra0"]) * np.cos(np.radians(g["dec0"])) * 3600
    dy = (dec - g["dec0"]) * 3600
    a = np.radians(g["pa"])
    along = dx * np.sin(a) + dy * np.cos(a)
    perp = dx * np.cos(a) - dy * np.sin(a)
    return along, perp


def line_profile(path, z, line, ytrace):
    """Continuum-subtracted line profile and continuum profile along the slit."""
    sci = fits.getdata(path, "SCI")
    wave = Table.read(path, "SPEC1D")["wave"].data
    lam = MAIN_LINE[line] * (1 + z)
    on = np.abs(wave - lam) <= 0.0015 * (1 + z)            # +-15 AA rest
    side = (np.abs(wave - lam) > 0.0030 * (1 + z)) & (np.abs(wave - lam) <= 0.0060 * (1 + z))
    if on.sum() < 2 or side.sum() < 4:
        return None, None, np.nan, wave, sci, lam
    cont = np.nanmedian(sci[:, side], axis=1)
    prof = np.nansum(sci[:, on] - cont[:, None], axis=1)
    rows = np.arange(sci.shape[0])
    near = np.abs(rows - ytrace) <= 8
    w = np.clip(prof, 0, None) * near
    cen = float(np.sum(w * rows) / np.sum(w)) if np.sum(w) > 0 else np.nan
    return prof, cont, cen, wave, sci, lam


GRATING_RANGE = {"G235H": (1.66, 3.05), "G235M": (1.66, 3.07),
                 "G395H": (2.87, 5.14), "G395M": (2.87, 5.10)}


def synth_flux(path, band, grating):
    """Median f_nu (uJy) of the 1D spectrum through a top-hat band and its S/N.

    Only bands inside the grating's nominal range (msaexp extends some
    spectra beyond it, where second-order light can enter).
    """
    t = Table.read(path, "SPEC1D")
    w, f, e = np.asarray(t["wave"]), np.asarray(t["flux"]), np.asarray(t["err"])
    lo, hi = BANDS[band]
    glo, ghi = GRATING_RANGE[grating]
    if lo < glo or hi > ghi:
        return np.nan, np.nan
    sel = (w >= lo) & (w <= hi) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    if sel.sum() < 10 or (w[sel].max() - w[sel].min()) < 0.9 * (hi - lo):
        return np.nan, np.nan
    med = float(np.median(f[sel]))
    err = float(1.2533 * np.median(e[sel]) / np.sqrt(sel.sum()))
    return med, med / err


def main():
    a = parse_args()
    print("check_close_pairs.py")
    for p in (a.pairs, a.zsys, a.isaac):
        print(f"  reading  {os.path.abspath(p)}")
    pairs = pd.read_csv(a.pairs)
    zs = pd.read_csv(a.zsys).set_index("ID")
    isaac = Table.read(a.isaac).to_pandas().set_index("Number")
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for gi, ids in enumerate(groups_from_pairs(pairs), start=1):
        have = [i for i in ids if i in zs.index]
        if not have:
            print(f"\nGroup {gi} {ids}: no member has a z_sys, skipped")
            continue
        r0 = zs.loc[have[0]]
        z, line, g = float(r0["z_sys"]), r0["z_sys_line"], r0["z_sys_grating"]
        gf = f"{g}_{GRATING_FILTER[g]}"
        path = os.path.join(a.spectra_root, gf, f"{have[0]}_{gf}_spectra.fits")
        print(f"\nGroup {gi}  IDs {ids}  z_sys {z:.5f} from {line} {g}")
        print(f"  spectrum {path}")
        if not os.path.exists(path):
            print("  spectrum missing, skipped")
            continue
        geo = slit_geometry(path, a.flip_slit)
        prof, cont, cen, wave, sci, lam = line_profile(path, z, line, geo["ytrace"])
        print(f"  slit PA {geo['pa']:.1f} deg, {abs(geo['kappa']):.2f} rows/arcsec "
              f"(sign {np.sign(geo['kappa']):+.0f}), line centroid row {cen:.2f} "
              f"(trace row {geo['ytrace']:.0f})")

        # a line-free band the spectrum covers, for the flux check
        flux_band, f_spec, f_snr = None, np.nan, np.nan
        for b in sorted(BANDS, key=lambda k: (not k.endswith("m"), k)):
            if contaminated(b, z):
                continue
            fb, sn = synth_flux(path, b, g)
            if np.isfinite(fb) and (not np.isfinite(f_snr) or sn > f_snr):
                flux_band, f_spec, f_snr = b, fb, sn
        if flux_band:
            print(f"  spectrum continuum in {flux_band}: {f_spec:.3f} uJy (S/N {f_snr:.1f})"
                  + ("  too faint for the flux check" if f_snr < 3 else ""))

        grp = []
        for i in ids:
            if i not in isaac.index:
                continue
            o = isaac.loc[i]
            along, perp = offsets(o["RA"], o["Dec"], geo)
            rpred = geo["ytrace"] + geo["kappa"] * along
            eb, er, es = line_excess(o, z, line)
            other = "Ha" if line == "OIII" else "OIII"
            eb2, er2, es2 = line_excess(o, z, other)
            ap = o.get(flux_band, np.nan) if flux_band else np.nan
            tot = ap * o["total_correction"] if np.isfinite(ap) and ap > -90 else np.nan
            rec = {
                "group": gi, "ID": i, "group_IDs": " ".join(map(str, ids)),
                "has_zsys": int(i in zs.index), "z_sys": z, "z_sys_line": line,
                "z_sys_grating": g, "spectrum": path,
                "d_along": round(along, 3), "d_perp": round(perp, 3),
                "row_pred": round(rpred, 2), "line_row": round(cen, 2),
                "drow_line": round(rpred - cen, 2) if np.isfinite(cen) else np.nan,
                "in_shutter": int(abs(perp) <= 0.10 + 0.20 * abs(geo["srcx"])),
                "in_extraction": int(geo["ymin"] - 0.5 <= rpred - geo["ytrace"] <= geo["ymax"] + 0.5),
                "excess_band": eb, "excess_ratio": er, "excess_snr": es,
                f"excess2_line": other, "excess2_band": eb2, "excess2_ratio": er2, "excess2_snr": es2,
                "flux_band": flux_band, "spec_flux": f_spec, "spec_flux_snr": f_snr,
                "spec_over_ap": f_spec / ap if np.isfinite(ap) and ap > 0 else np.nan,
                "spec_over_tot": f_spec / tot if np.isfinite(tot) and tot > 0 else np.nan,
                "f356w_ap": o.get("f356w", np.nan),
                "z_phot": o["z_phot"], "sigma_z": o["sigma_z"],
                "dz_phot": (o["z_phot"] - z) / (1 + z) if o["z_phot"] > -90 else np.nan,
                "Flag": o["Flag"],
            }
            grp.append(rec)
            print(f"    {i:<7d} along {along:+.3f}\" perp {perp:+.3f}\"  row {rpred:5.1f} "
                  f"(line {rec['drow_line']:+.1f})  in_shutter {rec['in_shutter']}  "
                  f"{line} {eb} x{er:.2f} ({es:.1f} sig)  {other} {eb2} x{er2:.2f} ({es2:.1f} sig)  "
                  f"spec/tot {rec['spec_over_tot']:.2f}  z_phot {o['z_phot']:.2f}")
        if not grp:
            continue
        best = min((r for r in grp if np.isfinite(r["drow_line"])),
                   key=lambda r: abs(r["drow_line"]), default=None)
        for r in grp:
            r["closest_to_line"] = int(best is not None and r["ID"] == best["ID"])
        rows += grp

        # ---------------- figure ----------------
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        col = plt.cm.tab10(np.arange(len(grp)))
        # sky positions with the slit
        a0 = ax[0, 0]
        pa = np.radians(geo["pa"])
        ua = np.array([np.sin(pa), np.cos(pa)])
        up = np.array([np.cos(pa), -np.sin(pa)])
        L = 1.0
        for off, ls in ((0.0, "-"), (0.10, "--"), (-0.10, "--")):
            p0 = off * up - L * ua
            p1 = off * up + L * ua
            a0.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.5", lw=1 if off == 0 else 0.8, ls=ls)
        for r, c in zip(grp, col):
            o = isaac.loc[r["ID"]]
            dx = (o["RA"] - geo["ra0"]) * np.cos(np.radians(geo["dec0"])) * 3600
            dy = (o["Dec"] - geo["dec0"]) * 3600
            a0.scatter(dx, dy, color=c, s=60, zorder=3)
            a0.annotate(str(r["ID"]), (dx, dy), xytext=(4, 4), textcoords="offset points", fontsize=8)
        a0.scatter(0, 0, marker="x", color="k", zorder=4, label="MSA target")
        a0.set_aspect("equal")
        lim = max(0.6, 1.3 * max(abs(r["d_along"]) + abs(r["d_perp"]) for r in grp))
        a0.set_xlim(lim, -lim)
        a0.set_ylim(-lim, lim)
        a0.set_xlabel("East offset [arcsec]  (East left)")
        a0.set_ylabel("North offset [arcsec]")
        a0.set_title(f"Slit axis at PA {geo['pa']:.0f} deg through the target, "
                     f"dashed: +-0.1\" (open shutter width)", fontsize=9)
        a0.legend(fontsize=7, loc="lower left")

        # 2D spectrum around the line
        a1 = ax[0, 1]
        sel = np.abs(wave - lam) <= 0.008 * (1 + z)
        if sel.sum() > 3:
            sub = sci[:, sel]
            v = np.nanpercentile(sub, [5, 99])
            a1.imshow(sub, origin="lower", aspect="auto", cmap="Greys", vmin=v[0], vmax=v[1],
                      extent=[wave[sel][0], wave[sel][-1], -0.5, sci.shape[0] - 0.5])
            a1.axvline(lam, color="tab:red", lw=0.8, ls=":")
        for r, c in zip(grp, col):
            a1.axhline(r["row_pred"], color=c, lw=1.2, ls="--")
        a1.axhline(geo["ytrace"], color="k", lw=0.6)
        a1.set_xlabel(r"wavelength [$\mu$m]")
        a1.set_ylabel("row along slit")
        a1.set_title(f"2D spectrum around {line} (dashed: predicted object rows)", fontsize=9)

        # profiles along the slit
        a2 = ax[1, 0]
        if prof is not None:
            rr = np.arange(len(prof))
            a2.step(rr, prof / np.nanmax(np.abs(prof)), where="mid", color="tab:red", label=f"{line} (cont. sub.)")
            a2.step(rr, cont / np.nanmax(np.abs(cont)), where="mid", color="0.4", label="continuum")
            a2.axvline(cen, color="tab:red", lw=0.8, ls=":")
        for r, c in zip(grp, col):
            a2.axvline(r["row_pred"], color=c, lw=1.2, ls="--", label=str(r["ID"]))
        a2.axvspan(geo["ytrace"] + geo["ymin"] - 0.5, geo["ytrace"] + geo["ymax"] + 0.5,
                   color="0.85", zorder=0, label="1D extraction")
        a2.set_xlabel("row along slit")
        a2.set_ylabel("normalised")
        a2.legend(fontsize=7)
        a2.set_title("Where the line and continuum sit along the slit", fontsize=9)

        # photometry
        a3 = ax[1, 1]
        for r, c in zip(grp, col):
            o = isaac.loc[r["ID"]]
            bb = [b for b in BANDS if o.get(b, -99) > -90]
            x = [np.mean(BANDS[b]) for b in bb]
            y = [o[b] for b in bb]
            ye = [o[f"{b}_err"] for b in bb]
            a3.errorbar(x, y, yerr=ye, fmt="o", ms=4, color=c, label=f"{r['ID']}  z_phot {o['z_phot']:.2f}")
        for nm, l0 in STRONG.items():
            a3.axvline(l0 * (1 + z), color="0.8", lw=0.6)
        if np.isfinite(f_spec) and f_snr >= 3:
            a3.scatter(np.mean(BANDS[flux_band]), f_spec, marker="s", s=60, facecolor="none",
                       edgecolor="k", label=f"spectrum in {flux_band}")
        a3.set_xscale("log")
        a3.set_yscale("symlog", linthresh=0.01)
        a3.set_xlabel(r"wavelength [$\mu$m]")
        a3.set_ylabel(r"aperture flux [$\mu$Jy]")
        a3.legend(fontsize=7)
        a3.set_title(f"Isaac photometry (grey: strong lines at z_sys)", fontsize=9)
        fig.suptitle(f"Group {gi}: {' '.join(map(str, ids))}   z_sys {z:.4f} ({line}, {g})", fontsize=11)
        fig.tight_layout()
        png = os.path.abspath(os.path.join(a.out_dir, f"group_{gi:02d}_{'_'.join(map(str, ids))}.png"))
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"  figure  {png}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"\nWrote {os.path.abspath(a.out_csv)}  ({len(df)} rows, "
          f"{df['group'].nunique() if len(df) else 0} groups)")
    print(f"Figures in {os.path.abspath(a.out_dir)}")


if __name__ == "__main__":
    main()