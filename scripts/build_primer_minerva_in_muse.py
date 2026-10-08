#!/usr/bin/env python
"""Build the P2 parent sample from Isaac's PRIMER + MINERVA catalogue.

Every Isaac object that lies in the MUSE footprint and has a DJA medium or
high resolution NIRSpec spectrum (G235M, G235H, G395M, G395H) within
--radius arcsec is kept. The source ID is Isaac's catalogue Number.

Inputs
------
  jwst_catalogs/cosmos_primer_minerva_production.fits   Isaac's catalogue
  jwst_catalogs/dja_msaexp_emission_lines_v4.4.csv.gz   DJA v4.4 Zenodo release, one row per
                                                        spectrum, the main spectrum source
  jwst_catalogs/JELS_F356W_DJA_*_match_0p3as.fits       JELS-DJA match catalogues (Ken, Aug 2026),
                                                        used only for spectra not in v4.4. These
                                                        are the Blue Jay 2 (bluejay2-v4, PID 5427)
                                                        G235H spectra, which are on the DJA server
                                                        but in no DJA table we have
  MEGA_CUBE_VAR_2.fits                                  MUSE footprint (valid pixels)

Outputs
-------
  jwst_catalogs/primer_minerva_in_muse.csv
      One row per Isaac object. For each grating G it gives the best spectrum
      (highest grade, then nearest): G (0/1), z_G, grade_G, sep_G, root_G,
      file_G, nspec_G (number of spectra in that grating within the radius).
      Also dja_z (from the best spectrum over all gratings), z_av, the spread
      of DJA z between gratings in km/s, MUSE footprint and edge flags,
      AO_block, nominal [OIII] and Halpha coverage, the close-pair flags and
      Isaac's photometric columns.
  jwst_catalogs/primer_minerva_close_pairs.csv
      One row per (ID, near_ID) where both Isaac objects lie within --radius
      of the same spectrum. Gives positions, the pair separation in arcsec
      and kpc, each object's distance from the slit position, F356W fluxes,
      photo-z, Flags and the DJA z of the shared spectrum.
  field_images/primer_minerva_muse_footprint.png
      Footprint check with the selected objects and close pairs marked.

Cuts on the DJA table
---------------------
  grating in G235M, G235H, G395M, G395H
  grade >= --grade-min (default 2)
  --zmin < DJA z < --zmax (default 2.9 to 6.7, Lya inside MUSE)
An Isaac object counts as inside the MUSE footprint only if it sits on valid
MUSE data at least --min-edge arcsec (default 1) from the field edge.
The spectrum sources are merged on the spectrum file name. Every v4.4
spectrum is used, and a JELS match row is added only if its file is not
already in v4.4. DJA z is the graded redshift, zgrade in v4.4 and z in the
JELS match catalogues. Both go through the same cuts and the same positional
match to Isaac's catalogue. The column table_<grating> in the main CSV
records whether each chosen spectrum came from v4.4 or jels. Downloading is
the same for both, from <root>/<file> on the DJA server.

Cosmology for kpc: flat LCDM, H0 = 70, Om = 0.3 (as in Paper 1).

Usage
-----
python build_primer_minerva_in_muse.py
python build_primer_minerva_in_muse.py --grade-min 3 --radius 0.3
python build_primer_minerva_in_muse.py --dry-run     # print counts, write nothing
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from scipy.ndimage import distance_transform_edt
from skimage import measure

P2 = "/ceph/cephfs/apatrick/P2"
CAT_DIR = f"{P2}/jwst_catalogs"
ISAAC_PATH = f"{CAT_DIR}/cosmos_primer_minerva_production.fits"
DJA_PATH = f"{CAT_DIR}/dja_msaexp_emission_lines_v4.4.csv.gz"
JELS_MATCH_PATHS = [
    f"{CAT_DIR}/JELS_F356W_DJA_G235H_F170LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G235M_F170LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G395H_F290LP_match_0p3as.fits",
    f"{CAT_DIR}/JELS_F356W_DJA_G395M_F290LP_match_0p3as.fits",
]
CUBE_PATH = "/ceph/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_VAR_2.fits"
OUT_MAIN = f"{CAT_DIR}/primer_minerva_in_muse.csv"
OUT_PAIRS = f"{CAT_DIR}/primer_minerva_close_pairs.csv"
OUT_PNG = f"{P2}/field_images/primer_minerva_muse_footprint.png"

GRATINGS = ["G235H", "G235M", "G395H", "G395M"]
GRATING_FILTER = {"G235H": "F170LP", "G235M": "F170LP",
                  "G395H": "F290LP", "G395M": "F290LP"}
# Nominal wavelength coverage in micron (detector gap of the H gratings ignored).
GRATING_RANGE = {"G235H": (1.66, 3.05), "G235M": (1.66, 3.07),
                 "G395H": (2.87, 5.14), "G395M": (2.87, 5.10)}
OIII_REST_UM = 0.500824
HA_REST_UM = 0.656461
LYA_REST = 1215.67
AO_LO, AO_HI = 5802.0, 5967.0
ARCSEC_PER_PIX = 0.2
EDGE_THRESH_ARCSEC = 6.0
MIN_EDGE_ARCSEC = 1.0  # set from --min-edge in main()
C_KMS = 299792.458

ISAAC_COLS = ["Number", "RA", "Dec", "Flag", "ap_diam", "total_correction",
              "r_half_f356w", "N_bands", "f356w", "f356w_err", "z_phot",
              "sigma_z", "z_best", "zspec", "zspec_cat", "mass_lephare"]
DJA_REQUIRED = ["ra", "dec", "grating", "grade", "root", "file"]

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isaac", default=ISAAC_PATH)
    p.add_argument("--dja", default=DJA_PATH, help="DJA v4.4 Zenodo emission-line table")
    p.add_argument("--jels-match", nargs="*", default=JELS_MATCH_PATHS,
                   help="JELS-DJA match catalogues. Only spectra not in the DJA v4.4 table "
                        "are taken from these (the Blue Jay 2 spectra). Pass with no paths to skip.")
    p.add_argument("--cube", default=CUBE_PATH)
    p.add_argument("--out-main", default=OUT_MAIN)
    p.add_argument("--out-pairs", default=OUT_PAIRS)
    p.add_argument("--out-png", default=OUT_PNG)
    p.add_argument("--radius", type=float, default=0.3, help="match radius, arcsec")
    p.add_argument("--grade-min", type=int, default=2)
    p.add_argument("--min-edge", type=float, default=1.0,
                   help="minimum distance from the MUSE field edge, arcsec, to count as inside")
    p.add_argument("--zmin", type=float, default=2.9)
    p.add_argument("--zmax", type=float, default=6.7)
    p.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    return p.parse_args()


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_isaac(path):
    print(f"Reading Isaac catalogue  {path}")
    t = Table.read(path)
    missing = [c for c in ISAAC_COLS if c not in t.colnames]
    if missing:
        print(f"  ERROR missing columns {missing}")
        print(f"  columns are: {', '.join(t.colnames)}")
        sys.exit(1)
    df = t[ISAAC_COLS].to_pandas()
    df["zspec_cat"] = df["zspec_cat"].apply(
        lambda x: x.decode().strip() if isinstance(x, bytes) else str(x).strip())
    df["Number"] = df["Number"].astype(int)
    df = df.replace(-99, np.nan)
    df["Flag"] = df["Flag"].fillna(-99).astype(int)
    print(f"  {len(df)} objects")
    return df


def normalise_grating(x):
    m = re.search(r"G\d{3}[MH]", str(x).upper())
    return m.group(0) if m else None


def _clean_str(series):
    return series.apply(lambda x: x.decode() if isinstance(x, bytes) else x).astype(str).str.strip()


def _load_one(label, path):
    """Read one spectrum table into ra, dec, grating, grade, z, root, file."""
    if path.endswith((".fits", ".fit")):
        df = Table.read(path).to_pandas()
    else:
        df = pd.read_csv(path, low_memory=False)
    missing = [c for c in DJA_REQUIRED if c not in df.columns]
    if missing:
        print(f"  ERROR {label} is missing columns {missing}")
        print(f"  columns are: {', '.join(df.columns)}")
        sys.exit(1)
    # Graded redshift: zgrade in the Zenodo release, z in the public table.
    zcol = "zgrade" if "zgrade" in df.columns else ("z" if "z" in df.columns else None)
    if zcol is None:
        print(f"  ERROR {label} has no zgrade or z column")
        sys.exit(1)
    out = pd.DataFrame({
        "ra": pd.to_numeric(df["ra"], errors="coerce"),
        "dec": pd.to_numeric(df["dec"], errors="coerce"),
        "grating": _clean_str(df["grating"]),
        "grade": pd.to_numeric(df["grade"], errors="coerce"),
        "dja_z": pd.to_numeric(df[zcol], errors="coerce"),
        "root": _clean_str(df["root"]),
        "file": _clean_str(df["file"]),
    })
    out["dja_table"] = label
    print(f"  {label:<8} {len(out):>6} rows ({out['file'].nunique()} unique files, z from {zcol})  {path}")
    return out


def read_dja(tables, grade_min, zmin, zmax):
    """Union of several DJA spectrum tables, one row per spectrum file.

    tables is a list of (label, path) in priority order. Where a file is in
    more than one table, the values (grade, z, position) come from the first.
    """
    print("Reading DJA spectrum tables (priority order)")
    parts = [_load_one(lab, p) for lab, p in tables]
    allt = pd.concat(parts, ignore_index=True)
    allt = allt[allt["file"].str.len() > 0]
    in_tables = allt.groupby("file")["dja_table"].apply(lambda s: ",".join(dict.fromkeys(s)))
    dja = allt.drop_duplicates("file", keep="first").copy()
    dja["in_tables"] = dja["file"].map(in_tables)
    print(f"  union: {len(dja)} unique spectrum files")

    dja["G"] = dja["grating"].apply(normalise_grating)
    keep = dja["G"].isin(GRATINGS)
    print(f"  G235M/H and G395M/H: {int(keep.sum())}")
    keep &= dja["grade"] >= grade_min
    print(f"  and grade >= {grade_min}: {int(keep.sum())}")
    keep &= (dja["dja_z"] > zmin) & (dja["dja_z"] < zmax)
    print(f"  and {zmin} < z < {zmax}: {int(keep.sum())}")
    keep &= (dja["root"].str.len() > 0) & np.isfinite(dja["ra"]) & np.isfinite(dja["dec"])
    print(f"  and with root and position: {int(keep.sum())}")
    out = dja.loc[keep, ["ra", "dec", "G", "grade", "dja_z", "root", "file",
                         "dja_table", "in_tables"]].reset_index(drop=True)
    for g in GRATINGS:
        print(f"    {g}: {int((out['G'] == g).sum())}")
    print("  passing spectra by the tables they appear in:")
    for k, v in out["in_tables"].value_counts().items():
        print(f"    {k}: {v}")
    return out


def read_footprint(path):
    print(f"Reading MUSE footprint   {path}")
    with fits.open(path, memmap=True) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr).celestial
        mid = hdr["NAXIS3"] // 2
        plane = np.array(hdul[0].data[mid, :, :])
    valid = np.isfinite(plane) & (plane != 0)
    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    dist_pix = distance_transform_edt(padded)[1:-1, 1:-1]
    print(f"  slice {mid}, {int(valid.sum())} valid pixels")
    return wcs, plane, valid, dist_pix


def footprint_flags(ra, dec, wcs, valid, dist_pix):
    """Footprint flags for a set of positions.

    in_muse    1 if on a valid pixel at least MIN_EDGE_ARCSEC from the nearest
               invalid pixel or the field boundary, else 0
    edge       0 if >= EDGE_THRESH_ARCSEC from an edge, else the rounded arcsec
               distance (only set where in_muse = 1)
    edge_dist  distance to the nearest edge in arcsec (NaN off the data)
    x, y       pixel positions
    """
    x, y = wcs.world_to_pixel_values(ra, dec)
    ny, nx = valid.shape
    fin = np.isfinite(x) & np.isfinite(y)
    xi = np.full(len(ra), -1, dtype=int)
    yi = np.full(len(ra), -1, dtype=int)
    xi[fin] = np.round(x[fin]).astype(int)
    yi[fin] = np.round(y[fin]).astype(int)
    on = fin & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    on_valid = np.zeros(len(ra), dtype=bool)
    on_valid[on] = valid[yi[on], xi[on]]
    edge_dist = np.full(len(ra), np.nan)
    edge_dist[on_valid] = dist_pix[yi[on_valid], xi[on_valid]] * ARCSEC_PER_PIX
    in_muse = (on_valid & (np.nan_to_num(edge_dist) >= MIN_EDGE_ARCSEC)).astype(int)
    edge = np.zeros(len(ra), dtype=int)
    v = in_muse == 1
    d = edge_dist[v]
    edge[v] = np.where(d >= EDGE_THRESH_ARCSEC, 0, np.round(d).astype(int))
    return in_muse, edge, edge_dist, x, y


# ----------------------------------------------------------------------
# Building the catalogues
# ----------------------------------------------------------------------
def best_spectrum(rows):
    """Highest grade, then nearest."""
    return rows.sort_values(["grade", "sep"], ascending=[False, True]).iloc[0]


def build_main(pairs, isaac, spec):
    """pairs: one row per (isaac index, spectrum index) within the radius."""
    out = []
    for ii, grp in pairs.groupby("i_isaac"):
        obj = isaac.iloc[ii]
        row = {"ID": int(obj["Number"]), "ra": obj["RA"], "dec": obj["Dec"]}
        zs = []
        for g in GRATINGS:
            sub = grp[grp["G"] == g]
            if len(sub) == 0:
                row.update({g: 0, f"z_{g}": np.nan, f"grade_{g}": np.nan,
                            f"sep_{g}": np.nan, f"root_{g}": "", f"file_{g}": "",
                            f"table_{g}": "", f"nspec_{g}": 0})
                continue
            b = best_spectrum(sub)
            row.update({g: 1, f"z_{g}": b["dja_z"], f"grade_{g}": int(b["grade"]),
                        f"sep_{g}": round(b["sep"], 4), f"root_{g}": b["root"],
                        f"file_{g}": b["file"], f"table_{g}": b["dja_table"],
                        f"nspec_{g}": len(sub)})
            zs.append(b["dja_z"])
        b = best_spectrum(grp)
        row["dja_z"] = b["dja_z"]
        row["dja_z_grating"] = b["G"]
        row["dja_grade"] = int(b["grade"])
        row["z_av"] = float(np.mean(zs))
        row["dv_gratings_kms"] = (C_KMS * (max(zs) - min(zs)) / (1 + np.mean(zs))
                                  if len(zs) > 1 else 0.0)
        row["n_gratings"] = len(zs)
        row["n_spec"] = len(grp)
        row["n_isaac_in_slit"] = int(grp["n_isaac"].max())
        row["close_pair"] = int(row["n_isaac_in_slit"] > 1)
        row["nearest_to_slit"] = int(grp["is_nearest"].any())
        out.append((row, obj))

    rows = []
    for row, obj in out:
        z = row["dja_z"]
        present = [g for g in GRATINGS if row[g] == 1]
        row["oiii_cov"] = int(any(_covered(OIII_REST_UM, z, g) for g in present))
        row["ha_cov"] = int(any(_covered(HA_REST_UM, z, g) for g in present))
        lya = LYA_REST * (1 + z)
        row["AO_block"] = int(AO_LO <= lya <= AO_HI)
        for c in ISAAC_COLS:
            if c in ("Number", "RA", "Dec"):
                continue
            row[f"pm_{c.lower()}"] = obj[c]
        rows.append(row)
    return pd.DataFrame(rows)


def _covered(rest_um, z, g):
    lo, hi = GRATING_RANGE[g]
    return lo <= rest_um * (1 + z) <= hi


def build_pairs(pairs, isaac, spec, main_ids):
    """One row per (ID, near_ID) sharing at least one spectrum."""
    recs = {}
    for js, grp in pairs.groupby("i_spec"):
        if len(grp) < 2:
            continue
        s = spec.iloc[js]
        for _, a in grp.iterrows():
            ida = int(isaac.iloc[a["i_isaac"]]["Number"])
            if ida not in main_ids:
                continue
            for _, b in grp.iterrows():
                if a["i_isaac"] == b["i_isaac"]:
                    continue
                key = (ida, int(isaac.iloc[b["i_isaac"]]["Number"]))
                r = recs.setdefault(key, {"ia": a["i_isaac"], "ib": b["i_isaac"],
                                          "gratings": set(), "n": 0,
                                          "sep_a": np.inf, "sep_b": np.inf,
                                          "best": None})
                r["gratings"].add(s["G"])
                r["n"] += 1
                r["sep_a"] = min(r["sep_a"], a["sep"])
                r["sep_b"] = min(r["sep_b"], b["sep"])
                if r["best"] is None or (s["grade"], -a["sep"]) > (r["best"]["grade"], -r["best_sep"]):
                    r["best"] = s
                    r["best_sep"] = a["sep"]

    rows = []
    for (ida, idb), r in recs.items():
        a = isaac.iloc[r["ia"]]
        b = isaac.iloc[r["ib"]]
        ca = SkyCoord(a["RA"] * u.deg, a["Dec"] * u.deg)
        cb = SkyCoord(b["RA"] * u.deg, b["Dec"] * u.deg)
        sep = ca.separation(cb).arcsec
        z = r["best"]["dja_z"]
        kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
        f_a, f_b = a["f356w"], b["f356w"]
        rows.append({
            "ID": ida, "ra": a["RA"], "dec": a["Dec"],
            "near_ID": idb, "near_ra": b["RA"], "near_dec": b["Dec"],
            "sep_pair_arcsec": round(sep, 4),
            "sep_pair_kpc": round(sep * kpc_per_arcsec, 3),
            "dja_z": z, "dja_grade": int(r["best"]["grade"]),
            "shared_gratings": ",".join(sorted(r["gratings"])),
            "n_shared_spectra": r["n"],
            "sep_slit": round(r["sep_a"], 4), "near_sep_slit": round(r["sep_b"], 4),
            "closer_to_slit": int(r["sep_a"] < r["sep_b"]),
            "f356w": f_a, "near_f356w": f_b,
            "near_f356w_ratio": (f_b / f_a) if np.isfinite(f_a) and f_a > 0 else np.nan,
            "z_phot": a["z_phot"], "sigma_z": a["sigma_z"],
            "near_z_phot": b["z_phot"], "near_sigma_z": b["sigma_z"],
            "zphot_minus_dja": a["z_phot"] - z,
            "near_zphot_minus_dja": b["z_phot"] - z,
            "flag": int(a["Flag"]), "near_flag": int(b["Flag"]),
            "near_in_main": int(idb in main_ids),
        })
    return pd.DataFrame(rows).sort_values(["ID", "near_ID"]).reset_index(drop=True) \
        if rows else pd.DataFrame()


def plot_footprint(plane, valid, x, y, close, x_un, y_un, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    if np.isfinite(plane).any():
        ax.imshow(plane, origin="lower", cmap="Greys_r",
                  norm=simple_norm(plane, "sqrt", percent=99.0))
    for c in measure.find_contours(valid.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color="cyan", lw=0.8)
    ax.scatter(x[~close], y[~close], s=8, c="lime", edgecolors="k", linewidths=0.2,
               label=f"selected ({(~close).sum()})")
    ax.scatter(x[close], y[close], s=14, c="orange", edgecolors="k", linewidths=0.2,
               label=f"close pair ({close.sum()})")
    if len(x_un):
        ax.scatter(x_un, y_un, s=14, marker="x", c="red", linewidths=0.8,
                   label=f"DJA spectrum, no Isaac match ({len(x_un)})")
    ax.set_xlim(0, plane.shape[1])
    ax.set_ylim(0, plane.shape[0])
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title("PRIMER + MINERVA objects with DJA M/H spectra in the MUSE footprint")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    global MIN_EDGE_ARCSEC
    MIN_EDGE_ARCSEC = args.min_edge
    print(f"Inside the MUSE footprint means at least {MIN_EDGE_ARCSEC:g} arcsec from a field edge")
    isaac = read_isaac(args.isaac)
    tables = [("v4.4", args.dja)]
    tables += [("jels", p) for p in args.jels_match]
    spec = read_dja(tables, args.grade_min, args.zmin, args.zmax)
    wcs, plane, valid, dist_pix = read_footprint(args.cube)

    # Spectra in the footprint (for the unmatched check).
    s_in, _, _, s_x, s_y = footprint_flags(spec["ra"].values, spec["dec"].values,
                                        wcs, valid, dist_pix)
    spec["in_muse"] = s_in
    print(f"\nDJA spectra inside the MUSE footprint: {int(s_in.sum())}")

    # Every Isaac object within the radius of every spectrum.
    c_spec = SkyCoord(spec["ra"].values * u.deg, spec["dec"].values * u.deg)
    c_isaac = SkyCoord(isaac["RA"].values * u.deg, isaac["Dec"].values * u.deg)
    i_spec, i_isaac, sep, _ = c_isaac.search_around_sky(c_spec, args.radius * u.arcsec)
    pairs = pd.DataFrame({"i_spec": i_spec, "i_isaac": i_isaac, "sep": sep.arcsec})
    pairs["G"] = spec["G"].values[pairs["i_spec"]]
    pairs["grade"] = spec["grade"].values[pairs["i_spec"]]
    pairs["dja_z"] = spec["dja_z"].values[pairs["i_spec"]]
    pairs["root"] = spec["root"].values[pairs["i_spec"]]
    pairs["file"] = spec["file"].values[pairs["i_spec"]]
    pairs["dja_table"] = spec["dja_table"].values[pairs["i_spec"]]
    pairs["n_isaac"] = pairs.groupby("i_spec")["i_isaac"].transform("count")
    pairs["is_nearest"] = pairs["sep"] == pairs.groupby("i_spec")["sep"].transform("min")

    # Footprint flags for the Isaac objects that touch a spectrum.
    touched = np.unique(pairs["i_isaac"].values)
    o_in, o_edge, o_dist, _, _ = footprint_flags(isaac["RA"].values[touched],
                                         isaac["Dec"].values[touched],
                                         wcs, valid, dist_pix)
    in_map = dict(zip(touched, o_in))
    edge_map = dict(zip(touched, o_edge))
    dist_map = dict(zip(touched, o_dist))
    n_near = int(((o_in == 0) & np.isfinite(o_dist)).sum())
    print(f"Isaac objects near a spectrum on MUSE data but within {MIN_EDGE_ARCSEC:g} arcsec "
          f"of the edge (excluded): {n_near}")
    pairs = pairs[pairs["i_isaac"].map(in_map) == 1].reset_index(drop=True)

    # Spectra in the footprint with no Isaac object within the radius.
    matched_spec = set(np.unique(i_spec))  # any Isaac object, in footprint or not
    unmatched = spec[(spec["in_muse"] == 1) & ~spec.index.isin(matched_spec)]
    print(f"DJA spectra in the footprint with no Isaac object within "
          f"{args.radius:g} arcsec: {len(unmatched)}")
    if len(unmatched):
        near_idx, near_sep, _ = SkyCoord(unmatched["ra"].values * u.deg,
                                         unmatched["dec"].values * u.deg
                                         ).match_to_catalog_sky(c_isaac)
        for (_, r), ni, ns in zip(unmatched.iterrows(), near_idx, near_sep.arcsec):
            print(f"    ra={r['ra']:.6f} dec={r['dec']:.6f} {r['G']} grade={int(r['grade'])} "
                  f"z={r['dja_z']:.3f}  nearest Isaac {int(isaac.iloc[ni]['Number'])} "
                  f"at {ns:.2f}\"  {r['root']}/{r['file']}")

    main_df = build_main(pairs, isaac, spec)
    if main_df.empty:
        print("\nNo Isaac objects in the footprint with a matching spectrum. Nothing written.")
        return
    main_df["in_muse"] = 1
    i_of_id = dict(zip(isaac["Number"].values, range(len(isaac))))
    main_df["edge"] = [int(edge_map[i_of_id[i]]) for i in main_df["ID"]]
    main_df["edge_dist_arcsec"] = [round(float(dist_map[i_of_id[i]]), 2) for i in main_df["ID"]]
    lead = (["ID", "ra", "dec"] + GRATINGS + [f"z_{g}" for g in GRATINGS]
            + ["dja_z", "dja_z_grating", "dja_grade", "z_av", "dv_gratings_kms",
               "n_gratings", "n_spec", "in_muse", "edge", "edge_dist_arcsec", "AO_block",
               "oiii_cov", "ha_cov", "n_isaac_in_slit", "close_pair",
               "nearest_to_slit"])
    rest = [c for c in main_df.columns if c not in lead]
    main_df = main_df[lead + rest].sort_values("ID").reset_index(drop=True)

    pairs_df = build_pairs(pairs, isaac, spec, set(main_df["ID"]))

    # Summary
    print("\n--- Summary ---")
    print(f"Isaac objects in the MUSE footprint with a DJA M/H spectrum: {len(main_df)}")
    for g in GRATINGS:
        print(f"  with {g}: {int(main_df[g].sum())}  "
              f"(more than one {g} spectrum: {int((main_df[f'nspec_{g}'] > 1).sum())})")
    print(f"  in 1/2/3/4 gratings: "
          + "/".join(str(int((main_df['n_gratings'] == k).sum())) for k in (1, 2, 3, 4)))
    print(f"  unique spectra used: {pairs['file'].nunique()}")
    print(f"  within {EDGE_THRESH_ARCSEC:g} arcsec of a MUSE edge: {int((main_df['edge'] > 0).sum())}")
    print(f"  Lya in the AO gap: {int(main_df['AO_block'].sum())}")
    print(f"  [OIII] nominally covered: {int(main_df['oiii_cov'].sum())}, "
          f"Halpha: {int(main_df['ha_cov'].sum())}, "
          f"neither: {int(((main_df['oiii_cov'] == 0) & (main_df['ha_cov'] == 0)).sum())}")
    print(f"  DJA z spread between gratings > 300 km/s: "
          f"{int((main_df['dv_gratings_kms'] > 300).sum())}")
    print(f"  sharing a spectrum with another Isaac object: {int(main_df['close_pair'].sum())}")
    print(f"  of those, nearest to the slit: "
          f"{int((main_df['close_pair'] & main_df['nearest_to_slit']).sum())}")
    print(f"Close-pair rows (ID, near_ID): {len(pairs_df)}")

    if args.dry_run:
        print("\nDry run, nothing written.")
        return

    for p in (args.out_main, args.out_pairs, args.out_png):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    main_df.to_csv(args.out_main, index=False)
    pairs_df.to_csv(args.out_pairs, index=False)

    _, _, _, mx, my = footprint_flags(main_df["ra"].values, main_df["dec"].values,
                                   wcs, valid, dist_pix)
    un_x = s_x[unmatched.index.values] if len(unmatched) else np.array([])
    un_y = s_y[unmatched.index.values] if len(unmatched) else np.array([])
    plot_footprint(plane, valid, mx, my, main_df["close_pair"].values == 1,
                   un_x, un_y, args.out_png)

    print("\nWrote")
    print(f"  {os.path.abspath(args.out_main)}")
    print(f"  {os.path.abspath(args.out_pairs)}")
    print(f"  {os.path.abspath(args.out_png)}")


if __name__ == "__main__":
    main()