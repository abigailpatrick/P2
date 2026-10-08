# P2 pipeline

Spectroscopy of Lyα emitters at 3 < z < 6 with VLT/MUSE and JWST/NIRSpec. Every script lives in this folder and is run from the terminal on the cluster.

```bash
conda activate env39 (or env312 for the lime fits)
cd /ceph/cephfs/apatrick/P2/scripts
```

All data paths are under `/ceph/cephfs/apatrick/P2`. Each script prints the full paths of what it reads and writes. The end product is `master_catalog/p2_master_catalog.csv`, rebuilt from the step outputs by `build_master_catalog.py` (step 28). Rerun any step, then step 28, and the master is up to date.

## Systemic redshift rule

One rule is used everywhere, defined once in `p2_common.py` (`pick_zsys`).

1. [OIII] if its S/N > 13
2. otherwise Hα if its S/N > 13
3. otherwise no z_sys

The S/N is LiMe's line S/N from `systemic_redshifts_by_JELS_ID.csv`, which only holds successful fits and already takes the best grating. Hβ, [NII] and [OII] are still fitted but are not used for z_sys. The MUSE steps fall back to the DJA redshift (`z_dja`) for sources with no z_sys. `build_zsys_catalog.py`, `find_delta_v_from_best_zsys_line.py`, `fit_muv_beta.py`, `stack_lya_systematic.py` and `build_master_catalog.py` all import it. Each takes `--oiii-snr-min` and `--ha-snr-min` to change the thresholds.

## Steps in run order

Each step lists the command for all sources and, where the script allows it, for one source. Replace `<ID>` with the JELS ID.

### A. NIRSpec sample and spectra

**1. Download the DJA spectra** `extract_nirspec_spectra.py`
Writes `jwst_spectra/<grating>/<ID>_<grating>_spectra.fits` and a PNG of each.
```bash
python extract_nirspec_spectra.py
```
No single-source mode.

**2. Convert to LiMe format** `make_lime_spectra_format.py`
Writes `<ID>_<grating>_spectra_lime.fits` next to each spectrum.
```bash
for g in G235H_F170LP G235M_F170LP G395H_F290LP G395M_F290LP; do
  python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/$g
done
# one spectrum
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra/G395H_F290LP/<ID>_G395H_F290LP_spectra.fits
```

**3. Merge the four grating catalogues** `merge_grating_catalogs.py`
Writes `jwst_catalogs/grating_sources_by_JELS_ID.csv` (with in_muse, edge, duplicate, AO_block), `grating_sources_by_JELS_ID_good.csv` and `field_images/muse_footprint_check.png`.
```bash
python merge_grating_catalogs.py
```
No single-source mode. Only rerun when the source list changes.

### B. Systemic redshifts

**4. Fit [OIII]** `lime_OIII_jointfit.py`
Writes `jwst_catalogs/OIII_results_by_JELS_ID.csv`, `OIII_summary_by_JELS_ID.csv` and fit figures under `jwst_spectra/OIII_fits/<ID>/`.
```bash
python lime_OIII_jointfit.py
# one source, by hand, patching only its rows in the two OIII CSVs
python lime_OIII_singlereview.py <ID>
python lime_OIII_singlereview.py <ID> --z 5.9412 --gratings G235M --dry-run
```

**5. Fit Hβ, Hα, [NII], [OII]** `lime_lines_jointfit.py`
Writes `<line>_results_by_JELS_ID.csv` and `<line>_summary_by_JELS_ID.csv` for each line, including the Hα and Hβ fluxes and the Hα FWHM.
```bash
python lime_lines_jointfit.py
```
No single-source mode.

**6. Merge the line redshifts** `merge_systematic_redshifts.py`
Writes `jwst_catalogs/systemic_redshifts_by_JELS_ID.csv`, one row per source, best successful grating per line.
```bash
python merge_systematic_redshifts.py
```
Fast. Always runs on every source. Rerun after step 4, 5 or any `lime_OIII_singlereview.py` fix.

**7. z_sys for the MUSE steps** `build_zsys_catalog.py`
Writes `jwst_catalogs/grating_sources_with_zsys.csv`, which steps 9 to 14 read. It prints which sources gained, lost or changed z_sys compared with the file it replaces.
```bash
python build_zsys_catalog.py --dry-run   # see what would change
python build_zsys_catalog.py
```
Fast. Always runs on every source.

### C. MUSE Lyα

**8. Cut subcubes from the megacube** `extract_subcubes.py`
Writes `MUSE_subcubes/subcube_JELSID_<ID>.fits`. Slow, as it reads the 200 GB cube. Does not depend on z, so only rerun for new sources.
```bash
python extract_subcubes.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID.csv \
  --cube /ceph/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_WITH_VAR_2.fits \
  --out-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes --in_muse_check
```
No single-source flag. For one source, pass a CSV holding only that row.

**9. Continuum subtraction** `lya_local_contsub.py`
Writes `MUSE_subcubes/contsub/source_<ID>_lya_contsub_cube_velocity.fits`, `source_<ID>_continuum_cube_velocity.fits` and QA plots.
```bash
python lya_local_contsub.py --mask_mode velocity --dv_blue 300 --dv_red 1200 \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --cube_dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/ \
  --outdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub
# one source: add  --source <ID>
```

**10. Optimise the aperture position** `optimize_lya_position_grating.py`
Writes `MUSE_catalogs/optimal_offsets_grating.csv`.
```bash
python optimize_lya_position_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --prior-scale 1.0 --dx-max 0.4 --dx-step 0.1 --dv-blue 300 --dv-red 1200 \
  --outfile /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv
```
No single-source mode. It always reruns every source, which is quick.

**11. Extract aperture spectra** `ap_extract_specs_grating.py`
Writes `MUSE_subcubes/dataproducts/<ID>_spectrum.npz`.
```bash
python ap_extract_specs_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --offset-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/optimal_offsets_grating.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --aperture 0.6 --dv-blue 300 --dv-red 1200
# one source: add  --id <ID>
```

**12. Sliding-window S/N** `sliding_snr_lya_grating.py`
Writes `MUSE_catalogs/lya_sliding_snr_grating.csv` and pseudo-NB figures in `dataproducts/snr_figures/`.
```bash
python sliding_snr_lya_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --aperture 0.6
# one source: add  --id <ID>   (replaces only that row in the CSV)
```

**13. Fit Lyα** `fit_lya_properties_grating.py`
Writes `MUSE_catalogs/lya_properties.csv` and `dataproducts/snr_figures/<ID>_lya_fit.png`.
```bash
python fit_lya_properties_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --snr-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --outfile /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv \
  --figdir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/snr_figures
# one source: add  --id <ID>   (replaces only that row in the CSV)
```

**14. Monte Carlo errors** `mc_lya_errors_grating.py` + `merge_mc_partials.py`
Writes `MUSE_catalogs/mc_partials/mc_row<N>.csv`, merged into `MUSE_catalogs/lya_properties_mc.csv`. Use this file for science, not `lya_properties.csv`.
```bash
cd slurm
AID=$(sbatch --parsable run_mc_lya_errors.slurm)
sbatch --dependency=afterok:$AID merge_mc_lya_errors.slurm
cd ..
```
Set `--array=0-(N-1)` in `run_mc_lya_errors.slurm`, where N is the number of successful fits in `lya_properties.csv`. For one source, find its row number among the successful fits, rerun that row, then merge.
```bash
python -c "import pandas as pd; d=pd.read_csv('/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv'); d=d[d.fit_success.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True); print(d.index[d.ID==<ID>][0])"
python mc_lya_errors_grating.py \
  --indir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts \
  --snr-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv \
  --catalog /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv \
  --properties-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv \
  --out-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/mc_partials/mc_row<N>.csv \
  --n-mc 500 --mode model --row-index <N>
python merge_mc_partials.py --partials-dir /ceph/cephfs/apatrick/P2/MUSE_catalogs/mc_partials
```
The row numbers shift if the set of successful fits changes. In that case rerun the whole array.

### D. Detection thresholds

Only needed when the continuum-subtracted cubes of the good sample change.

**15. Place false positions** `false_pos_grating.py`
Writes `dataproducts/false_positions/source_<ID>_false_positions.csv`, a PNG and the manifest `false_positions_kept.csv`.
```bash
python false_pos_grating.py \
  --csv /ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_by_JELS_ID_good.csv \
  --cube-dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/ --n 500 --box 8.0
# one source: add  --id <ID>
```

**16. Optimise the false positions** `optimize_lya_positions_f_grating.py`
Writes `source_<ID>_false_optimal_snr.csv`. Slow, so run all sources on SLURM.
```bash
sbatch slurm/run_optimize_lya_positions_f_grating.slurm
# one source
python optimize_lya_positions_f_grating.py \
  --false-csv /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions/source_<ID>_false_positions.csv \
  --cube /ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/source_<ID>_lya_contsub_cube_velocity.fits \
  --aperture 0.6 --dx-max 0.4 --dx-step 0.1 --prior-scale 1.0
```

**17. Detection thresholds** `false_pos_percentiles.py`
Writes `MUSE_catalogs/false_pos_snr_percentiles.csv` (98 and 99.5 per cent S/N) and `plots/false_pos_summary.png`.
```bash
python false_pos_percentiles.py \
  --dir /ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions \
  --out-csv /ceph/cephfs/apatrick/P2/MUSE_catalogs/false_pos_snr_percentiles.csv
```

### E. Velocity offsets and grouping

**18. Δv** `find_delta_v_from_best_zsys_line.py`
Writes `MUSE_catalogs/delta_v_from_best_zsys_line.csv`.
```bash
python find_delta_v_from_best_zsys_line.py
```
Fast. Always runs on every source.

**19. Automatic tiers** `group_lya_sample.py`
Writes `MUSE_catalogs/lya_group_<gold|silver|bronze|stone|bad>.csv` and contact-sheet PDFs.
```bash
python group_lya_sample.py
```

**20. Update the label file** `update_manual_labels.py`
Writes `MUSE_catalogs/lya_manual_labels.csv`. On the first run it copies the labels out of the tier CSVs. After that it only adds new IDs with a blank label. Fill in any blank labels by hand.
```bash
python update_manual_labels.py --dry-run
python update_manual_labels.py
```

**21. Regroup by label** `group_by_manual.py`
Writes `MUSE_catalogs/lya_group_<a|b|c|d>.csv` and PDFs.
```bash
python group_by_manual.py
```

### F. Galaxy properties

**21b. Match to PRIMER + MINERVA** `match_primer_minerva.py`
Matches every JELS source to Isaac's COSMOS catalogue (`jwst_catalogs/cosmos_primer_minerva_production.fits`) by position, within 0.3 arcsec. Writes `jwst_catalogs/primer_minerva_by_JELS_ID.csv`, one row per JELS ID. Each row has the match, the catalogue flag and total correction, the photo-z and LePhare mass, and for each of the 19 bands the aperture flux (`<band>_ap`) and the total flux (`<band>_tot`, aperture × total_correction), all in µJy. Steps 22 to 24 and 28 read it. Only rerun when the catalogue or the source list changes.
```bash
python match_primer_minerva.py
```

**22. M_UV and β** `fit_muv_beta.py`
Uses the PRIMER + MINERVA total fluxes by default (`--aper aperture` for the aperture fluxes), including the MINERVA medium bands.
Writes `jwst_catalogs/muv_beta_by_JELS_ID.csv` and `jwst_spectra/MUV_fits/<ID>/<ID>_muv_beta_fit.png`.
```bash
python fit_muv_beta.py --all
# one source, look only (a non-dry --id run writes a one-row CSV)
python fit_muv_beta.py --id <ID> --dry-run
```

**23. M_UV and β checks** `muv_beta_diagnostics.py`
Writes `jwst_catalogs/muv_beta_diagnostics.csv` and `plots/muv_beta_diagnostics.png`.
```bash
python muv_beta_diagnostics.py
```

**24. Hα and Hβ corrections** `build_ha_flux_corrections.py`
Slit-loss correction of the NIRSpec line fluxes to the PRIMER + MINERVA total photometry, then the Balmer-decrement dust correction. Writes `jwst_catalogs/ha_hb_flux_corrections.csv` and `_values.csv`. The method is described at the top of the script.
```bash
python build_ha_flux_corrections.py --download-filters   # once, gets the medium-band filter curves
python build_ha_flux_corrections.py
python build_ha_flux_corrections.py --phot aperture \
  --out /ceph/cephfs/apatrick/P2/jwst_catalogs/ha_hb_flux_corrections_aper.csv
python build_ha_flux_corrections.py --slit-mode dja    # DJA path loss only, no photometric correction
```

### G. Figures

**25. Lyα against the systemic line, per source** `plot_lya_vs_systemic.py`
Writes `plots/<ID>/`.
```bash
python plot_lya_vs_systemic.py --group a
python plot_lya_vs_systemic.py --id <ID>
```

**26. Science plots** `plot_lya_science.py`
Writes to `plots/`.
```bash
python plot_lya_science.py --label all
```

**27. Stacks** `stack_lya_systematic.py`
Systemic-anchored stacks in bins of any master column, normalised by f1500, Hα or nothing. Writes composites, measurements and figures to `plots/stacks/`.
```bash
python stack_lya_systematic.py --norms f1500 halpha none --nboot 100
python stack_lya_systematic.py --splits z_sys:3.5 M_UV:-19 beta:-2
```

### H. Master catalogue

**28. Build the master** `build_master_catalog.py`
Writes `master_catalog/p2_master_catalog.csv` and `master_catalog/p2_master_columns.csv`.
```bash
python build_master_catalog.py --dry-run   # show what would change
python build_master_catalog.py
```

## What depends on what

When an input changes, these are the steps downstream of it. Steps marked * can be run for single sources.

| If this changes | Rerun |
|---|---|
| a systemic line fit (4*, 5) | 6, 7, then for the sources whose z_sys moved 9*, 10, 11*, 12*, 13*, 14*, then 18 to 22, 24, 28 |
| the source list | 1 to 3, then everything |
| continuum subtraction settings | 9 to 14, 15 to 17 if the good sample is affected, then 18 to 21, 28 |
| a Lyα fit setting | 13, 14, 18 to 21, 28 |
| a by-eye label | edit `lya_manual_labels.csv`, then 21, 28 |
| photometry catalogue | 21b, 22 to 24, 27, 28 |
| Hα method | 24, 27, 28 |
| SED or fesc outputs | 28 |

Step 7 lists the sources whose z_sys changed. The master also carries `zsys_mismatch_kms`, the difference between the z the MUSE products were made with and the current one.

## Manual labels

`MUSE_catalogs/lya_manual_labels.csv` (ID, manual, tier_when_labelled, note) is the only home for the a/b/c/d labels. No script overwrites an existing row. `group_lya_sample.py`, `group_by_manual.py` and `build_master_catalog.py` all read it, through a `--labels-csv` option. If the file does not exist they fall back to the `manual` column of `lya_group_<tier>.csv`, as before.

## Master catalogue

The master has one row per source in `grating_sources_by_JELS_ID.csv` (94 at present) and 144 columns. `p2_master_columns.csv` gives each column's block, source file, source column, unit and description. The blocks run in this order.

sample, zsys, Lyα detection, Lyα fit, Δv, tier, manual labels, UV, Hα/Hβ fluxes and FWHM, Hα corrections, SED, fesc, provenance

- The SED and fesc blocks are empty placeholders until their scripts exist.
- A missing input leaves its block blank with a warning, so the build never fails because a step has not run.
- To add a column, add a line to `BLOCKS` in `build_master_catalog.py`.
- `is_primary` is False for the dropped member of each duplicate pair (42990, 43604, 49296, 49694 at present).
- `lya_det_98` and `lya_det_99p5` compare `lya_peak_snr` with the false-positive thresholds from step 17.
- Provenance columns at the end record which redshift each product was made with. `zsys_muse` is the z the MUSE steps used and `zsys_mismatch_kms` is its offset from the current one. `z_sys_dv` is the z_sys used for Δv.
- Each build prints, per column, which IDs changed since the previous build.

`lya_flux_cgs` assumes the MUSE cube unit is 1e-20 erg/s/cm². Change `--muse-flux-unit` if `BUNIT` says otherwise.

## What changed in this reorganisation

**New files**

- `p2_common.py` holds the z_sys rule, the shared constants and the fallback grating used for z_dja.
- `build_master_catalog.py` builds the master catalogue.
- `update_manual_labels.py` creates and extends the label file.
- `README.md` is this file.

**Rewritten.** `build_zsys_catalog.py` now uses the shared rule. The old version took the best-S/N [OIII] grating with no threshold and no success check, so failed fits fed the MUSE steps. The output name and first columns are unchanged, plus a new `z_sys_line`. `z_sys_quality` is now a ([OIII]), b (Hα) or d (none). Where there is no z_sys, `z_dja` comes from the grating with the highest attempted [OIII] S/N. DJA redshifts from different gratings can differ by about 1000 km/s (17416 has 5.878 in G235H and 5.851 in G395H).

**Edited**

- `find_delta_v_from_best_zsys_line.py`, `fit_muv_beta.py` and `stack_lya_systematic.py` now import the rule from `p2_common.py` instead of carrying their own copies. Those copies allowed Hβ, [NII] and [OII] and had no Hα threshold. With the current fits the result is identical, because every source without a good [OIII] has Hα S/N above 14.
- `group_lya_sample.py` and `group_by_manual.py` gained `--labels-csv` and read the labels from the label file.
- `sliding_snr_lya_grating.py --id` and `fit_lya_properties_grating.py --id` now replace just that source's row. Before, they cut the CSV down to one row.
- `lime_OIII_singlereview.py` had its usage text corrected to its own file name.
- `.gitignore` now ignores `*.log` and `*.err`.

**Moved.** `proposal_sample_checks.py` went to `proposal/` and `falsepos_opt_10474818_runtime.log` to `slurm/`.

**Archived** (in `archive/`, kept for reference, not run)

| Script | Why |
|---|---|
| `lime_OIII_singlefit.py` | superseded by `lime_OIII_jointfit.py`. It also wrote old z_OIII columns into the base catalogue |
| `lime_OIII_single_review.py` | patched `grating_sources_with_zsys.csv`, which step 7 overwrites, so its fixes never reached Δv. `lime_OIII_singlereview.py` patches the CSVs that feed step 6. Its `--force-double`, `--cont-side` and `--mask` options still need porting across |
| `group_by_dv_err.py` | superseded by `group_lya_sample.py` |
| `find_OIII_filter.py` | its `OIII_filter` column is not read anywhere |
| `add_detection_flags.py` | replaced by `lya_peak_snr`, `lya_det_98` and `lya_det_99p5` in the master |
| `checks/spec_nb_test.py` | duplicate of `checks/spec_nb_single.py` |
| `checks/false_pos_percentiles.py` | loose version of `false_pos_percentiles.py` |

## CSVs that are no longer needed

- `jwst_catalogs/OIII_results_by_JELS_ID_2.csv` and `OIII_summary_by_JELS_ID_2.csv` differ from the main files only for 49939 G235H, by 15 km/s. Nothing reads them.
- `MUSE_catalogs/lya_properties_a30.csv`, `lya_properties_grouped.csv`, `lya_sample_tiered.csv` and `mc_partials_preLSF/` are earlier versions.
- `jwst_catalogs/ha_hb_flux_corrections_2as*.csv` can go once the Hα rewrite is in.
- `jwst_catalogs/audit_missing_OIII.csv` and `duplicate_sources.csv` are one-off check outputs.
- The extra columns other scripts wrote into `grating_sources_by_JELS_ID.csv` (peak_snr, over_98, over_99.5, OIII_filter, z_OIII_*) vanish the next time step 3 runs. Nothing reads them now.

`lya_properties.csv` is still needed as the input to step 14.

## Still to do

**Small fixes**

- Confirm that `MEGA_CUBE_VAR_2.fits` (read in step 3 for in_muse and edge) and `MEGA_CUBE_WITH_VAR_2.fits` (step 8) cover the same footprint.
- Port `--force-double`, `--cont-side` and `--mask` into `lime_OIII_singlereview.py`.
- AO gap flag and the edge false-position rejection.

**Hα corrections** (`build_ha_flux_corrections.py`)

- Add the SED dust fallback (A_V from Ken's fits scaled to Hα) for sources without a usable Hβ.
- Check the size of the corrections with the new total photometry. With the old 0.6 arcsec photometry the median total was ×4.4 and the largest ×46.
- Decide how to match the Lyα aperture. The Hα is now corrected to total, but the Lyα flux is a 0.6 arcsec MUSE aperture flux, so fesc needs either a total Lyα flux or an aperture Hα.

**Stacking** (`stack_lya_systematic.py`)

- Fix the per-pixel clipping (clip whole sources instead) and measure B/R against a local baseline, once the choices on the decision list are agreed.

**SED inputs** (new script)

- Read Ken's BAGPIPES fits (run on the PRIMER + MINERVA catalogue) and match on JELS ID, through `primer_minerva_by_JELS_ID.csv` if they are keyed on the catalogue Number.
- Correct M* and SFR to total with `pm_total_correction` if the fits used aperture fluxes.
- Write `sed_logM`, `sed_sfr10`, `sed_ssfr10`, `sed_av`, `sed_ebv` and their errors, the names the master already expects.
- Those fits used photometric redshift priors. Check whether any sources need refitting at z_sys.

**Lyα escape fractions** (new script)

- fesc = F_Lyα / (8.7 F_Hα,corr) per source (Paper 1, Eq. 7).
- Detections at the 98 or 99.5 per cent threshold, 5σ upper limits for non-detections.
- Sample median with reverse Kaplan-Meier (lifelines) and bootstrap errors, plus the stacked-flux fesc, as in Paper 1 Section 5.2.
- Write `fesc_lya`, its errors, `fesc_lya_is_limit` and `fesc_ha_dust` for the master.