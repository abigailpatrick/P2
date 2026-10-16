# P2 pipeline

Spectroscopy of Lyα emitters at 3 < z < 6 with VLT/MUSE and JWST/NIRSpec. Every script lives in this folder and is run from the terminal on the cluster.

```bash
conda activate env39 (or env312 for the lime fits)
cd /ceph/cephfs/apatrick/P2/scripts
```

All data paths are under `/ceph/cephfs/apatrick/P2`. Each script prints the full paths of what it reads and writes. The end product is `master_catalog/p2_master_catalog.csv`, rebuilt from the step outputs by `build_master_catalog.py` (step 28). Rerun any step, then step 28, and the master is up to date.

## Rebuild in progress (October 2026)

The pipeline is being rewritten step by step to start from Isaac's PRIMER + MINERVA catalogue. Every source is now keyed on Isaac's catalogue `Number` (the `ID` column), not the JELS ID. Sections A to D below are rewritten (sample, z_sys, MUSE Lyα, detection thresholds, the Lyα catalogue and the Lyα against systemic figures), as are steps 22, 23 (M_UV and β) and 26 (science plots). Step 14b (double-peak fit) is new. Steps 18 to 21, 21b, 24, 27 and 28 still describe the old JELS-ID version and will be updated as each is redone. Step 25 has moved to 17d.

## Systemic redshift rule

One rule is used everywhere, defined once in `p2_common.py` (`pick_zsys`).

1. [OIII] if it is detected, A/noise ≥ 5
2. otherwise Hα if detected, A/noise ≥ 5
3. otherwise no z_sys, and the source leaves the z_sys sample

A/noise is the fitted line amplitude divided by the flux scatter in the two adjacent continuum bands, the quantity the LiMe paper (Fernández et al. 2024, Sect. 5) uses, and 5 is its detection boundary for lines about as wide as the resolution. A fit counts only if it also has a FWHM of at least 1 pixel (not a single-pixel spike) and a centre error ≤ 2.5 Å. Every fit uses LiMe's adjacent-band continuum, lines in the detector gap are not fitted ([OIII] needs 5007 itself on the spectrum), and Hα's z comes from the Hα + [NII] blend. For each line the grating with the highest A/noise among the passing fits is used. Hβ and [OII] are fitted and kept but not used for z_sys. `build_zsys_catalog.py` takes `--oiii-snr-min` and `--ha-snr-min` to change the thresholds.

## Steps in run order

### A. NIRSpec sample and spectra

**1. Build the sample** `build_primer_minerva_in_muse.py`
Every Isaac object at least 1″ inside the MUSE footprint with a DJA G235M/H or G395M/H spectrum (grade ≥ 2, 2.9 < z < 6.7) within 0.3″. Spectra come from the DJA v4.4 table plus, only where v4.4 lacks them, the JELS-DJA match catalogues (the Blue Jay 2 G235H spectra). Writes `jwst_catalogs/primer_minerva_in_muse.csv`, `jwst_catalogs/primer_minerva_close_pairs.csv` and `field_images/primer_minerva_muse_footprint.png`.
```bash
python build_primer_minerva_in_muse.py --dry-run
python build_primer_minerva_in_muse.py
```

**2. Download the DJA spectra** `extract_nirspec_spectra.py`
Writes `jwst_spectra_pm/<grating>_<filter>/<ID>_<grating>_<filter>_spectra.fits`, a PNG of each and `jwst_spectra_pm/download_log.csv`. Existing files are skipped unless `--overwrite`.
```bash
python extract_nirspec_spectra.py --dry-run
python extract_nirspec_spectra.py
python extract_nirspec_spectra.py --ids <ID>
```

**3. Convert to LiMe format** `make_lime_spectra_format.py`
Writes `<ID>_<grating>_<filter>_spectra_lime.fits` next to each spectrum. With no argument it converts every grating folder in `jwst_spectra_pm/`.
```bash
python make_lime_spectra_format.py
python make_lime_spectra_format.py /ceph/cephfs/apatrick/P2/jwst_spectra_pm/G395H_F290LP/<ID>_G395H_F290LP_spectra.fits
```

### B. Systemic redshifts

**4. Fit [OIII], Hβ, Hα (+[NII]) and [OII]** `lime_fit_lines.py` (env312)
Writes figures to `lime_fits/<ID>/<line>_fits/` (contsub and fit PNGs; the fit PNG is the fit that gave z, the [NII] blend for Hα), `jwst_catalogs/lime_<line>_fits.csv` (one row per source and grating) and `jwst_catalogs/lime_<line>_summary.csv` (one row per source, best fit, best Hβ/Hα flux, best Hα FWHM).
```bash
conda activate env312
python lime_fit_lines.py
python lime_fit_lines.py --lines OIII Ha
# review one source, replacing only its rows in the CSVs
python lime_fit_lines.py --ids <ID>
python lime_fit_lines.py --ids <ID> --lines OIII --gratings G235M --z 5.9412 --line-margin 25
python lime_fit_lines.py --ids <ID> --lines OIII --force-single --dry-run
```

**5. z_sys** `build_zsys_catalog.py` (env39)
Writes `jwst_catalogs/systemic_redshifts.csv` (every source, best z per line, z_sys where there is one) and `jwst_catalogs/primer_minerva_in_muse_zsys.csv` (only the sources with a z_sys, with `z_sys`, `z_sys_err`, `z_sys_snr`, `z_sys_line`, `z_sys_grating` and `dv_sys_dja_kms`). Prints why each dropped source has no z_sys.
```bash
python build_zsys_catalog.py --dry-run
python build_zsys_catalog.py
```
Rerun after any `lime_fit_lines.py --ids` review.

### C. MUSE Lyα

**8. Cut subcubes from the megacube** `extract_subcubes.py` (rewritten for Isaac IDs)
Cuts a 101 × 101 px (20.2″) subcube, flux and variance, around every source in `jwst_catalogs/primer_minerva_in_muse_zsys.csv`, centred on Isaac's RA/Dec. Each source gets a folder `MUSE_subcubes/<ID>/`, which the later Lyα steps also write into. Writes `MUSE_subcubes/<ID>/<ID>_subcube.fits` (extensions DATA and STAT, with the exact source position in the primary header as `SRC_RA`, `SRC_DEC`, `SRC_X`, `SRC_Y`), a white-light PNG `<ID>_subcube_white.png` and `MUSE_subcubes/subcube_log.csv`. Slow, as it reads the 200 GB cube. Does not depend on z, so only rerun for new sources.
```bash
python extract_subcubes.py --dry-run
python extract_subcubes.py
python extract_subcubes.py --ids <ID>
python extract_subcubes.py --skip-existing    # only sources without a subcube
```

**9. Continuum subtraction** `lya_local_contsub.py` (rewritten for Isaac IDs)
Running-median continuum (101 channels) per spaxel, masking -300/+1200 km/s about Lyα at z_sys, interpolated across the mask and subtracted. Lyα is placed at its air wavelength, since the cube is in air and z_sys is a vacuum redshift. Writes into each source's folder `MUSE_subcubes/<ID>/<ID>_lya_contsub_cube.fits` (use this for Lyα), `<ID>_continuum_model_cube.fits` (the model, not for Lyα) and `<ID>_contsub_QA.png`. Sources without a subcube are skipped.
```bash
python lya_local_contsub.py
python lya_local_contsub.py --ids <ID>
python lya_local_contsub.py --skip-existing
```

**10. Optimise the aperture position** `optimize_lya_position_grating.py` (rewritten for Isaac IDs)
Moves the 0.6″ aperture over ±0.4″ in 0.1″ steps about Isaac's position and keeps the offset with the highest band S/N times a Gaussian prior (σ = 1″), as in Paper 1. New in P2, a trial position is only allowed if it is at least as close to the target as to any other object in Isaac's catalogue, so the aperture cannot move onto a neighbour (`--no-object-rule` turns this off). Writes `MUSE_catalogs/lya_optimal_offsets.csv` (with the S/N at the original and moved positions, `at_cap` for offsets on the edge of the grid, `n_blocked` and the nearest neighbour) and `MUSE_subcubes/<ID>/<ID>_lya_position.png` (Lyα narrowband image with both apertures and nearby Isaac objects, the S/N over the grid with blocked offsets in grey, and both spectra). A rerun lists the sources whose aperture changed, which need new MC errors. `--ids` patches only those rows.
```bash
python optimize_lya_position_grating.py
python optimize_lya_position_grating.py --ids <ID>
```

**11. Extract aperture spectra** `ap_extract_specs_grating.py` (rewritten for Isaac IDs)
0.6″ radius aperture at the optimised position, over the full wavelength range. Writes `MUSE_subcubes/<ID>/<ID>_lya_spectrum.npz`, which also holds the spectrum at the original position (`flux_orig`, `var_orig`). `--no-offsets` extracts at Isaac's position without the offsets CSV.
```bash
python ap_extract_specs_grating.py
python ap_extract_specs_grating.py --ids <ID>
```

**12. Lyα detection S/N** `sliding_snr_lya_grating.py` (rewritten for Isaac IDs)
Σ_Lyα as in Paper 1. A 10 Å window slides in 1 Å steps along each aperture spectrum, the flux in the window over its propagated error is smoothed with σ = 2 steps, and the peak inside the Lyα band gives `best_center` and `peak_snr_slide`. The S/N function lives in `p2_common.py` (`sliding_snr`, `best_centre`) and is shared with the position optimiser and the false positives. Writes `MUSE_catalogs/lya_sliding_snr.csv` and `MUSE_subcubes/<ID>/<ID>_lya_sliding_snr.png`.
```bash
python sliding_snr_lya_grating.py
python sliding_snr_lya_grating.py --ids <ID>
```

**13. Fit Lyα** `fit_lya_properties_grating.py` (rewritten for Isaac IDs)
Fits a skew-normal convolved with the MUSE LSF within ±25 Å of `best_center`, giving z_lya, Δv, flux and FWHM, as Paper 1 but with the LSF in the model. `lya_snr` is the Σ_Lyα from step 12, the S/N for detection. As in Paper 1 the fitted flux is checked against a direct integration over the line window (`flux_line`, the central 90% of the fitted line), and the pseudo-NB is collapsed over the 10 Å detection window (`--nb-window line` uses the line window). Every source is fitted. Writes `MUSE_catalogs/lya_fits.csv`, `<ID>_lya_nb.fits` and `<ID>_lya_fit.png` (spectrum, fit and detection window on the left, 3″ pseudo-NB on the right).
```bash
python fit_lya_properties_grating.py
python fit_lya_properties_grating.py --ids <ID>
```

**14. Monte Carlo errors** `mc_lya_errors_grating.py` + `merge_mc_partials.py` (rewritten for Isaac IDs)
500 parametric-bootstrap refits per successful fit (method unchanged). Each source writes `MUSE_subcubes/<ID>/<ID>_lya_mc.csv` and the merge joins them onto the fits as `MUSE_catalogs/lya_fits_mc.csv`, which is the file for science. The merge leaves out, and lists, any MC result whose refit peak is more than 5 km/s from the current fit's peak (`--peak-tol-kms`), i.e. the fit has changed since the MC. Smaller differences (well under 1 km/s) come from refitting the same spectrum and are kept and listed. Array task N does the Nth ID of the sorted sample, so the numbering only changes with the sample.
```bash
python mc_lya_errors_grating.py --ids <ID> --n-mc 100     # quick test, merges at the end
cd slurm
N=$(python -c "import pandas as pd; print(len(pd.read_csv('/ceph/cephfs/apatrick/P2/jwst_catalogs/primer_minerva_in_muse_zsys.csv')))")
AID=$(sbatch --parsable --array=0-$((N-1)) run_mc_lya_errors.slurm)
sbatch --dependency=afterok:$AID merge_mc_lya_errors.slurm
```

**14b. Double-peak fit, optional** `fit_lya_double_peak.py` (new)
A second fit to every source with a single fit, of two Gaussians each convolved with the MUSE LSF, to find lines with a blue and a red peak. The blue centre is free between −1000 and +100 km/s of systemic Lyα and the red between −100 and +1000 km/s (the overlap allows for the z_sys error). It uses the same pixels as the single fit (`best_center` ± 25 Å) and the best of nine starting points. The two models are compared with the BIC, χ² + k ln n (k = 4 for the skewed Gaussian, 6 for two Gaussians, n the pixels fitted), and `dbic` = BIC_single − BIC_double, so positive favours two peaks. A source is flagged `double_peak` = 1 if dbic > 6 (strong evidence, Kass & Raftery 1995, `--dbic-min`), both peak fluxes are at least 2σ (`--peak-snr-min`) and the peaks are at least 100 km/s apart (`--min-sep-kms`). Otherwise `double_reason` says why (dbic, weak_peak, peaks_merged, no_single_fit, fit_failed, no_spectrum). The MC errors are built in, so there is no separate MC step or merge as there is for the single fit. Each source gets parametric-bootstrap refits of its best double model (`--n-mc`, default 200, 500 in the slurm script to match step 14), and the z_sys error is added in quadrature to the peak velocities, as for `delta_v_err`. The `flux_blue` and `flux_red` errors from the MC also decide the 2σ peak test. About 3 s per source at 100 refits, so the whole sample takes roughly 30 to 40 minutes at 500, one slurm job.

Writes `MUSE_catalogs/lya_double_fits.csv` (dv_blue, dv_red, peak_sep, flux_blue, flux_red, blue_red_ratio, each with an error, the intrinsic FWHM of each peak, χ² and BIC of both models, dbic, double_peak, double_reason) and `MUSE_subcubes/<ID>/<ID>_lya_double_fit.png`, laid out as `<ID>_lya_fit.png`: the spectrum with the single skewed Gaussian dashed grey, the double fit in red, each peak dotted with its centre marked, systemic in orange and a velocity axis on top, and the pseudo-NB on the right. The peak values are filled for every fitted source, so use them only where `double_peak` = 1. On synthetic spectra at S/N 20 to 30 every double was flagged with its peaks recovered within about 10 km/s and no single line was flagged. At Σ ≈ 4 to 5 no singles were flagged but only 4 of 15 doubles were, so the flag is conservative for faint lines. `delta_v` in the catalogue stays the single fit.
```bash
python fit_lya_double_peak.py --ids 26917 59658          # quick look, replaces these rows
sbatch slurm/run_fit_lya_double_peak.slurm                # all sources, 500 MC refits
python fit_lya_double_peak.py                            # or interactively, 200 refits
python fit_lya_double_peak.py --n-mc 50 --no-plots      # fast, CSV only
```

### D. Detections (rewritten for Isaac IDs)

The detection rule is chosen each time the Lyα catalogue is built (17b), so several rules can be tried and compared. A tested source is a detection if `lya_snr` (Σ_Lyα, the 10 Å sliding-window S/N), rounded to two decimals as it is printed in the PDFs, is at least a fixed S/N (default 5, `LYA_DET_SNR` in `p2_common.py`, so 4.996 counts as 5.00), or instead above a percentile of the false-position S/N (90th, 95th, 98th or 99.5th). Optionally it must also pass one fixed extra cut, applied to every source alike.
- `dv_err`: the Δv error (Lyα MC and z_sys errors in quadrature) at or below a number of km/s.
- `centre_err`: the Lyα line-centre error alone (the MC scatter of the fitted peak, in km/s, no systemic error) at or below a number of km/s.

Every other tested source is a non-detection and stays in the catalogue with all its values. The rule is stored as a tag, e.g. `snr5` (the default), `p95`, `snr5_dverr100` or `p98_cenerr60`, and every PDF and plot made from the catalogue carries it in its file name, so runs with different rules sit side by side. The false positions (Paper 1 Sect. 3.3.3) go through exactly the search the real sources go through. Rerun 15 to 17 whenever steps 9 to 12 change.

Close pairs. The groups of Isaac objects sharing one NIRSpec spectrum stay flagged in the catalogue (`close_pair_group`, `pair_status`). In groups 4 and 5 the spectrum belongs to 50803 and 51988, which stay in the analysis, and the other member (50914, 51887) is a lower-redshift interloper and is shown nowhere (`CLOSE_PAIR_KEEP` in `p2_common.py`). Every member of the other groups is left out of the analysis (`in_analysis` = 0) and gets its own close-pairs PDFs.

**15. Place false positions** `false_pos_grating.py`
500 positions per source outside an 8″ box and 1″ from the edges, with a fixed seed per source. A position is rejected if its starting aperture would contain an object from Isaac's catalogue, i.e. one within 1.0″ (`--clear`: the 0.6″ radius plus about half the PSF FWHM, with 0.15″ more for the pixel snap). Positions are kept about 1.45″ from any spaxel without data (where the subcube runs past the footprint edge), so no shifted aperture can reach one. Only sources whose Lyα band has data in under half its channels (the AO gap, e.g. 65863) are left out. Writes `MUSE_subcubes/<ID>/<ID>_false_positions.csv` and `.png`, and the manifest `MUSE_catalogs/lya_false_pos_sources.csv`.
```bash
python false_pos_grating.py
python false_pos_grating.py --ids <ID>
```

**16. Run them through the search** `optimize_lya_positions_f_grating.py`
Calls `search_offsets` from `optimize_lya_position_grating.py`, so the band, the 10 Å sliding S/N and the 0.4″ shift with its prior are identical to the real sources. Any shift that would bring an Isaac object within 1.0″ of the aperture is blocked, the counterpart of the real sources' nearest-object rule, so the distribution measures noise and never another galaxy. Writes `MUSE_subcubes/<ID>/<ID>_false_optimal_snr.csv`. About 1.5 to 3 minutes per source, so run on slurm.
```bash
cd slurm
N=$(python -c "import pandas as pd; print(len(pd.read_csv('/ceph/cephfs/apatrick/P2/jwst_catalogs/primer_minerva_in_muse_zsys.csv')))")
sbatch --array=0-$((N-1)) run_optimize_lya_positions_f_grating.slurm
cd ..
# one source, quick test
python optimize_lya_positions_f_grating.py --ids <ID> --max-positions 50
```

**17. Detection thresholds** `false_pos_percentiles.py`
Pools every false position and gives the 90th, 95th, 98th and 99.5th percentiles (`--percentiles`), and the percentile each fixed S/N in `--snr-values` (default 5) sits at, for the pool and for each source. Prints how many real sources pass each, and how many offsets were blocked by Isaac objects for the false positions and the real sources, to check the two had similar freedom. Writes `MUSE_catalogs/lya_false_pos_thresholds.csv` (p90, p95, p98, p99_5, snr5_percentile), `lya_false_pos_per_source.csv`, `lya_detections.csv` (status, lya_snr), `plots/lya_false_pos_hist.png` and `plots/lya_false_pos_vs_wavelength.png`. The histogram is as Paper 1 Fig. 5, with the detection threshold dashed and labelled with its percentile (S/N = 5 by default, `--det-snr`, or `--det-percentile 95` to draw a percentile instead), real sources split at it, the 99.5th dotted for comparison with Paper 1, and an optional extra line (`--snr-line`). Its S/N axis is linear to 20 and compressed on a log scale above it, shaded and with break marks (`--xbreak`, `--compress-frac`). The wavelength plot shows each source's own percentile at S/N = 5 (or its own percentile value) against its Lyα wavelength, i.e. is the threshold equally strict at every redshift. If you want a fixed S/N other than 5 in 17b, add it to `--snr-values` here so its percentile is known.
```bash
python false_pos_percentiles.py
python false_pos_percentiles.py --snr-values 4.5 5 --det-percentile 95
```

**17b. Lyα catalogue and detections** `build_lya_catalog.py`
One row per z_sys source with ID, z_sys, z_sys_err, z_sys_grating, z_lya, z_lya_err, lya_snr, delta_v, delta_v_err, `lya_centre_err` (km/s) and `lya_centre_err_ang` (Å), `lya_det` (1 detection, 0 non-detection, blank not tested), `det_reason` (blank, low_snr, dv_err, centre_err, no_err or no_fit), `det_rule` (the tag), lya_status, `close_pair_group`, `pair_status` (single, kept, interloper, close_pair), `in_analysis`, the double-peak columns from step 14b (`double_peak`, `dbic_double`, dv_blue, dv_red, peak_sep and blue_red_ratio with their errors, blank if 14b has not been run) and the photometric UV columns muv_fit, muv_fit_err, beta_fit, beta_fit_err, beta_fit_good, f1500_fit and f1500_fit_err (from step 22).

It prints the S/N threshold with its percentile, how many sources pass it and the counts for the percentiles, and, for the sources above the S/N threshold, the median, robust σ (1.4826 × MAD) and median + 3σ of both `delta_v_err` and `lya_centre_err`, whichever rule is used, to help choose a fixed extra cut. With an extra cut it lists the sources that fail it. no_err means a fit with no MC error, which counts as a non-detection until the MC is rerun.

Writes `MUSE_catalogs/lya_catalog.csv` (the latest run, which every later script reads by default), a kept copy `lya_catalog_<tag>.csv`, and the rule with its counts in `lya_detection_cuts.csv` and `lya_detection_cuts_<tag>.csv`. Close-pair groups use the same function and numbering as `check_close_pairs.py` (it imports it, so keep the two in the same folder). It also prints how many sources are double peaked. Science scripts use `in_analysis == 1` and `lya_det`.
```bash
python build_lya_catalog.py                                                    # S/N >= 5, snr5
python build_lya_catalog.py --det-percentile 95                                # p95
python build_lya_catalog.py --extra-cut dv_err --dv-err-max 100                # snr5_dverr100
python build_lya_catalog.py --det-percentile 98 --extra-cut centre_err --centre-err-max 60
```
Rerun it after step 22 so the UV columns are filled, and after any MC rerun. Then rerun 17c, 17d `--pdf-only`, 26 and 26b for that rule. To remake the outputs of an older rule without rebuilding, give those scripts `--catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_<tag>.csv`.

**17c. Lyα fits by eye** `make_lya_fit_pdfs.py`
Three PDFs of the `_lya_fit.png` figures (spectrum, fit, detection window and pseudo-NB), four per page sorted by S/N, each captioned with Σ, z_sys, Δv with its error and the Lyα centre error, the detection state (with the reason for a non-detection) and any fit flag. The first page gives the rule and lists the sources. Writes `plots/lya_fit_pdfs/lya_fits_<tag>_detections.pdf`, `lya_fits_<tag>_non_detections.pdf` (both `in_analysis == 1`) and `lya_fits_<tag>_close_pairs.pdf` (every `close_pair` source, with its group number). Interlopers and untested sources are in none. If step 14b has been run each caption also says single or double with its ΔBIC, and the red and blue Δv for a double. `--figure double` makes the same three PDFs from the `_lya_double_fit.png` figures instead, named `lya_double_fits_<tag>_<set>.pdf`. Needs 17b first.
```bash
python make_lya_fit_pdfs.py
python make_lya_fit_pdfs.py --figure double
python make_lya_fit_pdfs.py --catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_p95.csv
```

**17d. Lyα against the systemic line** `plot_lya_vs_systemic.py` (env312, the [OIII] or Hα fit is repeated with LiMe)
Two figures per source in `MUSE_subcubes/<ID>/`, `<ID>_lya_systemic_panels.png` and `<ID>_lya_systemic_overlay.png`, then six PDFs in `plots/lya_systemic_pdfs/`, `lya_systemic_<panels|overlay>_<tag>_<detections|non_detections|close_pairs>.pdf`, split and sorted as in 17c. The figures don't depend on the detection rule, so after rebuilding the catalogue `--pdf-only` remakes the PDFs in seconds without refitting (and runs in env39). An `--ids` run makes only the figures, so it can't overwrite the full PDFs with a few sources.
```bash
conda activate env312
python plot_lya_vs_systemic.py --ids 23070 50803        # test, figures only
python plot_lya_vs_systemic.py                          # all figures and PDFs
python plot_lya_vs_systemic.py --pdf-only               # PDFs only, for the latest catalogue
sbatch slurm/run_plot_lya_vs_systematic.slurm
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

**21b. Match to PRIMER + MINERVA** `match_primer_minerva.py` (old, JELS IDs)
No longer needed for steps 22 and 23, which read Isaac's catalogue by `Number` directly. Steps 24 and 28 still read its output `jwst_catalogs/primer_minerva_by_JELS_ID.csv` until they are rewritten.

**22. M_UV and β** `fit_muv_beta.py` (rewritten for Isaac IDs)
Power-law fit f_λ ∝ λ^β to the rest-frame UV photometry, as Paper 1 Sect. 2.4 (after Pirie et al. 2025), at z_sys for every source in `primer_minerva_in_muse_zsys.csv`. The photometry is read from `cosmos_primer_minerva_production.fits` by `Number`, with −99 (band not imaged) treated as missing. Total fluxes by default (aperture × `total_correction`, `--phot aperture` for the aperture fluxes), Isaac's errors as they are. A band is used if its blue half-power edge is redward of 1250 Å rest and its pivot is at or below 3000 Å rest, and at least two bands are needed. Below z ≈ 3.5 only F814W, F090W and F115W qualify, so β is poorly constrained there. Errors are inflated by √χ²_ν when χ²_ν > 1. No dust correction. `beta_fit_good` is 1 where σ_β < 0.5 (Paper 1). Isaac's Flag is carried, and `phot_flag` is 1 where it contains 1, 2 or 3. Writes `jwst_catalogs/muv_beta_fits.csv` (muv_fit, beta_fit, their errors, f1500_fit in nJy, the bands used, χ², fit_flag and every band's flux) and `MUSE_subcubes/<ID>/<ID>_muv_beta_fit.png`. `--ids` replaces only those rows.
```bash
python fit_muv_beta.py --dry-run
python fit_muv_beta.py
python fit_muv_beta.py --ids <ID>
```

**23. M_UV and β checks** `muv_beta_diagnostics.py` (rewritten)
Three panels in `plots/muv_beta_diagnostics.png`: β against z_sys (coloured by the number of bands, open where σ_β ≥ 0.5), muv_fit against a single-band M_UV from the band nearest rest 1500 Å (coloured by that band's rest wavelength, since the two differ by about −2.5(β+2)log₁₀(λ/1500)), and σ_β against the number of bands. Prints the median σ_β per number of bands and any source where the two M_UV differ by more than 0.5 mag. Writes `jwst_catalogs/muv_beta_diagnostics.csv`.
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

**25. Lyα against the systemic line, per source**
Moved to step 17d.

**26. Science plots** `plot_lya_science.py` (rewritten for Isaac IDs)
The same plots as before (histograms of FWHM, Δv, z and M_UV, Δv against FWHM with Verhamme+18, the FWHM model comparison with Prieto-Lyon+25, Mason+18 and Mason+19, Δv and FWHM against M_UV, each also split by redshift and M_UV) for the detections and for the non-detections (`lya_det` 1 or 0), using only `in_analysis == 1`, so close pairs and interlopers are left out and 50803 and 51988 are kept. Untested sources (65863) are in neither set. M_UV is muv_fit where muv_fit_err < 0.5. Reads `MUSE_catalogs/lya_catalog.csv` and `lya_fits_mc.csv` (for the FWHM). Writes `plots/lya_science/<tag>_<detections|non_detections>_<plot>.png`, e.g. `snr5_detections_dv_vs_fwhm.png`, 25 per set, where `<tag>` is the catalogue's detection rule. `--fwhm-kind obs` adds an `obs_` tag. `--catalog` points at a kept `lya_catalog_<tag>.csv` to remake an older rule's plots.
```bash
python plot_lya_science.py
python plot_lya_science.py --sets detections
```

**26b. Check the Lyα widths** `checks/check_lya_fwhm.py`
Tests whether the fitted FWHM is a fair measure of the lines, for the detections with `in_analysis == 1` (`--sample non_detections` or `all` for the others). Four panels in `plots/lya_fwhm_checks_<tag>.png`. (a) The FWHM measured straight from each aperture spectrum (1-pixel Gaussian smoothing, half-maximum crossings) against the convolved model measured on the same pixels with the same code. The direct numbers are larger than `fwhm_obs_kms` because of the smoothing and sampling, so compare them only with the model column next to them. (b) FWHM_int against Σ_Lyα, for noise bias. (c) FWHM_int against the continuum residual in the fit window (pixels where the model is below 5% of its peak), as S/N. (d) FWHM_int against the fraction of MC refits with α at either bound (0 or 15), which is near 0.5 when the skewness is unconstrained. Writes `MUSE_catalogs/lya_fwhm_checks_<tag>.csv`.
```bash
python checks/check_lya_fwhm.py
python checks/check_lya_fwhm.py --sample all
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
| a Lyα fit setting | 13, 14, 14b, 17b, 17c, 18 to 21, 28 |
| a double-peak setting (14b) | 14b, 17b, 17c |
| a by-eye label | edit `lya_manual_labels.csv`, then 21, 28 |
| photometry catalogue | 22, 23, 17b, 26, then 24, 27, 28 |
| Hα method | 24, 27, 28 |
| SED or fesc outputs | 28 |
| the close-pair list (step 1) or `CLOSE_PAIR_KEEP` | 17b, 17c, 17d `--pdf-only`, then anything that reads `lya_catalog.csv` |
| the detection rule (17b options) | 17b, 17c, 17d `--pdf-only`, 26, 26b |
| an MC error (14*) | 17b, then as for the detection rule |
| M_UV or β (22) | 23, 17b, 26 |

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