# P2 pipeline

Spectroscopy of Lyα emitters at 3 < z < 6 with VLT/MUSE and JWST/NIRSpec. Every script lives in this folder and is run from the terminal on the cluster.

```bash
conda activate env39 (or env312 for the lime fits)
cd /ceph/cephfs/apatrick/P2/scripts
```

All data paths are under `/ceph/cephfs/apatrick/P2`. Each script prints the full paths of what it reads and writes. The end product is `master_catalog/p2_master_catalog.csv`, rebuilt from the step outputs by `build_master_catalog.py` (step 28). Rerun any step, then step 28, and the master is up to date.

## Rebuild in progress (October 2026)

The pipeline is being rewritten step by step to start from Isaac's PRIMER + MINERVA catalogue. Every source is now keyed on Isaac's catalogue `Number` (the `ID` column), not the JELS ID. Sections A to D below are rewritten (sample, z_sys, MUSE Lyα, detection thresholds, the Lyα catalogue and the Lyα against systemic figures), as are steps 22, 23 (M_UV and β) and 26 (science plots). Steps 14b (double-peak fit), 14c (f_blue), 23b (SED properties), 24b (f_esc) and 24c (Hα checks) are new, step 24 (Hα) is rewritten, and step 26 makes every science figure. Steps 18 to 21, 21b, 27 and 28 still describe the old JELS-ID version and will be updated as each is redone. Step 25 has moved to 17d.

## Systemic redshift rule

One rule is used everywhere, defined once in `p2_common.py` (`pick_zsys`).

1. [OIII] inside its grating's nominal range, A/noise ≥ 5
2. otherwise Hα inside its grating's nominal range, A/noise ≥ 5
3. otherwise [OIII] outside the nominal range, A/noise ≥ 5
4. otherwise Hα outside the nominal range, A/noise ≥ 5
5. otherwise no z_sys, and the source leaves the z_sys sample

The nominal ranges (`pc.NIRSPEC_NOMINAL_UM`, JDox) are G235M 1.66 to 3.07, G235H 1.66 to 3.05, G395M 2.87 to 5.10 and G395H 2.87 to 5.14 µm. The DJA spectra run past them, but there the calibration is less certain and above about 3.4 µm in G235 second-order light contaminates. 18264 had [OIII] at 3.27 µm in G235M 139 km/s from its Hα inside G395M, and Hα fluxes past the range disagreed with G395 by factors of 0.6 to 3.2, so a line inside the range is always preferred. `z_sys_in_range` is 0 where z_sys still comes from outside it (in October 2026 19338, 51717 and 52330).

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
Reads the per-grating [OIII] and Hα fits (`lime_<line>_fits.csv`) and takes each line's best successful fit inside and outside its grating's nominal range (rule above). Writes `jwst_catalogs/systemic_redshifts.csv` (every source, best z per line, with `z_<line>_ext` columns for the best fit outside the range, z_sys where there is one) and `jwst_catalogs/primer_minerva_in_muse_zsys.csv` (only the sources with a z_sys, with `z_sys`, `z_sys_err`, `z_sys_snr`, `z_sys_line`, `z_sys_grating`, `z_sys_in_range` and `dv_sys_dja_kms`). Prints why each dropped source has no z_sys, the sources whose z_sys is outside the range, and every source whose z_sys changed since the last run, with the shift in km/s. Those need the MUSE steps rerun with `--ids`.
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
Fits a skew-normal convolved with the MUSE LSF within ±25 Å of `best_center`, giving z_lya, Δv, flux and FWHM, as Paper 1 but with the LSF in the model. `lya_snr` is the Σ_Lyα from step 12, the S/N for detection. As in Paper 1 the fitted flux is checked against a direct integration over the line window (`flux_line`, the central 90% of the fitted line), and the pseudo-NB is collapsed over the 10 Å detection window (`--nb-window line` uses the line window). Every source is fitted. Writes `MUSE_catalogs/lya_fits.csv`, `<ID>_lya_nb.fits` and `<ID>_lya_fit.png` (spectrum, fit and detection window on the left, ±150 Å about systemic with 2-pixel display bins, `--plot-halfwidth` and `--plot-bin`, and the 3″ pseudo-NB on the right). `--plots-only` redraws the figures from `lya_fits.csv` and the saved pseudo-NBs in about a minute, without refitting or writing anything else.
```bash
python fit_lya_properties_grating.py
python fit_lya_properties_grating.py --ids <ID>
python fit_lya_properties_grating.py --plots-only                  # figures only
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

**14b. Double-peak fit** `fit_lya_double_peak.py` (new)
A second fit to every source with a single fit, to find lines with a blue peak as well as the red one. The model is the single fit's skewed Gaussian for the red peak plus a Gaussian for the blue peak, both convolved with the MUSE LSF. The red peak has exactly the single fit's limits (location within `best_center` ± 10 Å, α from 0 to 15) and the blue centre is free between −1000 and +100 km/s of systemic Lyα. With no blue flux the double is the single fit, so the comparison asks only whether a blue peak is needed, and the red peak keeps its asymmetric tail. (A first version used two symmetric Gaussians. It lost on the red tail what it gained from the blue peak, as in 31364, and split one asymmetric line into a narrow and a broad peak, as in 34630.) The red peak's Δv is the peak of its LSF-convolved profile, as for the single fit's `delta_v`. It uses the same pixels as the single fit (`best_center` ± 25 Å) and the best of seven starts.

The two models are compared with the BIC, χ² + k ln n (k = 4 single, 7 double, n the pixels fitted), and `dbic` = BIC_single − BIC_double, so positive favours the double. A source is flagged `double_peak` = 1 if dbic > 2 ("positive" evidence on the Kass & Raftery 1995 scale, `--dbic-min`), both peak fluxes are at least 2σ (`--peak-snr-min`), the peaks are at least 100 km/s apart (`--min-sep-kms`) and the blue peak is blueward of systemic. Otherwise `double_reason` says why (dbic, weak_peak, peaks_merged, blue_not_blueward, no_single_fit, fit_failed, no_spectrum). With about 41 pixels, dbic > 2 means the double lowers χ² by at least about 13.

The threshold comes from synthetic spectra like ours (Σ ≈ 7 to 18, noise and pixels as MUSE). No single line reached dbic > 2 (0 of 90, including 30 with long red tails like 34630). A blue peak with a quarter of the red flux at −330 to −420 km/s was flagged in 17 of 30 at dbic > 2, against 10 of 30 at dbic > 6, and every blue peak with 0.6 of the red flux was flagged. So faint blue peaks are still missed about half the time, and the double fraction is a lower limit. For the flagged sources the red peak was recovered within a median 7 km/s and the blue within a median 20 km/s (largest 56 km/s, against typical MC errors of 40 km/s).

The MC errors are built in, so there is no separate MC step or merge. Each source gets parametric-bootstrap refits of its best double model (`--n-mc`, default 200, 500 in the slurm script to match step 14), and the z_sys error is added in quadrature to the peak velocities, as for `delta_v_err`. The flux errors from the MC also decide the 2σ peak test. About 7 s per source at 100 refits, so the sample takes roughly 45 to 90 minutes at 500, one slurm job.

Writes `MUSE_catalogs/lya_double_fits.csv` (dv_blue, dv_red, peak_sep, flux_blue, flux_red, flux_total, blue_red_ratio, the red peak's z_lya_red, wavelength and centre precision, the intrinsic FWHM of each peak and the observed FWHM of the red one, each with an MC error, whether the red peak is resolved, the seven fit parameters, χ² and BIC of both models, dbic, double_peak, double_reason) and `MUSE_subcubes/<ID>/<ID>_lya_double_fit.png`, laid out as `<ID>_lya_fit.png` with the same ±150 Å range and 2-pixel display bins (`--plot-halfwidth`, `--plot-bin`): the spectrum with the single skewed Gaussian dashed grey, the double fit in red, the blue Gaussian and red skewed peak dotted with their peaks marked, systemic in orange and a velocity axis on top, the S/N, Δv_blue, Δv_red and peak separation in the corner (ΔBIC, B/R and the preferred model are in the CSV and the PDF captions), and the pseudo-NB on the right. The CSV has the double-fit values for every fitted source. The catalogue (17b) then picks the preferred fit for each source. `--plots-only` redraws the figures from the parameters saved in the CSV in about a minute, without refitting or changing the CSV, e.g. with `--plot-halfwidth 60` to zoom in on the peaks.
```bash
python fit_lya_double_peak.py --ids 26917 59658          # quick look, replaces these rows
sbatch slurm/run_fit_lya_double_peaks.slurm               # all sources, 500 MC refits
python fit_lya_double_peak.py                            # or interactively, 200 refits
python fit_lya_double_peak.py --n-mc 50 --no-plots      # fast, CSV only
python fit_lya_double_peak.py --plots-only               # redraw the figures only, ±150 Å
python fit_lya_double_peak.py --plots-only --plot-halfwidth 60 --ids 31364   # zoomed in
```

**14c. Blue flux fraction** `measure_lya_shape.py` (new)
f_blue = F(v < 0) / F(−1000 < v < +1000 km/s), the fraction of the Lyα flux blueward of systemic, for every source with a Lyα fit (`--vmin`, `--vmax` set the window). A purely red line gives 0 and a line symmetric about systemic 0.5. It is the v < 0 fraction of Trainor et al. (2015), and the blue/red ratios of Erb et al. (2014) and Hayes et al. (2021) convert as R/(1 + R). It always uses the catalogue's `lya_model`. Both scripts take it from `pc.preferred_lya_model` with the same inputs (z_lya in `lya_fits_mc.csv` and double_peak in `lya_double_fits.csv`), so the double fit where double_peak = 1 and the single skewed Gaussian otherwise. There is no fallback, so a source whose preferred model has missing parameters gets no f_blue (`shape_status` no_double_params or no_fit_params). Rerun it after 14 and 14b. Three versions are measured.
- `f_blue_int`, from the model before the LSF. Not smeared by the MUSE resolution, but for a single skewed Gaussian it is set mostly by the model shape (α ≥ 0 puts the tail on the red side), so singles come out near 0.
- `f_blue_conv`, the same model with the LSF, to compare like for like with the data.
- `f_blue_data`, the flux summed either side of systemic in the aperture spectrum, with fractional pixels at the ends. Model independent, but smeared by the LSF, so lines near systemic gain blue flux from the instrument alone.

`f_blue_int_single` keeps the skewed-Gaussian value for every source. The model fractions are exact, from the skew-normal and Gaussian cumulative distributions. Errors come from `--n-mc` (200) draws, each with z_sys drawn from its error (so the dividing line moves) and either a refit of the preferred model to the best-fit model plus noise (same pixels and limits as 13 and 14b) or the spectrum plus noise for the data version. Each value has `_err_lo` and `_err_hi` (to the 16th and 84th percentiles) and `_err` (their mean). On synthetic spectra the model values match the input exactly and the data version is unbiased, with a scatter of about 0.02 to 0.05 at Σ ≈ 10. Writes `MUSE_catalogs/lya_shape.csv` and `MUSE_subcubes/<ID>/<ID>_lya_fblue.png` (the line in velocity with the blue side shaded, data, model with and without the LSF, and the three values). About 15 s for six sources at 200 draws, so a few minutes for the sample, no slurm needed. Then rerun 17b.
```bash
python measure_lya_shape.py
python measure_lya_shape.py --ids 31364 26917       # replaces only these rows
python measure_lya_shape.py --n-mc 50 --no-plots    # quick
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
One row per z_sys source. `lya_model` says which fit is preferred, single (the skewed Gaussian of steps 13 and 14) or double (where step 14b set double_peak = 1), and every line quantity comes from that fit, the skewed Gaussian for a single and the red peak for a double, all with MC errors.

| Column | Meaning |
|---|---|
| ID, z_sys, z_sys_err, z_sys_grating, lya_snr | |
| `lya_model` | single or double, blank with no fit |
| z_lya, z_lya_err | from the vacuum Lyα (red) peak |
| delta_v, delta_v_err | km/s, z_sys error included |
| `lya_centre_err`, `lya_centre_err_ang` | line-centre precision alone, km/s and Å |
| fwhm_int_kms, fwhm_obs_kms (+ _err), fwhm_lsf_kms, `fwhm_resolved` | intrinsic and observed FWHM, the LSF FWHM, 1 if resolved |
| lya_flux, lya_flux_err | 10⁻²⁰ erg s⁻¹ cm⁻², the whole line (blue + red for a double) |
| lya_det, det_reason, det_rule, lya_status | detection (1, 0, blank not tested), why not, the rule tag |
| close_pair_group, pair_status, in_analysis | close pairs, as below |
| `dbic_double`, `double_reason` | the BIC difference for every source 14b ran on, and why the single fit was kept |
| dv_blue, peak_sep, blue_red_ratio, flux_blue, flux_red, fwhm_blue_kms (+ _err) | only where lya_model is double, blank otherwise |
| delta_v_single (+ _err) | the skewed-Gaussian Δv for every source, to see how much the choice moves it |
| f_blue_int, f_blue_conv, f_blue_data (each + _err_lo, _err_hi, _err), f_blue_int_single (+ _err) | blue flux fraction (step 14c) |
| z_sys_line | [OIII] or Hα, the line that set z_sys |
| ha_fwhm_int_kms (+ _err), ha_fwhm_obs_kms, ha_fwhm_unresolved, ha_fwhm_grating | best Hα FWHM from `jwst_catalogs/lime_Ha_summary.csv` (step 4), the _int value corrected for the NIRSpec resolution |
| muv_fit, muv_fit_err, beta_fit, beta_fit_err, beta_fit_good, f1500_fit, f1500_fit_err | photometric UV fit (step 22) |
| fesc_lya, fesc_lya_err_lo, fesc_lya_err_hi, fesc_lya_err, fesc_lya_is_limit, fesc_status, fesc_lya_sed (+ errs), lya_flux_lim_5sig | Lyα escape fraction (24b), blank if made under another rule |
| ha_case, ha_source, dust_method, balmer_spectra, a_ha (+ errs), ha_flux_corr (+ errs) | how Hα was measured and dust corrected (24) |
| sed_ok, sed_zred_ok, sed_\<q\> with _err_lo, _err_hi, _err | Ken's SED fits (step 23b), q = muv, logM, logsfr10, logsfr100, logssfr10, logssfr100, av, aha, logxion. Blank where sed_ok is 0 |

Without `lya_double_fits.csv` every source is single and it says so. Without `jwst_catalogs/sed_properties.csv` the SED columns are blank and it says so. The same holds for `MUSE_catalogs/lya_shape.csv` (f_blue) and `lime_Ha_summary.csv`, and any source whose f_blue was measured with a different model from its `lya_model` (a stale `lya_shape.csv`) gets blank f_blue columns and is listed, so another model's value never reaches the catalogue. Rerun 14c, then 17b. It prints how many sources have a usable SED and lists any whose SED fit was made at a redshift far from z_sys. The extra detection cuts use the errors of the preferred fit.

It prints the S/N threshold with its percentile, how many sources pass it and the counts for the percentiles, and, for the sources above the S/N threshold, the median, robust σ (1.4826 × MAD) and median + 3σ of both `delta_v_err` and `lya_centre_err`, whichever rule is used, to help choose a fixed extra cut. With an extra cut it lists the sources that fail it. no_err means a fit with no MC error, which counts as a non-detection until the MC is rerun.

Writes `MUSE_catalogs/lya_catalog.csv` (the latest run, which every later script reads by default), a kept copy `lya_catalog_<tag>.csv`, and the rule with its counts in `lya_detection_cuts.csv` and `lya_detection_cuts_<tag>.csv`. Close-pair groups use the same function and numbering as `check_close_pairs.py` (it imports it, so keep the two in the same folder). It also lists the sources with the double preferred, with Δv_red beside the single-fit Δv. Science scripts use `in_analysis == 1` and `lya_det`.
```bash
python build_lya_catalog.py                                                    # S/N >= 5, snr5
python build_lya_catalog.py --det-percentile 95                                # p95
python build_lya_catalog.py --extra-cut dv_err --dv-err-max 100                # snr5_dverr100
python build_lya_catalog.py --det-percentile 98 --extra-cut centre_err --centre-err-max 60
```
Rerun it after steps 14c, 22 and 23b so the f_blue, UV and SED columns are filled, and after any MC rerun. Then rerun 17c, 17d `--pdf-only`, 26 and 26b for that rule. To remake the outputs of an older rule without rebuilding, give those scripts `--catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_<tag>.csv`.

**17c. Lyα fits by eye** `make_lya_fit_pdfs.py`
Three PDFs showing each source's preferred fit, the `_lya_fit.png` figure (spectrum, skewed Gaussian, detection window and pseudo-NB) for a single and the `_lya_double_fit.png` figure for a double. Four per page sorted by S/N, each captioned with Σ, z_sys, the model and its ΔBIC, Δv (Δv_red for a double) with its error and the Lyα centre error, the blue peak and B/R for a double, the detection state (with the reason for a non-detection) and any fit flag. The first page gives the rule and lists the sources. Writes `plots/lya_fit_pdfs/lya_fits_<tag>_detections.pdf`, `lya_fits_<tag>_non_detections.pdf` (both `in_analysis == 1`) and `lya_fits_<tag>_close_pairs.pdf` (every `close_pair` source, with its group number). Interlopers and untested sources are in none. `--figure single` or `--figure double` shows that figure for every source instead, in `lya_single_fits_<tag>_<set>.pdf` or `lya_double_fits_<tag>_<set>.pdf`, to compare the two fits side by side. Needs 17b first.
```bash
python make_lya_fit_pdfs.py                    # preferred fit per source
python make_lya_fit_pdfs.py --figure double    # every source's double fit
python make_lya_fit_pdfs.py --figure single    # every source's skewed Gaussian
python make_lya_fit_pdfs.py --catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_p95.csv
```

**17d. Lyα against the systemic line** `plot_lya_vs_systemic.py` (env312, the [OIII] or Hα fit is repeated with LiMe)
Two figures per source in `MUSE_subcubes/<ID>/`, `<ID>_lya_systemic_panels.png` and `<ID>_lya_systemic_overlay.png`, then six PDFs in `plots/lya_systemic_pdfs/`, `lya_systemic_<panels|overlay>_<tag>_<detections|non_detections|close_pairs>.pdf`, split and sorted as in 17c. The figures don't depend on the detection rule, so after rebuilding the catalogue `--pdf-only` remakes the PDFs in seconds without refitting (and runs in env39). An `--ids` run makes only the figures, so it can't overwrite the full PDFs with a few sources. Sources with `lya_model` double are drawn with the double-peak model (from `lya_double_fits.csv`, no grey MC band) and the Δv line marks the red peak, so remake their figures once 14b and 17b have run.
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

**23b. SED properties** `build_sed_catalog.py` (new)
Reads Ken's BAGPIPES fits of the z_sys sample, `jwst_catalogs/primer_minerva_full_in_muse_with_grating_zsys_ceridwen_output.fits`, keyed on Isaac's `Number` and fitted at a fixed redshift (`zred_fixed`). The fits are assumed to use total fluxes, like muv_fit, so no aperture correction is applied (to confirm with Ken). Four checks are printed. Every Number appears once and the IDs match the z_sys sample, with any missing on either side listed. RA/Dec agree with the sample to 0.1″ (`--pos-tol`). `zred_fixed` is within 100 km/s of the current z_sys (`--dz-tol-kms`), otherwise `sed_zred_ok` = 0, which catches a fit made at an old redshift. Every p16 ≤ p50 ≤ p84 is finite, otherwise `sed_ok` = 0 and `build_lya_catalog.py` blanks that source's SED values. Each quantity q gets `sed_<q>` (p50), `sed_<q>_err_lo` (p50 − p16), `sed_<q>_err_hi` (p84 − p50) and `sed_<q>_err` (their mean).

| Column | Ken's column | |
|---|---|---|
| sed_muv | MUV | AB mag. err_lo is towards brighter |
| sed_logM | log10_mass_surviving | log M*/M☉ |
| sed_logsfr10, sed_logsfr100 | log10_sfr10, log10_sfr100 | log M☉/yr, 10 and 100 Myr |
| sed_logssfr10, sed_logssfr100 | log10_ssfr10_surviving, log10_ssfr100_surviving | log /yr |
| sed_av, sed_aha | A_V, A_Ha | mag |
| sed_logxion | log10_xion | log ξ_ion |
| sed_zred, dv_sed_zsys_kms, sed_zred_ok, sed_ok, sed_log_evidence, sed_log_likelihood | | checks and fit quality |

The 10 Myr SFR and sSFR are often poorly constrained, since they depend on a recent burst the photometry barely sees, so bin by the 100 Myr values. The script also prints the sample medians and how many sources have SFR errors above 0.5 dex. It plots sed_muv against the interim muv_fit with the 1:1 line, and their difference against z_sys (open points have muv_fit_err ≥ 0.5), and prints the median offset, the robust scatter and any source more than 0.5 mag apart (`--muv-flag`). Writes `jwst_catalogs/sed_properties.csv` and `plots/sed_muv_comparison.png`. Then rerun 17b and 26.
```bash
python build_sed_catalog.py --dry-run    # checks only, nothing written
python build_sed_catalog.py
python build_lya_catalog.py
python plot_lya_science.py
```

**24. Corrected Hα** `build_ha_flux_corrections.py` (rewritten for Isaac IDs)
The total, dust-corrected Hα flux of every z_sys source, F_Hα,corr = F_Hα,total × 10^(0.4 A_Hα). Each source falls in one case, recorded in `ha_case`.

| ha_case | ha_source | dust_method | Hα | A_Hα |
|---|---|---|---|---|
| 1 | spec | balmer | NIRSpec Hα × slit factor S | Balmer decrement |
| 2 | spec | sed | NIRSpec Hα × slit factor S | Ken's SED A_Hα (no usable Hβ) |
| 3 | phot | sed | excess in the band containing Hα (already total) | Ken's SED A_Hα |
| 0 | none | none | no Hα, or no A_Hα (`ha_reason` says why) | |

Hβ is only used in the decrement, never in place of Hα. A line counts at flux S/N ≥ 3 (`--ha-snr-min`, `--hb-snr-min`), and only inside the grating's nominal range (G235M 1.66 to 3.07, G235H 1.66 to 3.05, G395M 2.87 to 5.10, G395H 2.87 to 5.14 µm). The DJA spectra continue beyond these limits, but the flux there is poorly calibrated and contaminated by second-order light. In the first run 11 G235M Hα fluxes at 3.1 to 4.5 µm gave Balmer decrements with a median of 5.2, against 3.6 inside the range (`--allow-extended` keeps them). The excluded fits are listed.
- **Slit factor.** S = f_ν(Isaac total photometry) / f_ν(synthetic, spectrum through the same band), in a band with transmission ≥ half its peak at the line, the line on the spectrum, ≥ 70 per cent of the band covered and both fluxes at S/N ≥ 3. Medium bands first. A direct factor outside 0.5 to 5 (`--slit-range`) is not believed (below 0.5 the spectrum holds twice the galaxy's total light, e.g. 55001 gave 0.01 and f_esc 23). Those, sources with photometry Flag 1, 2 or 3, and sources with no passing band get the sample median with its NMAD (`ha_slit_source` direct or sample_median, `ha_slit_note` says why, with any rejected value). For each line a spectrum with a direct factor is chosen first, then the highest S/N.
- **Balmer decrement.** `balmer_spectra` same where one spectrum has both lines (the observed ratio, as the slit loss largely cancels, `--balmer-basis slitcorr` corrects both first), different where they are in different spectra (each slit-corrected first). E(B−V) = 2.5/(k_Hβ − k_Hα) log10(R/2.86), A_Hα = k_Hα E(B−V), 0 below 2.86, Calzetti k_Hα = 3.33, k_Hβ = 4.60.
- **Photometric Hα.** F_Hα = Δf_ν c ∫T/λ dλ / Σ r_i T(λ_i) λ_i, with Δf_ν the band flux minus a power-law continuum through the nearest band on each side free of strong lines, and the sum over Hα, [NII] at the sample median [NII]/Hα and optional [SII] (`--sii-ha`, default 0). Computed for every source, so it is also checked against the spectral Hα.
- **Errors.** 5000 Monte Carlo draws of every input (line fluxes, slit factors, photometry, the SED A_Hα as a split normal), with the decrement redone in each draw. The draws of `ha_flux_corr` and `ha_flux_corr_sed` are saved for step 24b.
- `ha_flux_corr_sed` is every source's Hα corrected with the SED A_Hα whatever its case, to test the dust method.

It prints the cases (all, in the analysis, Lyα detections), the Lyα detections with no Hα and why, slit factors above 3 or below 0.7, the median slit factor per grating, the same-spectrum decrement observed against slit-corrected, A_Hα Balmer against SED, photometric against spectral Hα per band, and the case 3 sources. Writes `jwst_catalogs/ha_flux_corrections.csv`, `jwst_catalogs/ha_flux_corrections_mc.npz` and `plots/ha_flux_corrections.png` (slit factors, A_Hα Balmer against SED, photometric/spectral Hα, A_Hα used against z by case). Needs the NIRCam filter curves in `jwst_catalogs/nircam_filters/` (`--download-filters` once) and `sed_properties.csv` (23b). On synthetic galaxies with a known intrinsic Hα and a 40 per cent shutter, all three cases recovered the intrinsic Hα within 4 per cent.
```bash
python build_ha_flux_corrections.py --download-filters    # only if the curves are missing
python build_ha_flux_corrections.py
python build_ha_flux_corrections.py --balmer-basis slitcorr
```

**24b. Lyα escape fractions** `measure_lya_fesc.py` (new)
f_esc = F_Lyα / (8.7 F_Hα,corr) (Paper 1 Eq. 7), with F_Lyα the observed `lya_flux` (0.6″ aperture, not dust-corrected) and F_Hα,corr from step 24. For Lyα non-detections a 5σ upper limit, F_lim = 5 σ_rms √N_pix Δλ (Paper 1 Eq. 6), in a window the median observed FWHM of the detections wide, centred on Lyα at z_sys shifted by the median detected Δv (`--ul-centre systemic` to centre on systemic). Errors from 5000 draws, F_Lyα from its error and F_Hα,corr from the saved draws of step 24. `fesc_lya_sed` uses the SED A_Hα for every source. It prints the status counts, every detection with its case, A_Hα and both f_esc, the median per case, f_esc Balmer/SED for case 1, the range of the limits and the reverse Kaplan-Meier sample median (lifelines, bootstrap errors) overall, with SED dust throughout and per case. Writes `MUSE_catalogs/lya_fesc.csv` and `plots/lya_fesc.png` (f_esc against z and M_UV, symbols by case, limits open). Then rerun 17b, which joins f_esc and the Hα flags, and blanks f_esc if `lya_fesc.csv` was made under another detection rule.
```bash
python build_lya_catalog.py          # so lya_flux and lya_det are current
python build_ha_flux_corrections.py
python measure_lya_fesc.py
python build_lya_catalog.py          # joins fesc_lya and the flags
```

**24c. Check the Hα corrections** `checks/check_ha_corrections.py` (new)
Changes no values. For `in_analysis == 1` (`--sample all` or `detections`), Lyα detections filled and non-detections open. It recomputes the slit factor S in every NIRCam band each spectrum covers, not only the band used, and writes them to `jwst_catalogs/ha_slit_factors_all_bands.csv` (`--reuse-bands` reads it back instead of reopening the spectra). Bands outside the grating's nominal range are shown but left out of every number. Four figures in `plots/ha_checks/`.
- `slit_factors.png`: S against wavelength per spectrum, G395 against G235 at a common 3 µm (each band moved with the median slope), max/min S within a spectrum, S_Hα used against Isaac's `total_correction` and against z, and S_Hβ against S_Hα in one spectrum.
- `balmer_dust.png`: A_Hα Balmer against SED (colour Hβ S/N), the decrement against z, S_Hα/S_Hβ for the same-spectrum decrements, the Balmer minus SED offset against Hβ S/N and stellar mass, and A_Hα used by case.
- `ha_phot_vs_spec.png`: photometric against slit-corrected spectral Hα, and their ratio against S, z and photometric S/N.
- `fesc_checks.png`: f_esc Balmer against SED for case 1, f_esc against S and against A_Hα, and f_esc by case.

It prints the case counts, median S per grating, the median slope d ln S / d ln λ and the A_Hα error it puts on a same-spectrum observed decrement, the G395/G235 ratio, the Balmer minus SED offset, the photometric/spectral ratio, and every flagged source. Flags (`jwst_catalogs/ha_correction_flags.csv`): `slit_out` (direct S_Hα outside `--slit-warn` 0.8 to 3), `slit_median`, `grating` (G395/G235 S differ by more than `--ratio-warn` 1.5), `chromatic` (max/min S in one spectrum above 1.5), `balmer_low` (below 2.86 by > 2σ), `dust_offset` (|Balmer − SED| > 0.5 mag and > 3σ), `phot_spec` (outside `--phot-warn` 0.5 to 2), `fesc_gt1` and `fesc_dust` (Balmer/SED f_esc outside 1/3 to 3).
```bash
python checks/check_ha_corrections.py
python checks/check_ha_corrections.py --sample all
python checks/check_ha_corrections.py --reuse-bands
```

### G. Figures

**25. Lyα against the systemic line, per source**
Moved to step 17d.

**26. Science plots** `plot_lya_science.py` (rewritten, every figure)
Makes every science figure with a physical reason to exist, plus histograms of every property, so the paper figures can be chosen afterwards. Detections by default (`--sets detections non_detections` adds the other set), `in_analysis == 1` only. Reads only `MUSE_catalogs/lya_catalog.csv`, so Δv, FWHM and f_blue are those of the preferred fit (the red peak for a double). M_UV is `sed_muv` (step 23b) by default and the galaxy properties are Ken's SED values, all with asymmetric errors (`--muv-source fit` uses the interim muv_fit, names gain `muvfit_`).

Writes one folder per rule and set, `plots/lya_science/<rule>_<set>/` (e.g. `plots/lya_science/snr5_detections/`), with one PNG per figure named `<section>_<plot>[_<split>].png`, `stats.csv` and `all_figures.pdf` (every figure, one per page, to browse). `plots/lya_science/<rule>_det_vs_nondet/` holds the comparisons of the two sets. Reruns overwrite.

| Section | Figures | Physics and predictions |
|---|---|---|
| `hist_` | every property, plus `hist_overview.png`. FWHM, Δv and f_blue_int also by z | distributions |
| `sys_` | `dv_vs_fwhm`, `dv_vs_fwhm_verhamme` (with a linear fit), `verhamme_resid` (histogram, and against z and M_UV), `dvred_vs_halfsep` and `recovery_doubles` (doubles) | can the profile recover z_sys? Verhamme et al. (2018) Eq. 2 and the midpoint of the peaks |
| `rt_` | `fblue_<kind>_vs_dv`, `br_vs_sep`, `br_vs_dvred`, `sep_vs_z`, `fwhm_vs_ha_fwhm`, `dv_vs_ha_fwhm` | radiative transfer. Low N_HI gives small Δv and blue flux (Verhamme et al. 2015, CLASSY), separation as the LyC indicator (Izotov et al. 2018), RT broadening beyond the nebular width (Erb et al. 2014) |
| `igm_` | `dv_vs_z`, `fblue_<kind>_vs_z`, `double_frac_vs_z` (at ΔBIC > 2, 6 and 10, Wilson 68% intervals, a lower limit), `fwhm_vs_z` (with (1+z)^(2/3) and a fitted power-law index) | the IGM and redshift (Hayes et al. 2021, zELDA, Prieto-Lyon et al. 2025) |
| `gal_` | `dv_vs_muv` (Mason+18, with a fit of log Δv), `fwhm_vs_muv` (Mason+19, Prieto-Lyon+25), `fblue_int_vs_muv`, `fwhm_models[_split]` (as before), and dv, fwhm and fblue_int against M*, SFR100, sSFR100, A_V and ξ_ion | halo mass, feedback, dust and the ionising spectrum |
| `chk_` | `dv_by_zline`, `dv_by_grating`, `dv_single_vs_double`, `dv_vs_snr`, `fwhm_vs_snr`, `fblue_int_vs_data`, and in `det_vs_nondet` `flux_vs_z`, `snr_vs_z` and `det_vs_nondet_cdfs` | systematics and selection |

**f_esc figures** (new, folder `plots/lya_science/<rule>_fesc/`, e.g. `plots/lya_science/snr5_fesc/`). Detections and 5σ upper limits together, `in_analysis == 1`, colour and marker by Hα case, limits as open triangles. Each figure is made for three subsets of the Hα cases and two dust versions, and the names end `_<subset>_<dust>`.

| Name part | Meaning |
|---|---|
| `case1` | spectral Hα, Balmer A_Hα |
| `case12` | plus case 2 (spectral Hα, SED A_Hα) |
| `case123` | plus case 3 (photometric Hα, SED A_Hα) |
| `dustbest` | `fesc_lya`, Balmer A_Hα for case 1, SED otherwise |
| `dustsed` | `fesc_lya_sed`, SED A_Hα for every source |

- `fesc_vs_<p>_<subset>_<dust>.png` for p = muv, beta, logM, logsfr10, logssfr10, av, logxion, z, dv, fwhm, fblue_int, and aha_used as a check. Δv, FWHM and f_blue show detections only.
- `fesc_overview_<subset>_<dust>.png`, all of them on one page.
- `fesc_summary.png` and `fesc_summary.csv`, the median of the detections and the reverse Kaplan-Meier median of all per subset and dust version.
- `fesc_by_case.png`, f_esc of each case, and Balmer against SED f_esc for case 1.
- Statistics are Spearman ρ of the detections and a censored Kendall τ_c of detections and limits (Isobe, Feigelson & Nelson 1986 pair rule, p from `--n-perm` shuffles). τ_c is diluted by pairs whose order is unknown, so compare its p rather than its size. Panels marked "shares Hα" (SFR10, sSFR10, ξ_ion, A_Hα used) have Hα on both axes. The script prints a digest of every correlation for every subset at the end.


Splits, as suffixes. `zbins` (three z bins at `--z-edges` 3.5 4.5), `zbins2` (at `--z-split` 3.5), `muvbins`, `massbins`, `ssfrbins` and `avbins` (two bins at the set's median), and `model` (single against double). Only the splits that make sense for a figure are made, and `--bins` limits them. Each scatter panel shows Spearman ρ and p. `stats.csv` has one row per panel with N, Spearman ρ and p, the 16th to 84th percentile of ρ over `--n-stat-mc` (1000) redraws within the errors, Kendall τ and p, the partial ρ controlling for z_sys (where neither axis or split is z), the fit slope and intercept with bootstrap errors where a fit is drawn, and for the histogram, fraction and comparison figures their medians, fractions and KS, Kruskal-Wallis or Anderson-Darling p (SciPy caps the AD p between 0.001 and 0.25). The terminal lists every unbinned panel with p < 0.05. Fits allow for errors on both axes and leave out unresolved lines, which enter the correlations at their FWHM limit.

f_blue figures are made for each of `--fblue-kinds` (int and data). The data version falls with Δv from the LSF alone, so use int for the physics and data as a check. `--fwhm-kind obs` uses the observed widths (names gain `obs_`), the closer comparison for Verhamme+18. About two minutes per set, mostly writing the figures.
```bash
python plot_lya_science.py                                       # detections
python plot_lya_science.py --sets detections non_detections
python plot_lya_science.py --fesc-only                           # only the f_esc figures
python plot_lya_science.py --no-fesc
python plot_lya_science.py --bins zbins2 model --no-pdf          # fewer figures, quicker
python plot_lya_science.py --fwhm-kind obs
python plot_lya_science.py --muv-source fit                      # interim M_UV, muvfit_ names
python plot_lya_science.py --catalog /ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_catalog_p95.csv
```

**26b. Check the Lyα widths** `checks/check_lya_fwhm.py`
Tests whether the skewed-Gaussian FWHM (from `lya_fits_mc.csv`, whatever `lya_model` says) is a fair measure of the lines, for the detections with `in_analysis == 1` (`--sample non_detections` or `all` for the others). Four panels in `plots/lya_fwhm_checks_<tag>.png`. (a) The FWHM measured straight from each aperture spectrum (1-pixel Gaussian smoothing, half-maximum crossings) against the convolved model measured on the same pixels with the same code. The direct numbers are larger than `fwhm_obs_kms` because of the smoothing and sampling, so compare them only with the model column next to them. (b) FWHM_int against Σ_Lyα, for noise bias. (c) FWHM_int against the continuum residual in the fit window (pixels where the model is below 5% of its peak), as S/N. (d) FWHM_int against the fraction of MC refits with α at either bound (0 or 15), which is near 0.5 when the skewness is unconstrained. Writes `MUSE_catalogs/lya_fwhm_checks_<tag>.csv`.
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
| a Lyα fit setting | 13, 14, 14b, 14c, 17b, 17c, 26, 28 |
| a double-peak setting (14b) | 14b, 14c, 17b, 17c, 17d for the double sources, 26 |
| the f_blue window (14c) | 14c, 17b, 26 |
| a by-eye label | edit `lya_manual_labels.csv`, then 21, 28 |
| photometry catalogue | 22, 23, 17b, 26, then 24, 27, 28 |
| Hα method, or the Lyα fits or detection rule | 24, 24b, 17b, then 24c, 26, 27, 28 |
| SED or fesc outputs | 28 |
| the close-pair list (step 1) or `CLOSE_PAIR_KEEP` | 17b, 17c, 17d `--pdf-only`, then anything that reads `lya_catalog.csv` |
| the detection rule (17b options) | 17b, 17c, 17d `--pdf-only`, 26, 26b |
| an MC error (14*) | 17b, then as for the detection rule |
| M_UV or β (22) | 23, 23b (comparison plot), 17b, 26 |
| Ken's SED fits (23b) | 23b, 17b, 26 |

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

- `p2_common.py` holds the z_sys rule, the shared constants and the fallback grating used for z_dja, and `preferred_lya_model`, the one rule for which Lyα fit is preferred (used by 14c and 17b).
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
- `jwst_catalogs/ha_hb_flux_corrections*.csv` (all versions) and `jwst_catalogs/ha_hb_diagnostics.csv` are replaced by `ha_flux_corrections.csv` (step 24).
- `jwst_catalogs/audit_missing_OIII.csv` and `duplicate_sources.csv` are one-off check outputs.
- The extra columns other scripts wrote into `grating_sources_by_JELS_ID.csv` (peak_snr, over_98, over_99.5, OIII_filter, z_OIII_*) vanish the next time step 3 runs. Nothing reads them now.

`lya_properties.csv` is still needed as the input to step 14.

## Still to do

**Small fixes**

- Port `--force-double`, `--cont-side` and `--mask` into `lime_OIII_singlereview.py`.
- AO gap flag and the edge false-position rejection.

**Hα corrections and f_esc** (steps 24 and 24b, done)

- Ask Ken how `A_Ha` is defined in his fits (whether nebular lines get extra attenuation). The decrement gives a median 0.29 mag more.
- The Lyα flux is a 0.6″ aperture flux and Hα is total, as in Paper 1, so f_esc measures the central Lyα.
- Stacked-flux f_esc, once the stacks are rewritten.

**Stacking** (`stack_lya_systematic.py`)

- Fix the per-pixel clipping (clip whole sources instead) and measure B/R against a local baseline, once the choices on the decision list are agreed.

**SED inputs** (`build_sed_catalog.py`, step 23b, done)

- Confirm with Ken that the fits used total fluxes. If they used aperture fluxes, M* and SFR need scaling to total (M_UV too).
- Bin the science plots by M*, SFR100, sSFR100 and A_V (to be planned).