# From reconstructions to regional values

`data/regional_chi.csv.gz` is extracted from QSM-CI's harmonization track (QSM-CI repository, `scripts/repro_eval.py`), run on Bunya for each campaign:

1. `repro_eval.py register` rigidly registers every acquisition's first-echo magnitude to a per-subject reference acquisition (SimpleITK).
2. `repro_eval.py seg` runs SynthSeg on the reference magnitude, giving one segmentation per subject.
3. `repro_eval.py stats` resamples each reconstruction into the subject frame and, per SynthSeg label, records voxel count, mean, median and SD of χ minus the whole-brain mean of that reconstruction. A label is kept only if at least 90% of its voxels lie inside the reconstruction (`MIN_ROI_COVERAGE`).
4. `scripts/repro_slurm/harvest.sh` collects the per-shard rows into `results/repro_rois.json`.

`regional_chi.csv.gz` is a flat export of that file (rows whose acquisition starts with `s00`, i.e. the traveling heads, not the pilot cohort). `analysis/emtp.py` then reproduces `repro_eval.py fits` (inter-scanner pairs over every run combination): same pairs, same orthogonal fit and the same exclusions.

One difference: here the Bland–Altman limits of agreement use the sample SD (ddof = 1); QSM-CI uses ddof = 0, so QSM-CI shows them about 2.5% narrower for a 20-region fit.

## From a QSM-CI run to `data/`
`scripts/extract_qsmci.py` writes `data/regional_chi.csv.gz`, `data/acquisitions.csv` and `data/pipelines.csv` from:

- `results/repro_rois.json` of the QSM-CI run tree (the harvest; large, so not committed to QSM-CI);
- the QSM-CI repository at the commit with that run's published payload: `results/repro.json` (pipelines, acquisitions and pairs the site shows) and `results/index.json` (simulation scores);
- a scanner-id map from QSM-CI's scanner ids to the `site<N>-<model>` ids used here (kept outside this repository).

Only the published pipelines and acquisitions are written. `acquisitions.csv` gets each subject's scanner order from QSM-CI's inter-scanner pairs (`scanner_order`, which fixes which scan is x), and the script checks that `emtp.comparison_plan` gives exactly QSM-CI's pairs. `pipelines.csv` takes each pipeline's full-pipeline simulation XSIM and DGM linearity (QSM Challenge 2.0 phantom) from the QSM-CI run whose name is the pipeline's; pipelines without one (variants such as `.parity`, the GPU dipole inversions, the two-step pipelines) have none, and `emtp.good_pipelines` leaves them out. `default_settings` marks pipelines without a variant step.

Maps with any region more than `emtp.DIVERGED_MAP_PPM` (10 ppm) from the brain mean diverged numerically and are left out of every analysis (`emtp.regions`); QSM-CI keeps them and excludes most of their pairs by the slope cap. `scripts/check_qsmci_fits.py` compares the per-pipeline medians with QSM-CI's: all agree to QSM-CI's rounding except pipelines with diverged maps.

```
python scripts/extract_qsmci.py --rois repro_rois.json --qsmci <QSM-CI checkout> --ref origin/main --scanner-ids <map.csv>
python analysis/abstract_numbers.py
python scripts/check_qsmci_fits.py --qsmci <QSM-CI checkout> --ref origin/main
cd analysis && python followup.py
```
