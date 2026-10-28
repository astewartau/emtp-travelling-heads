# From reconstructions to regional values

`data/regional_chi.csv.gz` was extracted on QSM-CI's harmonization track (QSM-CI repository, `scripts/repro_eval.py`, identical at QSM-CI `d07cb23`), run on Bunya for the QSMxT v10.0.0 campaign (October 2026):

1. `repro_eval.py register` rigidly registers every acquisition's first-echo magnitude to a per-subject reference acquisition (SimpleITK).
2. `repro_eval.py seg` runs SynthSeg on the reference magnitude, giving one segmentation per subject.
3. `repro_eval.py stats` resamples each reconstruction into the subject frame and, per SynthSeg label, records voxel count, mean, median and SD of χ minus the whole-brain mean of that reconstruction. A label is kept only if at least 90% of its voxels lie inside the reconstruction (`MIN_ROI_COVERAGE`).
4. `scripts/repro_slurm/harvest.sh` collects the per-shard rows into `results/repro_rois.json`.

`regional_chi.csv.gz` is a flat export of that file (rows whose acquisition starts with `s00`, i.e. the traveling heads, not the pilot cohort). `analysis/emtp.py` then reproduces `repro_eval.py fits` (inter-scanner pairs over every run combination, as in QSM-CI from branch `repro/inter-scanner-all-runs`): same pairs, same orthogonal fit and the same exclusions. Its per-pipeline medians match QSM-CI to its 4-decimal rounding.

One difference: here the Bland–Altman limits of agreement use the sample SD (ddof = 1); QSM-CI uses ddof = 0, so QSM-CI shows them about 2.5% narrower for a 20-region fit.

## Rerunning the extraction
The reconstructions (45,927 maps) are on Hugging Face (`qsmxt/qsm-ci-volumes`) and the inputs on OSF. Rerunning steps 1–4 needs the QSM-CI checkout and its environment; on Bunya the run tree is `/scratch/user/uqaste15/qsmci-repro`. The next QSM-CI run will regenerate this file; replace it and run `python analysis/abstract_numbers.py`.
