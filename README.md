# EMTP traveling-brain QSM

Data, analysis and figures for the EMTP traveling-brain QSM calibration study: 6 subjects, 72 acquisitions (bridge and local protocols, 3 runs each) on 6 scanners from 3 vendors, reconstructed with 638 QSM pipelines on [QSM-CI](https://qsmxt.github.io/QSM-CI/).

This is the analysis snapshot used for the ISMRM 2027 abstract.

**Analysis notebook (with outputs): [analysis.ipynb](https://github.com/astewartau/emtp-travelling-heads/blob/main/analysis.ipynb)**

Subject IDs are pseudonymised (`s002001`–`s006001`).

## Layout
| Path | Contents |
|---|---|
| `data/acquisitions.csv` | Acquisition → subject, scanner, protocol, run |
| `data/pipelines.csv` | Pipeline → steps and methods, QSM Challenge 2.0 scores |
| `data/regional_chi.csv.gz` | Per pipeline, acquisition and SynthSeg label: voxel count and mean/median/SD χ (ppm, relative to the whole-brain mean). How it was extracted: `extraction/README.md` |
| `data/protocols.csv` | Acquisition parameters per acquisition, from the raw sidecars (`scripts/extract_protocols.py`) |
| `data/head_pose.csv` | Head pose per acquisition from rigid registration: motion within the session and repositioning between sessions (`scripts/head_pose.py`) |
| `data/b0_drift.csv` | Field drift between runs of a protocol within a session, from the multi-echo phase (`scripts/b0_drift.py`) |
| `data/image_quality.csv` | Ghosting outside the head and edge sharpness per acquisition (`scripts/image_quality.py`) |
| `data/echo_shift.csv` | Drift during each run, estimated from the echo-to-echo image shift (`scripts/echo_shift.py`) |
| `analysis/followup.py` | Follow-up analyses for the open issues; writes `results/followup/` |
| `analysis/emtp.py` | Pair definitions, orthogonal fits, per-pipeline summaries, sources of variability |
| `analysis/abstract_numbers.py` | Writes `results/`: the headline values (`numbers.json`), pair fits, per-pipeline medians, spread tables |
| `docs/decisions.md` | Analysis decisions since the abstract, each with the alternative not taken |
| `analysis.ipynb` | Walk-through of the analysis with outputs: dataset, regions per fit, agreement, sources of variability, key results and Figures 1–5 (rebuild: `python analysis/build_notebook.py`) |
| `figures/` | Figures 1–5; sources in `figures/src/` |

## Reproduce
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python analysis/abstract_numbers.py
cd figures/src
../../.venv/bin/python fig3_agreement.py
../../.venv/bin/python fig4_sources_of_variability.py
./render_fig1.sh                           # Figure 1 (headless Firefox)
../../scripts/fetch_figure2_data.sh && ../../.venv/bin/python fig_images.py   # Figure 2 (inputs not in git)
../../.venv/bin/python fig_qsmci.py        # Figure 5 from data/qsmci_screenshot; shots4.js recaptures it (Node + Playwright)
```

## Data version
QSM-CI harmonization run with QSMxT v10.0.0 (October 2026). Five sessions acquired in October 2026 at two further sites are not yet included.
