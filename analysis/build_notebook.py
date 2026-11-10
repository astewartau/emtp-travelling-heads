"""Build and execute analysis.ipynb (python analysis/build_notebook.py)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md('# EMTP traveling-brain QSM: analysis\n\nResults recomputed from the regional values in `data/`, with the figures. '
       'The same functions back `analysis/abstract_numbers.py` and the figure scripts in `figures/src/`.'),
    code("import sys; sys.path.insert(0, 'analysis')\nimport emtp, numpy as np, pandas as pd\n"
         "pd.set_option('display.precision', 3)\nreg = emtp.regions(); acq = emtp.acquisitions(); pipes = emtp.pipelines()\n"
         "print(reg.pipeline.nunique(), 'pipelines,', reg[['pipeline','acquisition']].drop_duplicates().shape[0], 'maps,', acq.shape[0], 'acquisitions')"),
    md('## Dataset'),
    code("acq.groupby(['subject', 'scanner']).size().unstack(fill_value=0)"),
    code("pipes.groupby('n_steps').size().rename({1: 'end-to-end', 2: 'field map + combined', 3: 'three-step'})"),
    md('## Regions per fit\n\nA region enters a scan’s values only if at least 90% of it lies inside that reconstruction; '
       'a pair is fitted on the regions present in both scans.'),
    code("fits = emtp.pair_fits(reg)\nok = fits[fits.excluded == '']\nprint(fits.excluded.value_counts())\n"
         "ok.n_rois.value_counts().sort_index()"),
    code("# which regions drop out, by scanner and protocol (share of maps missing the region)\n"
         "pres = reg.groupby(['acquisition'])['region'].value_counts().unstack(fill_value=0)\n"
         "maps = reg.groupby('acquisition').pipeline.nunique()\n"
         "miss = 1 - pres.div(maps, axis=0)\n"
         "miss = miss.join(acq.set_index('acquisition')[['scanner', 'protocol']]).groupby(['scanner', 'protocol']).mean()\n"
         "miss.loc[:, (miss > 0.05).any()].round(2)"),
    md('## Agreement between scans (Figure 3)\n\nPer pipeline: median over its scan pairs. Then the distribution across pipelines. '
       'The primary measure is |a − 1| for both test-retest and inter-scanner pairs; secondary measures are R², the limits '
       'of agreement after correcting each pair for its fitted scaling (corrected LoA), and for test-retest the SD of a '
       '(see `docs/decisions.md`).'),
    code("pp = {m: emtp.per_pipeline(fits, m) for m in ('abs_slope_dev', 'r2', 'ba_loa_ppm', 'ba_loa_corrected_ppm')}\n"
         "d = 100 * pp['abs_slope_dev']\n"
         "pd.DataFrame({'p25': d.quantile(.25), 'median': d.median(), 'p75': d.quantile(.75), 'p95': d.quantile(.95),\n"
         "              'pipelines > 25%': (d > 25).sum()}).loc[['test_retest', 'inter_scanner']].round(1)"),
    code("pd.DataFrame({'median R²': pp['r2'].median(), 'median LoA (± ppb)': 1000 * pp['ba_loa_ppm'].median(),\n"
         "              'median corrected LoA (± ppb)': 1000 * pp['ba_loa_corrected_ppm'].median()})"
         ".loc[['test_retest', 'inter_scanner', 'inter_protocol']].round(3)"),
    md('### By scanner pair (inter-scanner, all pipelines)'),
    code("isc = ok[ok.comparison == 'inter_scanner']\n"
         "(100 * isc.groupby(['scanner_x', 'scanner_y', 'protocol']).abs_slope_dev.median()).unstack().round(1)"),
    md('### By pipeline step (inter-scanner median |a − 1|, %)'),
    code("isc3 = isc.merge(pipes, on='pipeline')\n"
         "for step in ('field_mapping', 'background_removal'):\n"
         "    display((100 * isc3.groupby([step, 'protocol']).abs_slope_dev.median()).unstack().round(1))\n"
         "(100 * isc3.groupby(['dipole_inversion', 'protocol']).abs_slope_dev.median()).unstack().sort_values('bridge').round(1)"),
    md('## Sources of variability (Figure 4)\n\nEach bar is the SD of a region\'s mean χ when one factor varies and the others '
       'are fixed, median over pipelines (and over subjects and scanners within a pipeline). Rescan: SD over the three runs. '
       'Protocol: for each subject and scanner, the SD of the run-averaged values of the bridge and local protocols (two '
       'values), so it includes any scaling difference between the protocols. Scanner: SD over the scanners a subject was '
       'scanned on, same protocol. Pipeline: SD over all pipelines of the run-averaged values, same subject, scanner and protocol. Between subjects: SD over the subjects '
       'scanned on the reference scanner. Bars are not corrected for scaling (see `docs/decisions.md`).'),
    code("pd.concat({r: emtp.spread(reg, r) for r in ('Caudate', 'Pallidum', 'Putamen', 'Thalamus')}, axis=1).round(1)"),
    md('## Key results\n\nThe values quoted in the abstract, from `results/numbers.json` (written by `analysis/abstract_numbers.py`).'),
    code("import json\nn = json.load(open('results/numbers.json'))\n"
         "print(f\"{n['pipelines']['total']} pipelines ({n['pipelines']['three_step']} three-step: \"\n"
         "      f\"{n['pipelines']['field_mapping_methods']} field mapping x {n['pipelines']['background_removal_methods']} background removal x \"\n"
         "      f\"{n['pipelines']['dipole_inversion_methods']} dipole inversion; {n['pipelines']['combined_or_end_to_end']} combined or end-to-end), \"\n"
         "      f\"{n['pipelines']['maps']:,} maps, {n['pairs']['fitted']:,} scan pairs fitted\")\n"
         "pd.DataFrame(n['agreement']).T.rename(columns={'median_abs_slope_dev_pct': 'median |a-1| %', 'p25_pct': 'p25 %', 'p75_pct': 'p75 %',\n"
         "    'p95_pct': 'p95 %', 'pipelines_above_25pct': 'pipelines > 25%', 'median_r2': 'median R²', 'median_loa_ppb': 'median LoA (± ppb)',\n"
         "    'median_loa_corrected_ppb': 'median corrected LoA (± ppb)', 'median_sd_slope_pct': 'SD of a %',\n"
         "    'median_abs_slope_dev_without_run1_pct': 'median |a-1| without run 1 %'})"),
    code("print('Figure 5 subset, inter-scanner |a-1| range (%):', n['fig5_range_pct'])\n"
         "pd.concat({r: pd.DataFrame(v).T for r, v in n['spread_ppb'].items()}, axis=1)  # SD (ppb), Figure 4"),
    md('## Figures\n\nFigures 3 and 4 are drawn here from the data by the same code that writes `figures/*.png` '
       '(`figures/src/`). Figure 1 is a diagram, Figure 2 needs the NIfTI inputs that are not in git, and Figure 5 is a '
       'screenshot of QSM-CI; those are shown as saved images, with Figure 5\'s values recomputed below.'),
    code("from IPython.display import Image\nsys.path.insert(0, 'figures/src')\n%matplotlib inline"),
    md('### Figure 1: study design'),
    code("Image('figures/fig1_study_design.png', width=900)"),
    md('### Figure 2: representative images'),
    code("Image('figures/fig2_representative_images.png', width=900)"),
    md('### Figure 3: agreement between scans'),
    code("import fig3_agreement\nfig3_agreement.make(reg, fits);"),
    md('### Figure 4: sources of variability'),
    code("import fig4_sources_of_variability\nfig4_sources_of_variability.make(reg);"),
    md('### Figure 5: pipeline dependence of inter-scanner agreement on QSM-CI'),
    code("Image('figures/fig5_qsmci.png', width=800)"),
    md('The same grid computed from the data: inter-scanner median |a − 1| (%) over all scan pairs, ROMEO field mapping.'),
    code("import matplotlib.pyplot as plt\n"
         "isc_all = 100 * ok[ok.comparison == 'inter_scanner'].groupby('pipeline').abs_slope_dev.median()\n"
         "grid = pd.DataFrame([[isc_all.get(f'romeo-qsmrs+{b}+{d}') for b in emtp.FIG5_BFR] for d in emtp.FIG5_DIPOLE],\n"
         "                    index=emtp.FIG5_DIPOLE, columns=emtp.FIG5_BFR)\n"
         "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
         "ax.imshow(grid.values, cmap='RdYlGn_r', vmin=3, vmax=12, aspect='auto')\n"
         "for (i, j), v in np.ndenumerate(grid.values):\n"
         "    ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9)\n"
         "ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=30, ha='right'); ax.set_yticks(range(len(grid)), grid.index)\n"
         "ax.set_title('Inter-scanner |a − 1| (%)', loc='left'); plt.show()"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
NotebookClient(nb, timeout=900, kernel_name='emtp', resources={'metadata': {'path': str(ROOT)}}).execute()
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, ROOT / 'analysis.ipynb')
print('wrote analysis.ipynb')
