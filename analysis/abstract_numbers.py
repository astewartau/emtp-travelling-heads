"""Compute every number the abstract quotes and write results/:
  pair_fits.csv.gz        one row per pipeline and scan pair (emtp.pair_fits)
  per_pipeline.csv        per-pipeline medians of |a-1|, R^2 and LoA by comparison and protocol
  spread_<region>.csv     sources of variability (SD, ppb) for Figure 4
  numbers.json            the values quoted in the text
Run from the repo root: python analysis/abstract_numbers.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import emtp

OUT = Path(__file__).resolve().parent.parent / 'results'
OUT.mkdir(exist_ok=True)
SPREAD_REGIONS = ['Caudate', 'Pallidum']
reg = emtp.regions()
pipes = emtp.pipelines()
fits = emtp.pair_fits(reg)
fits.to_csv(OUT / 'pair_fits.csv.gz', index=False, float_format='%.6g', compression={'method': 'gzip', 'mtime': 0})

ok = fits[fits.excluded == '']
pp = pd.concat({m: emtp.per_pipeline(fits, m) for m in ('abs_slope_dev', 'r2', 'ba_loa_ppm')}, axis=1)
# inter-scanner pooled over protocols: the value QSM-CI shows in its pipeline matrix (Figure 5)
pp[('abs_slope_dev', 'inter_scanner', 'all')] = ok[ok.comparison == 'inter_scanner'].groupby('pipeline').abs_slope_dev.median()
pp.columns = ['__'.join(c) for c in pp.columns]
pp.sort_index().to_csv(OUT / 'per_pipeline.csv', float_format='%.6g')


def col(metric, comp, prot):
    return pp[f'{metric}__{comp}__{prot}'].dropna()


def pct(v):
    return round(100 * float(v), 1)


n = {
    'dataset': {
        'acquisitions': int(reg.acquisition.nunique()),
        'subjects': int(emtp.acquisitions().subject.nunique()),
        'scanners_analysed': int(emtp.acquisitions().scanner.nunique()),
    },
    'pipelines': {
        'total': int(len(pipes)),
        'maps': int(reg[['pipeline', 'acquisition']].drop_duplicates().shape[0]),
        'three_step': int((pipes.n_steps == 3).sum()),
        'combined_or_end_to_end': int((pipes.n_steps < 3).sum()),
        'field_mapping_methods': int(pipes[pipes.n_steps == 3].field_mapping.nunique()),
        'background_removal_methods': int(pipes[pipes.n_steps == 3].background_removal.nunique()),
        'dipole_inversion_methods': int(pipes[pipes.n_steps == 3].dipole_inversion.nunique()),
    },
    'pairs': {
        'fitted': int(len(ok)),
        'excluded_slope_cap': int((fits.excluded == 'slope_cap').sum()),
        'regions_per_fit_min': int(ok.n_rois.min()),
        'regions_per_fit_max': int(ok.n_rois.max()),
        'share_of_fits_with_all_25_regions_pct': pct((ok.n_rois == 25).mean()),
    },
    'agreement': {},
    'fig5_range_pct': None,
    'spread_ppb': {},
}
for comp in ('test_retest', 'inter_scanner'):
    for prot in ('bridge', 'local'):
        d = col('abs_slope_dev', comp, prot)
        n['agreement'][f'{comp}_{prot}'] = {
            'median_abs_slope_dev_pct': pct(d.median()),
            'p25_pct': pct(d.quantile(.25)), 'p75_pct': pct(d.quantile(.75)), 'p95_pct': pct(d.quantile(.95)),
            'pipelines_above_25pct': int((d > .25).sum()),
            'median_r2': round(float(col('r2', comp, prot).median()), 3),
            'median_loa_ppb': round(1000 * float(col('ba_loa_ppm', comp, prot).median()), 1),
            'n_pipelines': int(len(d)),
        }
f5 = col('abs_slope_dev', 'inter_scanner', 'all').reindex([f'romeo-qsmrs+{b}+{d}' for d in emtp.FIG5_DIPOLE for b in emtp.FIG5_BFR])
n['fig5_range_pct'] = [pct(f5.min()), pct(f5.max())]
for region in SPREAD_REGIONS:
    t = emtp.spread(reg, region)
    t.round(2).to_csv(OUT / f'spread_{region.lower()}.csv')
    n['spread_ppb'][region] = {k: {p: (None if np.isnan(v) else round(float(v), 1)) for p, v in row.items()}
                              for k, row in t.iterrows()}

(OUT / 'numbers.json').write_text(json.dumps(n, indent=2) + '\n')
print(json.dumps(n, indent=2))
