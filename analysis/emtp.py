"""Regional susceptibility values -> scan-pair agreement and sources of variability.

Everything quoted in the abstract is computed here from three files in data/:
  acquisitions.csv        acquisition -> subject, scanner, protocol, run
  pipelines.csv           pipeline -> n_steps, method per step, QSM Challenge 2.0 scores
  regional_chi.csv.gz     per pipeline, acquisition and SynthSeg label: voxel count and mean chi (ppm),
                          referenced to the whole-brain mean of that reconstruction

Region values were extracted by QSM-CI (scripts/repro_eval.py, `stats`); the pair definitions and the
fit below reproduce that script's `fits`, so pair_fits() matches the slopes published on QSM-CI.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'

# QSM-CI exclusion rules (repro_eval.py)
MIN_PAIR_ROIS = 8      # fewer regions present in both scans: no fit
SLOPE_DEV_CAP = 2.0    # |a - 1| above this is a failed reconstruction, not a reproducibility value

COMPARISONS = ['test_retest', 'inter_scanner', 'inter_protocol']
VENDOR = {'site1-achieva': 'Philips', 'site1-elition': 'Philips', 'site4-skyra': 'Siemens',
          'site5-mr750': 'GE', 'site5-prisma': 'Siemens', 'site6-ingenia': 'Philips'}
# Figure 5: the QSM-CI matrix subset shown (ROMEO field mapping; dipole inversion x background removal)
FIG5_DIPOLE = ['whqsm-qsmrs', 'hdqsm-qsmrs', 'medi-qsmrs', 'ilsqr-sti', 'rts-qsmrs', 'tkd-qsmrs', 'lpcnn', 'qsmnet-plus']
FIG5_BFR = ['sharp-qsmrs', 'resharp-qsmrs', 'pdf-qsmrs', 'vsharp-sti', 'lbv-qsmrs']


def acquisitions():
    return pd.read_csv(DATA / 'acquisitions.csv')


def pipelines():
    return pd.read_csv(DATA / 'pipelines.csv')


def regions():
    """Long table: pipeline, acquisition, label, region, n_voxels, mean_ppm, ..."""
    return pd.read_csv(DATA / 'regional_chi.csv.gz')


def bilateral(reg, region):
    """Mean chi (ppb) of a bilateral region per pipeline and acquisition: the voxel-weighted mean of the
    hemispheres present (a hemisphere is absent when under 90% of it lies inside the reconstruction)."""
    r = reg[reg.region.isin([f'Left-{region}', f'Right-{region}'])]
    w = r.assign(s=r.n_voxels * r.mean_ppm).groupby(['pipeline', 'acquisition'])[['s', 'n_voxels']].sum()
    out = (1000 * w.s / w.n_voxels).rename(region).reset_index()
    return out.merge(acquisitions(), on='acquisition')


def comparison_plan(acq):
    """(comparison, acquisition_x, acquisition_y) for every pair the design calls for, as in QSM-CI:
    pairs never cross subjects; the reference-site (site 1) scan is x, and bridge is x for inter_protocol.
      test_retest     same scanner and protocol, every two runs
      inter_scanner   same protocol, every run on one scanner against every run on the other (the
                      sessions are on different days, so run numbers do not correspond)
      inter_protocol  same scanner, bridge run k against local run k (acquired one after the other)"""
    by = {(a.subject, a.scanner, a.protocol, a.run): a.acquisition for a in acq.itertuples()}
    plan = []
    for subj, mine in acq.groupby('subject', sort=True):
        scanners = sorted(dict.fromkeys(mine.scanner), key=lambda s: not s.startswith('site1-'))
        protocols, runs = sorted(set(mine.protocol)), sorted(set(mine.run))
        for sc in scanners:
            for pr in protocols:
                for i, ra in enumerate(runs):
                    for rb in runs[i + 1:]:
                        plan.append(('test_retest', by.get((subj, sc, pr, ra)), by.get((subj, sc, pr, rb))))
        for pr in protocols:
            for i, sa in enumerate(scanners):
                for sb in scanners[i + 1:]:
                    for ra in runs:
                        for rb in runs:
                            plan.append(('inter_scanner', by.get((subj, sa, pr, ra)), by.get((subj, sb, pr, rb))))
        for sc in scanners:
            for run in runs:
                plan.append(('inter_protocol', by.get((subj, sc, 'bridge', run)), by.get((subj, sc, 'local', run))))
    return [p for p in plan if p[1] and p[2]]


def fit(x, y):
    """Orthogonal (total least squares) regression y = a x + b, first principal component of the
    centred cloud. Also returns R^2 and Bland-Altman bias and 95% limits of agreement (y - x), and the
    limits of agreement after the fitted scaling, y - (a x + b)."""
    xm, ym = x.mean(), y.mean()
    _, _, vt = np.linalg.svd(np.stack([x - xm, y - ym]).T, full_matrices=False)
    a = vt[0, 1] / vt[0, 0]
    b = ym - a * xm
    d = y - x
    return dict(slope=a, intercept_ppm=b, r2=np.corrcoef(x, y)[0, 1] ** 2,
                ba_bias_ppm=d.mean(), ba_loa_ppm=1.96 * d.std(ddof=1),
                ba_loa_corrected_ppm=1.96 * (y - a * x - b).std(ddof=1), n_rois=len(x))


def pair_fits(reg=None):
    """One row per pipeline and scan pair, with the fit over the regions present in both scans.
    Pairs failing the QSM-CI rules are kept with `excluded` set (too_few_rois, flat, slope_cap)."""
    reg = regions() if reg is None else reg
    acq = acquisitions().set_index('acquisition')
    plan = comparison_plan(acq.reset_index())
    rows = []
    for pipe, g in reg.groupby('pipeline', sort=True):
        vals = {a: dict(zip(h.label, h.mean_ppm)) for a, h in g.groupby('acquisition')}
        for comp, ax, ay in plan:
            if ax not in vals or ay not in vals:
                continue
            common = sorted(set(vals[ax]) & set(vals[ay]))
            row = dict(pipeline=pipe, comparison=comp, acquisition_x=ax, acquisition_y=ay, n_rois=len(common))
            x = np.array([vals[ax][k] for k in common]); y = np.array([vals[ay][k] for k in common])
            if len(common) < MIN_PAIR_ROIS:
                row['excluded'] = 'too_few_rois'
            elif x.std() < 1e-6 or y.std() < 1e-6:
                row['excluded'] = 'flat'
            else:
                row.update(fit(x, y))
                if abs(row['slope'] - 1) > SLOPE_DEV_CAP:
                    row['excluded'] = 'slope_cap'
            rows.append(row)
    df = pd.DataFrame(rows)
    df['excluded'] = df.get('excluded').fillna('')
    df['abs_slope_dev'] = (df.slope - 1).abs()
    for c in ('subject', 'scanner', 'protocol', 'run'):
        df[f'{c}_x'] = df.acquisition_x.map(acq[c]); df[f'{c}_y'] = df.acquisition_y.map(acq[c])
    # bridge / local for same-protocol pairs; 'local_vs_bridge' for inter_protocol
    df['protocol'] = np.where(df.protocol_x == df.protocol_y, df.protocol_x, 'local_vs_bridge')
    return df


def per_pipeline(fits, metric='abs_slope_dev'):
    """Median of `metric` over each pipeline's included pairs, by comparison and protocol:
    columns e.g. ('inter_scanner', 'bridge'), one row per pipeline."""
    f = fits[fits.excluded == '']
    return f.groupby(['pipeline', 'comparison', 'protocol'])[metric].median().unstack(['comparison', 'protocol'])


def spread(reg, region):
    """SD (ppb) of a region's mean chi when one factor varies and the others are fixed, by protocol.
    Each value is a median over pipelines (and over subjects/scanners within a pipeline)."""
    r = bilateral(reg, region)
    keys = ['pipeline', 'subject', 'scanner', 'protocol']
    cond = r.groupby(keys)[region].mean().reset_index()  # mean over runs
    out = {}
    for lab, prots in [('bridge', ['bridge']), ('local', ['local']), ('both', ['bridge', 'local'])]:
        c, x = cond[cond.protocol.isin(prots)], r[r.protocol.isin(prots)]
        sc = c.groupby(['pipeline', 'subject', 'protocol'])[region].agg(['std', 'count'])
        out[lab] = {
            'rescan': x.groupby(keys)[region].std().groupby('pipeline').median().median(),
            'protocol': cond.groupby(['pipeline', 'subject', 'scanner'])[region].std().groupby('pipeline')
            .median().median() if lab == 'both' else np.nan,
            'scanner': sc[sc['count'] > 1]['std'].groupby('pipeline').median().median(),
            'pipeline': c.groupby(['subject', 'scanner', 'protocol'])[region].std().median(),
            # between subjects: the six subjects all scanned at the reference site (site 1, Philips Achieva)
            'between_subjects': c[c.scanner == 'site1-achieva'].groupby(['pipeline', 'protocol'])[region].std()
            .groupby('pipeline').median().median(),
        }
    return pd.DataFrame(out)
