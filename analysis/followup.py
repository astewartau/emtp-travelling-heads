"""Follow-up analyses for the open GitHub issues; writes results/followup/:
  followup.json               the summary values, one key per issue
  slope_spread.csv            #2  per pipeline: SD of a over test-retest pairs, IQR of |a-1| across scanner pairs
  inter_scanner_pairs.csv     #7/#8  per scanner pair and protocol: median |a-1| over pipelines
  calibration_loso.csv.gz     #5  per local-vs-bridge pair: |a-1| and LoA uncorrected, leave-one-subject-out, in-sample
  inter_scanner_run_averaged.csv.gz  #10  inter-scanner fits on run-averaged scans
  pipeline_vs_consensus.csv.gz       #6  each pipeline fitted against the median over pipelines, per acquisition
Each summary is a median over pipelines of a per-pipeline value, as in abstract_numbers.py.
Run from the repo root: python analysis/followup.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import emtp

OUT = Path(__file__).resolve().parent.parent / 'results' / 'followup'
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOLS = ['bridge', 'local']
SPREAD_REGIONS = ['Caudate', 'Pallidum']


def r(v, k=1):
    return None if pd.isna(v) else round(float(v), k)


def pipeline_summary(per_pipe, scale=1, n_key='n_pipelines'):
    """Median and IQR of a series (by default one value per pipeline)."""
    d = per_pipe.dropna() * scale
    return {'median': r(d.median(), 2), 'p25': r(d.quantile(.25), 2), 'p75': r(d.quantile(.75), 2), n_key: int(len(d))}


def slope_spread(ok):
    """#2: test-retest by the SD of a (no systematic scaling is expected within a scanner and protocol),
    alongside median |a-1|; inter-scanner by median |a-1| and its IQR across scanner pairs."""
    tr = ok[ok.comparison == 'test_retest']
    g = tr.groupby(['pipeline', 'protocol'])
    per = pd.DataFrame({'tr_sd_slope': g.slope.std(), 'tr_mean_slope_minus_1': g.slope.mean() - 1,
                        'tr_median_abs_slope_dev': g.abs_slope_dev.median()})
    isc = ok[ok.comparison == 'inter_scanner']
    by_pair = isc.groupby(['pipeline', 'protocol', 'scanner_pair']).abs_slope_dev.median()
    q = by_pair.groupby(['pipeline', 'protocol'])
    per['is_median_abs_slope_dev'] = isc.groupby(['pipeline', 'protocol']).abs_slope_dev.median()
    per['is_iqr_across_scanner_pairs'] = q.quantile(.75) - q.quantile(.25)
    per['is_n_scanner_pairs'] = q.size()
    per.to_csv(OUT / 'slope_spread.csv', float_format='%.6g')
    out = {}
    for prot in PROTOCOLS:
        p = per.xs(prot, level='protocol')
        out[prot] = {
            'test_retest_sd_slope_pct': pipeline_summary(p.tr_sd_slope, 100),
            'test_retest_mean_slope_minus_1_pct': pipeline_summary(p.tr_mean_slope_minus_1, 100),
            'test_retest_median_abs_slope_dev_pct': pipeline_summary(p.tr_median_abs_slope_dev, 100),
            'inter_scanner_median_abs_slope_dev_pct': pipeline_summary(p.is_median_abs_slope_dev, 100),
            'inter_scanner_iqr_across_scanner_pairs_pct': pipeline_summary(p.is_iqr_across_scanner_pairs, 100),
            'n_scanner_pairs': int(p.is_n_scanner_pairs.max()),
        }
    return out


def corrected_loa(ok):
    """#3: Bland-Altman limits of agreement before and after the fitted scaling y -> (y - b) / a."""
    out = {}
    for comp in emtp.COMPARISONS:
        for prot, d in ok[ok.comparison == comp].groupby('protocol'):
            g = d.groupby('pipeline')
            out[f'{comp}_{prot}'] = {'loa_ppb': pipeline_summary(g.ba_loa_ppm.median(), 1000),
                                     'loa_corrected_ppb': pipeline_summary(g.ba_loa_corrected_ppm.median(), 1000)}
    return out


def by_scanner_relation(ok):
    """#7/#8: inter-scanner |a-1| split by site (same institution or not) and vendor (same or not)."""
    isc = ok[ok.comparison == 'inter_scanner']
    tab = (isc.groupby(['scanner_pair', 'site_relation', 'vendor_pair', 'protocol', 'pipeline']).abs_slope_dev.median()
           .groupby(['scanner_pair', 'site_relation', 'vendor_pair', 'protocol']).median().mul(100).round(2)
           .unstack('protocol'))
    subj = isc.groupby('scanner_pair').subject_x.unique().map(lambda s: ', '.join(sorted(s)))
    tab = tab.join(subj.rename('subjects'), on='scanner_pair')
    tab.to_csv(OUT / 'inter_scanner_pairs.csv')
    out = {}
    for key in ('site_relation', 'vendor_relation'):
        out[key] = {}
        for (grp, prot), d in isc.groupby([key, 'protocol']):
            s = pipeline_summary(d.groupby('pipeline').abs_slope_dev.median(), 100)
            s.update(n_scanner_pairs=int(d.scanner_pair.nunique()), n_subjects=int(d.subject_x.nunique()))
            out[key].setdefault(grp, {})[prot] = s
    out['scanner_pairs_median_abs_slope_dev_pct'] = {
        k[0]: {'site_relation': k[1], 'vendors': k[2], **{p: r(row[p], 2) for p in PROTOCOLS}, 'subjects': row.subjects}
        for k, row in tab.iterrows()}
    return out


def protocol_spread_corrected(reg, ok):
    """#4: the Fig. 4 protocol bar before and after mapping local values to bridge space with one linear
    transform per pipeline and scanner (median a, b of that scanner's inter-protocol fits, bridge = x).
    The transform is estimated on the same scans it corrects, so this is the best case for calibration;
    the out-of-sample test is #5."""
    ip = ok[ok.comparison == 'inter_protocol']
    ab = ip.groupby(['pipeline', 'scanner_x'])[['slope', 'intercept_ppm']].median()
    ab.index.names = ['pipeline', 'scanner']
    acq = emtp.acquisitions()[['acquisition', 'scanner', 'protocol']]
    rc = reg.merge(acq, on='acquisition').join(ab, on=['pipeline', 'scanner'])
    loc = (rc.protocol == 'local') & rc.slope.notna()
    rc.loc[loc, 'mean_ppm'] = (rc.mean_ppm[loc] - rc.intercept_ppm[loc]) / rc.slope[loc]
    rc = rc[reg.columns]
    out = {'transform': {'median_slope': r(ab.slope.median(), 3), 'p25_slope': r(ab.slope.quantile(.25), 3),
                         'p75_slope': r(ab.slope.quantile(.75), 3)}}
    for region in SPREAD_REGIONS:
        before, after = emtp.spread(reg, region), emtp.spread(rc, region)
        out[region] = {'protocol_sd_ppb': r(before.loc['protocol', 'both']),
                       'protocol_sd_corrected_ppb': r(after.loc['protocol', 'both']),
                       'rescan_sd_ppb': {p: r(before.loc['rescan', p]) for p in PROTOCOLS}}
    return out


def region_values(reg):
    """{pipeline: {acquisition: {label: mean_ppm}}}"""
    return {p: {a: dict(zip(h.label, h.mean_ppm)) for a, h in g.groupby('acquisition')}
            for p, g in reg.groupby('pipeline', sort=True)}


def fit_dicts(vx, vy):
    """emtp.fit over the regions present in both, with the QSM-CI exclusion rules; None if excluded."""
    common = sorted(set(vx) & set(vy))
    if len(common) < emtp.MIN_PAIR_ROIS:
        return None
    x = np.array([vx[k] for k in common]); y = np.array([vy[k] for k in common])
    if x.std() < 1e-6 or y.std() < 1e-6:
        return None
    f = emtp.fit(x, y)
    return None if abs(f['slope'] - 1) > emtp.SLOPE_DEV_CAP else f


def calibration_loso(vals, ok):
    """#5: local -> bridge calibration on one scanner, the transform the framework uses. For each held-out
    subject, a and b are the medians of the other subjects' local-vs-bridge fits on that scanner (bridge = x,
    local = y); the held-out subject's local scans are mapped to bridge space, (y - b) / a, and refitted
    against the bridge scan of the same run. Compared with no correction and with the subject's own
    transform (in-sample), on scanners with at least two subjects."""
    ip = ok[ok.comparison == 'inter_protocol']
    per_subj = ip.groupby(['pipeline', 'scanner_x', 'subject_x'])[['slope', 'intercept_ppm']].median()
    n_subj = ip.groupby('scanner_x').subject_x.nunique()
    scanners = n_subj[n_subj >= 2].index
    rows = []
    for row in ip[ip.scanner_x.isin(scanners)].itertuples():
        own = per_subj.loc[(row.pipeline, row.scanner_x)]
        others = own.drop(index=row.subject_x)
        if others.empty:
            continue
        vx, vy = vals[row.pipeline][row.acquisition_x], vals[row.pipeline][row.acquisition_y]
        out = dict(pipeline=row.pipeline, scanner=row.scanner_x, subject=row.subject_x,
                   none_abs_slope_dev=row.abs_slope_dev, none_loa=row.ba_loa_ppm,
                   n_training_subjects=len(others))
        for lab, (a, b) in (('loso', others.median()), ('in_sample', own.loc[row.subject_x])):
            f = fit_dicts(vx, {k: (v - b) / a for k, v in vy.items()})
            if f:
                out[f'{lab}_abs_slope_dev'] = abs(f['slope'] - 1); out[f'{lab}_loa'] = f['ba_loa_ppm']
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'calibration_loso.csv.gz', index=False, float_format='%.6g',
              compression={'method': 'gzip', 'mtime': 0})
    # stability of the transform: SD over subjects of a, against the rescan SD of a (#2)
    sd_a = per_subj.slope.groupby(['pipeline', 'scanner_x']).std()
    out = {'scanners': {s: int(n_subj[s]) for s in scanners},
           'sd_of_local_to_bridge_slope_over_subjects_pct': {
               s: pipeline_summary(sd_a.xs(s, level='scanner_x'), 100) for s in scanners}}
    for s in list(scanners) + ['all']:
        d = df if s == 'all' else df[df.scanner == s]
        g = d.groupby('pipeline')
        out[s] = {f'{lab}_{m}': pipeline_summary(g[f'{lab}_{m}'].median(), 100 if m == 'abs_slope_dev' else 1000)
                  for lab in ('none', 'loso', 'in_sample') for m in ('abs_slope_dev', 'loa')}
        out[s]['share_of_pipelines_loso_better_than_none'] = r(
            (g.loso_abs_slope_dev.median() < g.none_abs_slope_dev.median()).mean(), 2)
    return out


def noise_vs_systematic(vals, ok):
    """#10: how much of the inter-scanner slope deviation is rescan noise. Two estimates per pipeline:
    (1) RMS of ln a over inter-scanner pairs minus that over test-retest pairs (both compare two single
        scans, so the noise term is the same): systematic = sqrt(max(0, ms_inter - ms_retest));
    (2) inter-scanner fits on run-averaged scans (three runs averaged per region), which reduces noise.
    Test-retest pairs are within a session without repositioning, so 'systematic' includes repositioning."""
    out = {}
    la = np.log(ok.slope.clip(lower=1e-3))
    ms = (la ** 2).groupby([ok.pipeline, ok.comparison, ok.protocol]).mean().unstack(['comparison', 'protocol'])
    sysd = {}
    for prot in PROTOCOLS:
        sysd[prot] = np.sqrt((ms[('inter_scanner', prot)] - ms[('test_retest', prot)]).clip(lower=0))
        out[prot] = {'rms_ln_slope_test_retest_pct': pipeline_summary(np.sqrt(ms[('test_retest', prot)]), 100),
                     'rms_ln_slope_inter_scanner_pct': pipeline_summary(np.sqrt(ms[('inter_scanner', prot)]), 100),
                     'systematic_rms_pct': pipeline_summary(sysd[prot], 100)}
    both = pd.DataFrame(sysd).dropna()
    out['share_of_pipelines_bridge_systematic_lower'] = r((both.bridge < both.local).mean(), 2)

    # run-averaged scans
    acq = emtp.acquisitions()
    groups = acq.groupby(['subject', 'scanner', 'protocol']).acquisition.apply(list)
    plan = ok[ok.comparison == 'inter_scanner'][['subject_x', 'scanner_x', 'scanner_y', 'protocol']].drop_duplicates()
    rows = []
    for pipe, pv in vals.items():
        avg = {}
        for key, acqs in groups.items():
            present = [pv[a] for a in acqs if a in pv]
            if present:
                labels = set.intersection(*(set(v) for v in present))
                avg[key] = {k: np.mean([v[k] for v in present]) for k in labels}
        for p in plan.itertuples():
            kx, ky = (p.subject_x, p.scanner_x, p.protocol), (p.subject_x, p.scanner_y, p.protocol)
            if kx in avg and ky in avg:
                f = fit_dicts(avg[kx], avg[ky])
                if f:
                    rows.append(dict(pipeline=pipe, protocol=p.protocol, scanner_pair=f'{p.scanner_x}|{p.scanner_y}',
                                     subject=p.subject_x, abs_slope_dev=abs(f['slope'] - 1)))
    av = pd.DataFrame(rows)
    av.to_csv(OUT / 'inter_scanner_run_averaged.csv.gz', index=False, float_format='%.6g',
              compression={'method': 'gzip', 'mtime': 0})
    per = av.groupby(['pipeline', 'protocol']).abs_slope_dev.median().unstack().dropna()
    for prot in PROTOCOLS:
        out[prot]['run_averaged_inter_scanner_median_abs_slope_dev_pct'] = pipeline_summary(per[prot], 100)
    out['run_averaged_share_of_pipelines_bridge_lower'] = r((per.bridge < per.local).mean(), 2)
    out['run_averaged_by_scanner_pair_pct'] = (
        av.groupby(['scanner_pair', 'protocol', 'pipeline']).abs_slope_dev.median()
        .groupby(['scanner_pair', 'protocol']).median().mul(100).round(2).unstack().to_dict('index'))
    return out


def headline(ok):
    """The agreement values the abstract and the issues quote, from a set of included pair fits."""
    pp = ok.groupby(['pipeline', 'comparison', 'protocol']).abs_slope_dev.median().unstack(['comparison', 'protocol'])
    tr = ok[ok.comparison == 'test_retest'].groupby(['pipeline', 'protocol']).slope.std().unstack()
    out = {}
    for prot in PROTOCOLS:
        out[f'test_retest_{prot}_median_pct'] = 100 * pp[('test_retest', prot)].median()
        out[f'test_retest_{prot}_sd_slope_pct'] = 100 * tr[prot].median()
        if ('inter_scanner', prot) in pp:
            d = pp[('inter_scanner', prot)].dropna()
            out[f'inter_scanner_{prot}_median_pct'] = 100 * d.median()
            out[f'inter_scanner_{prot}_p75_pct'] = 100 * d.quantile(.75)
            out[f'inter_scanner_{prot}_p95_pct'] = 100 * d.quantile(.95)
    if ('inter_scanner', 'bridge') in pp:
        both = pp['inter_scanner'][PROTOCOLS].dropna()
        out['inter_scanner_share_pipelines_bridge_lower'] = (both.bridge < both.local).mean()
        out['inter_scanner_median_bridge_minus_local_pct'] = 100 * (both.bridge - both.local).median()
        for q in (.75, .95):
            out[f'inter_scanner_p{int(100 * q)}_bridge_minus_local_pct'] = 100 * (both.bridge.quantile(q) - both.local.quantile(q))
        out['inter_scanner_pipelines_above_25pct_bridge_minus_local'] = float((both.bridge > .25).sum() - (both.local > .25).sum())
    return out


def bootstrap_subjects(ok, n_boot=500, seed=0):
    """Percentile 95% intervals over subjects resampled with replacement (pairs never cross subjects).
    With six subjects, four of them in inter-scanner pairs, these intervals are rough and optimistic."""
    rng = np.random.default_rng(seed)
    subjects = sorted(ok.subject_x.unique())
    by = {s: g for s, g in ok.groupby('subject_x')}
    reps = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, len(subjects))
        reps.append(headline(pd.concat([by[s] for s in pick], ignore_index=True)))
    reps = pd.DataFrame(reps)
    est = headline(ok)
    return {k: {'estimate': r(v, 2), 'ci95': [r(reps[k].quantile(.025), 2), r(reps[k].quantile(.975), 2)],
                'n_valid_replicates': int(reps[k].notna().sum())} for k, v in est.items()}


def pipeline_conversion(reg, vals):
    """#6: are conversion factors between pipelines global? Each pipeline is fitted against the consensus
    (median over pipelines of each region's value, per acquisition): a, b per pipeline and acquisition.
    Then (1) how much a varies across acquisitions, split by factor; (2) the Fig. 4 spread across pipelines
    before and after mapping every pipeline to consensus space with one global a, b per pipeline,
    estimated either on all acquisitions (in-sample) or only on the reference-site bridge scans and
    applied to everything else (does a factor learned on one scanner/protocol transfer?)."""
    acq = emtp.acquisitions().set_index('acquisition')
    cons = reg.groupby(['acquisition', 'label']).mean_ppm.median()
    cons = {a: g.droplevel(0).to_dict() for a, g in cons.groupby(level=0)}
    rows = []
    for pipe, pv in vals.items():
        for a, v in pv.items():
            f = fit_dicts(cons[a], v)
            if f:
                rows.append(dict(pipeline=pipe, acquisition=a, slope=f['slope'], intercept_ppm=f['intercept_ppm'],
                                 r2=f['r2']))
    pf = pd.DataFrame(rows).join(acq, on='acquisition')
    pf.to_csv(OUT / 'pipeline_vs_consensus.csv.gz', index=False, float_format='%.6g',
              compression={'method': 'gzip', 'mtime': 0})
    pf['ln_a'] = np.log(pf.slope.clip(lower=1e-3))
    # spread of ln a within each pipeline: overall, across runs (same subject/scanner/protocol),
    # across protocols, scanners and subjects (on run-averaged ln a)
    g = pf.groupby('pipeline')
    run_avg = pf.groupby(['pipeline', 'subject', 'scanner', 'protocol']).ln_a.mean().reset_index()
    var = pd.DataFrame({
        'total': g.ln_a.std(),
        'runs': pf.groupby(['pipeline', 'subject', 'scanner', 'protocol']).ln_a.std().groupby('pipeline').median(),
        'protocols': run_avg.groupby(['pipeline', 'subject', 'scanner']).ln_a.std().groupby('pipeline').median(),
        'scanners': run_avg.groupby(['pipeline', 'subject', 'protocol']).ln_a.std().groupby('pipeline').median(),
        'subjects': run_avg[run_avg.scanner == 'site1-achieva'].groupby(['pipeline', 'protocol']).ln_a.std()
        .groupby('pipeline').median(),
    })
    out = {'median_slope_vs_consensus': pipeline_summary(g.slope.median(), 1),
           'sd_of_slope_within_pipeline_pct': {k: pipeline_summary(var[k], 100) for k in var}}

    def converted(train):
        ab = train.groupby('pipeline')[['slope', 'intercept_ppm']].median()
        rc = reg.join(ab, on='pipeline')
        rc = rc[rc.slope.notna()].copy()
        rc['mean_ppm'] = (rc.mean_ppm - rc.intercept_ppm) / rc.slope
        return rc[reg.columns]

    ref_bridge = pf[(pf.scanner == 'site1-achieva') & (pf.protocol == 'bridge')]
    held_out = set(acq.index[~((acq.scanner == 'site1-achieva') & (acq.protocol == 'bridge'))])
    variants = {'none': reg, 'global_in_sample': converted(pf), 'global_from_reference_bridge': converted(ref_bridge)}
    for region in SPREAD_REGIONS:
        out[region] = {}
        for lab, rr in variants.items():
            s = emtp.spread(rr, region)
            sh = emtp.spread(rr[rr.acquisition.isin(held_out)], region)
            out[region][lab] = {'pipeline_sd_ppb': {p: r(s.loc['pipeline', p]) for p in PROTOCOLS},
                                'pipeline_sd_held_out_scans_ppb': {p: r(sh.loc['pipeline', p]) for p in PROTOCOLS},
                                'scanner_sd_ppb': {p: r(s.loc['scanner', p]) for p in PROTOCOLS},
                                'protocol_sd_ppb': r(s.loc['protocol', 'both']),
                                'between_subjects_sd_ppb': {p: r(s.loc['between_subjects', p]) for p in PROTOCOLS}}
    return out


def scanner_spread_corrected(reg, ok):
    """#4: the Fig. 4 scanner bar after mapping each scanner to the subject's reference-site scanner with one
    transform per pipeline and scanner pair (median a, b of that pair's inter-scanner fits, reference = x).
    Most scanner pairs have one subject, so this is in-sample: the best case."""
    isc = ok[ok.comparison == 'inter_scanner']
    isc = isc[isc.scanner_x.str.startswith('site1-')]
    ab = isc.groupby(['pipeline', 'subject_x', 'protocol', 'scanner_y'])[['slope', 'intercept_ppm']].median()
    ab.index.names = ['pipeline', 'subject', 'protocol', 'scanner']
    acq = emtp.acquisitions()
    rc = reg.merge(acq, on='acquisition').join(ab, on=['pipeline', 'subject', 'protocol', 'scanner'])
    m = rc.slope.notna()
    rc.loc[m, 'mean_ppm'] = (rc.mean_ppm[m] - rc.intercept_ppm[m]) / rc.slope[m]
    rc = rc[reg.columns]
    out = {}
    for region in SPREAD_REGIONS:
        before, after = emtp.spread(reg, region), emtp.spread(rc, region)
        out[region] = {p: {'scanner_sd_ppb': r(before.loc['scanner', p]),
                           'scanner_sd_corrected_ppb': r(after.loc['scanner', p])} for p in PROTOCOLS}
    return out


def first_run_effect(ok):
    """#10: is the bridge's higher rescan deviation a first-scan effect? Bridge run 1 opens every session
    (order B1 L1 B2 L2 B3 L3). Agreement with and without pairs that involve run 1, per protocol."""
    tr = ok[ok.comparison == 'test_retest']
    t = tr.groupby(['subject_x', 'scanner_x', 'protocol', 'run_x', 'run_y', 'pipeline']).abs_slope_dev.median() \
        .groupby(['subject_x', 'scanner_x', 'protocol', 'run_x', 'run_y']).median().mul(100).round(2) \
        .unstack(['run_x', 'run_y'])
    t.columns = [f'{a}_vs_{b}' for a, b in t.columns]
    t.to_csv(OUT / 'test_retest_by_run_pair.csv')
    has_run1 = (ok.run_x == 'run1') | (ok.run_y == 'run1')
    out = {}
    for comp in ('test_retest', 'inter_scanner'):
        out[comp] = {}
        for prot in PROTOCOLS:
            d = ok[(ok.comparison == comp) & (ok.protocol == prot)]
            m = has_run1[d.index]
            out[comp][prot] = {
                'all_pairs_pct': pipeline_summary(d.groupby('pipeline').abs_slope_dev.median(), 100),
                'pairs_with_run1_pct': pipeline_summary(d[m].groupby('pipeline').abs_slope_dev.median(), 100),
                'pairs_without_run1_pct': pipeline_summary(d[~m].groupby('pipeline').abs_slope_dev.median(), 100)}
    # how often run 1 is the odd one out: the run whose two pairs are both above the third pair
    runs = ['run1', 'run2', 'run3']
    pairs = {'run1': ['run1_vs_run2', 'run1_vs_run3'], 'run2': ['run1_vs_run2', 'run2_vs_run3'],
             'run3': ['run1_vs_run3', 'run2_vs_run3']}
    odd = t.apply(lambda row: max(runs, key=lambda k: row[pairs[k]].min()), axis=1)
    out['odd_run_per_session'] = {prot: odd.xs(prot, level='protocol').value_counts().to_dict() for prot in PROTOCOLS}
    return out


def field_drift():
    """#10: field drift between runs (data/b0_drift.csv: each run against run 3 of the same protocol and
    session). Run 1 against run 2 as the question: is the first scan's field different?"""
    d = pd.read_csv(emtp.DATA / 'b0_drift.csv').merge(emtp.acquisitions(), on='acquisition')
    d['gradient_hz_per_cm'] = np.sqrt(d.gradient_x_hz_per_cm ** 2 + d.gradient_y_hz_per_cm ** 2 + d.gradient_z_hz_per_cm ** 2)
    d['abs_median_hz'] = d.median_hz.abs()
    cols = ['abs_median_hz', 'gradient_hz_per_cm', 'residual_sd_hz']
    t = d.groupby(['protocol', 'run'])[cols].median().round(3)
    d[['acquisition', 'scanner', 'protocol', 'run'] + cols].round(3).to_csv(OUT / 'field_drift.csv', index=False)
    return {f'{p} {run} vs run3': row.to_dict() for (p, run), row in t.iterrows()}


def run1_image_quality():
    """#10: does bridge run 1 show more motion or drift during the run? data/image_quality.csv (ghosting
    outside the head, edge sharpness in the brain) per run, relative to the mean of runs 2 and 3 of the
    same session and protocol, set against how odd run 1 is in test-retest (run-1 pairs minus run 2 vs 3)."""
    from scipy.stats import spearmanr
    q = pd.read_csv(emtp.DATA / 'image_quality.csv').merge(emtp.acquisitions(), on='acquisition')
    rel = []
    for (subj, sc, prot), g in q.groupby(['subject', 'scanner', 'protocol']):
        g = g.set_index('run')
        later = g.loc[['run2', 'run3']]
        row = {'subject': subj, 'scanner': sc, 'protocol': prot}
        for m in ('ghost_ratio', 'sharpness_per_mm'):
            row[f'{m}_run1_vs_later_pct'] = 100 * (g.loc['run1', m] / later[m].mean() - 1)
            row[f'{m}_run1_is_worst'] = bool(g[m].idxmax() == 'run1') if m == 'ghost_ratio' else bool(g[m].idxmin() == 'run1')
        rel.append(row)
    rel = pd.DataFrame(rel)
    tr = pd.read_csv(OUT / 'test_retest_by_run_pair.csv')
    tr['run1_excess_pct'] = tr[['run1_vs_run2', 'run1_vs_run3']].mean(axis=1) - tr.run2_vs_run3
    rel = rel.merge(tr[['subject_x', 'scanner_x', 'protocol', 'run1_excess_pct']]
                    .rename(columns={'subject_x': 'subject', 'scanner_x': 'scanner'}), on=['subject', 'scanner', 'protocol'])
    rel.round(3).to_csv(OUT / 'run1_image_quality.csv', index=False)
    out = {}
    for prot, g in rel.groupby('protocol'):
        out[prot] = {
            'ghost_run1_vs_later_pct_median': r(g.ghost_ratio_run1_vs_later_pct.median(), 1),
            'sessions_run1_most_ghosting': f'{int(g.ghost_ratio_run1_is_worst.sum())} of {len(g)}',
            'sharpness_run1_vs_later_pct_median': r(g.sharpness_per_mm_run1_vs_later_pct.median(), 1),
            'sessions_run1_least_sharp': f'{int(g.sharpness_per_mm_run1_is_worst.sum())} of {len(g)}',
            'spearman_ghost_vs_run1_excess': r(spearmanr(g.ghost_ratio_run1_vs_later_pct, g.run1_excess_pct)[0], 2),
            'spearman_sharpness_vs_run1_excess': r(spearmanr(g.sharpness_per_mm_run1_vs_later_pct, g.run1_excess_pct)[0], 2),
        }
    return out


def within_run_drift():
    """#10: field drift during each run, from the echo-to-echo shift (data/echo_shift.csv: Hz per voxel
    axis). Run 1 against runs 2 and 3 of the same session and protocol, per axis, and against how odd run 1
    is in test-retest."""
    from scipy.stats import spearmanr
    q = pd.read_csv(emtp.DATA / 'echo_shift.csv').merge(emtp.acquisitions(), on='acquisition')
    axes = [f'drift_hz_axis{i}' for i in range(3)]
    rows = []
    for (subj, sc, prot), g in q.groupby(['subject', 'scanner', 'protocol']):
        g = g.set_index('run')
        row = {'subject': subj, 'scanner': sc, 'protocol': prot}
        for a in axes:
            row[f'{a}_run1'] = g.loc['run1', a]
            row[f'{a}_later_mean'] = g.loc[['run2', 'run3'], a].mean()
            row[f'{a}_later_absdiff'] = abs(g.loc['run2', a] - g.loc['run3', a])   # spread between two ordinary runs
            row[f'{a}_run1_minus_later'] = g.loc['run1', a] - row[f'{a}_later_mean']
        rows.append(row)
    rel = pd.DataFrame(rows)
    tr = pd.read_csv(OUT / 'test_retest_by_run_pair.csv')
    tr['run1_excess_pct'] = tr[['run1_vs_run2', 'run1_vs_run3']].mean(axis=1) - tr.run2_vs_run3
    rel = rel.merge(tr[['subject_x', 'scanner_x', 'protocol', 'run1_excess_pct']]
                    .rename(columns={'subject_x': 'subject', 'scanner_x': 'scanner'}), on=['subject', 'scanner', 'protocol'])
    rel.round(3).to_csv(OUT / 'within_run_drift.csv', index=False)
    out = {}
    for prot, g in rel.groupby('protocol'):
        out[prot] = {a.replace('drift_hz_', ''): {
            'run1_median_hz': r(g[f'{a}_run1'].median(), 2),
            'later_median_hz': r(g[f'{a}_later_mean'].median(), 2),
            'run1_minus_later_median_hz': r(g[f'{a}_run1_minus_later'].median(), 2),
            'sessions_run1_higher': f'{int((g[f"{a}_run1_minus_later"] > 0).sum())} of {len(g)}',
            'run2_vs_run3_absdiff_median_hz': r(g[f'{a}_later_absdiff'].median(), 2),
            'spearman_abs_run1_minus_later_vs_run1_excess': r(spearmanr(g[f'{a}_run1_minus_later'].abs(), g.run1_excess_pct)[0], 2),
        } for a in axes}
    return out


def b0_angle(pose, a, b, frame):
    """Change in head orientation relative to B0 between two acquisitions (degrees)."""
    va = pose.loc[a, [f'{frame}_b0_{k}' for k in 'xyz']].to_numpy(float)
    vb = pose.loc[b, [f'{frame}_b0_{k}' for k in 'xyz']].to_numpy(float)
    # atan2 of the cross and dot products: stable for small angles (arccos of a dot product near 1 is not,
    # and the stored unit vectors carry 5 decimals)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(va, vb)), va @ vb)))


def repositioning(ok):
    """#9/#10: head orientation relative to B0 (data/head_pose.csv, rigid registration of the first-echo
    magnitude) against slope deviation. Inter-scanner pairs use the subject frame (reference-site bridge
    run 1), test-retest pairs the session frame (same session's bridge run 1). Per scan pair, |a-1| is the
    median over pipelines."""
    from scipy.stats import spearmanr
    pose = pd.read_csv(emtp.DATA / 'head_pose.csv').set_index('acquisition')
    out = {}
    per_pair = ok.groupby(['comparison', 'protocol', 'acquisition_x', 'acquisition_y']).abs_slope_dev.median().reset_index()
    for comp, frame in (('inter_scanner', 'subject'), ('test_retest', 'session')):
        d = per_pair[per_pair.comparison == comp].copy()
        d['b0_angle_deg'] = [b0_angle(pose, a, b, frame) for a, b in zip(d.acquisition_x, d.acquisition_y)]
        out[comp] = {}
        for prot, g in d.groupby('protocol'):
            rho, p = spearmanr(g.b0_angle_deg, g.abs_slope_dev)
            # run pairs within a session pair are not independent: rho is descriptive, no p-value
            out[comp][prot] = {'b0_angle_deg': pipeline_summary(g.b0_angle_deg, 1, 'n_scan_pairs') | {'max': r(g.b0_angle_deg.max(), 2)},
                               'spearman_rho_vs_abs_slope_dev': r(rho, 2)}
        if comp == 'test_retest':
            d['with_run1'] = d.acquisition_x.str.endswith('run1') | d.acquisition_y.str.endswith('run1')
            out[comp]['b0_angle_deg_with_vs_without_run1'] = {
                prot: {('with_run1' if k else 'without_run1'): r(v, 2) for k, v in g.groupby('with_run1').b0_angle_deg.median().items()}
                for prot, g in d.groupby('protocol')}
        if comp == 'inter_scanner':
            scanner = lambda a: a.str.split('-').str[1:3].str.join('-')
            d['scanner_pair'] = scanner(d.acquisition_x) + '|' + scanner(d.acquisition_y)
            d['subject'] = d.acquisition_x.str.split('-').str[0]
            t = d.groupby(['subject', 'scanner_pair', 'protocol']).agg(b0_angle_deg=('b0_angle_deg', 'median'),
                                                                     abs_slope_dev_pct=('abs_slope_dev', 'median'))
            t['abs_slope_dev_pct'] *= 100
            out['inter_scanner_session_level_spearman_rho'] = {
                prot: r(spearmanr(g.b0_angle_deg, g.abs_slope_dev_pct)[0], 2) for prot, g in t.groupby('protocol')}
            t.round(2).to_csv(OUT / 'repositioning_inter_scanner.csv')
            out['inter_scanner_by_session_pair'] = {f'{k[0]} {k[1]} {k[2]}': {c: r(v, 2) for c, v in row.items()}
                                                    for k, row in t.iterrows()}
    # motion between runs within a session, by protocol (relative to the session's bridge run 1)
    acq = emtp.acquisitions().set_index('acquisition')
    pj = pose.join(acq)
    mv = pj[pj.index != pj.session_reference]
    out['within_session_vs_bridge_run1'] = {
        prot: {'rotation_deg': pipeline_summary(g.session_rotation_deg, 1, 'n_scans') | {'max': r(g.session_rotation_deg.max(), 2)},
               'translation_mm': pipeline_summary(g.session_translation_mm, 1, 'n_scans') | {'max': r(g.session_translation_mm.max(), 2)}}
        for prot, g in mv.groupby('protocol')}
    return out


if __name__ == '__main__':
    reg = emtp.regions()
    fits = emtp.pair_fits(reg)
    ok = fits[fits.excluded == ''].copy()
    site = lambda s: s.str.split('-').str[0]
    ok['scanner_pair'] = ok.scanner_x + '|' + ok.scanner_y
    ok['site_relation'] = np.where(site(ok.scanner_x) == site(ok.scanner_y), 'intra_site', 'inter_site')
    vx, vy = ok.scanner_x.map(emtp.VENDOR), ok.scanner_y.map(emtp.VENDOR)
    ok['vendor_relation'] = np.where(vx == vy, 'same_vendor', 'cross_vendor')
    ok['vendor_pair'] = [' vs '.join(sorted(p)) for p in zip(vx, vy)]
    n = {
        'issue_2_slope_spread': slope_spread(ok),
        'issue_3_corrected_loa': corrected_loa(ok),
        'issue_7_8_scanner_relation': by_scanner_relation(ok),
        'issue_4_protocol_spread': protocol_spread_corrected(reg, ok),
    }
    vals = region_values(reg)
    n['issue_5_calibration'] = calibration_loso(vals, ok)
    n['issue_10_noise_vs_systematic'] = noise_vs_systematic(vals, ok)
    n['issue_4_scanner_spread'] = scanner_spread_corrected(reg, ok)
    n['issue_6_pipeline_conversion'] = pipeline_conversion(reg, vals)
    n['bootstrap_over_subjects'] = bootstrap_subjects(ok)
    n['issue_10_first_run'] = first_run_effect(ok)
    if (emtp.DATA / 'b0_drift.csv').exists():
        n['issue_10_field_drift'] = field_drift()
    if (emtp.DATA / 'echo_shift.csv').exists():
        n['issue_10_within_run_drift'] = within_run_drift()
    if (emtp.DATA / 'image_quality.csv').exists():
        n['issue_10_run1_image_quality'] = run1_image_quality()
    if (emtp.DATA / 'head_pose.csv').exists():
        n['issue_9_repositioning'] = repositioning(ok)
    (OUT / 'followup.json').write_text(json.dumps(n, indent=2) + '\n')
    print(json.dumps(n, indent=2))
