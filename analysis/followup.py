"""Follow-up analyses for the open GitHub issues; writes results/followup/:
  followup.json               the summary values, one key per issue
  slope_spread.csv            #2  per pipeline: SD of a over test-retest pairs, IQR of |a-1| across scanner pairs
  inter_scanner_pairs.csv     #7/#8  per scanner pair and protocol: median |a-1| over pipelines
  calibration_loso.csv.gz     #5  per local-vs-bridge pair: |a-1| and LoA uncorrected, leave-one-subject-out, in-sample
  inter_scanner_run_averaged.csv.gz  #10  inter-scanner fits on run-averaged scans
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


def pipeline_summary(per_pipe, scale=1):
    """Median and IQR over pipelines of a per-pipeline series."""
    d = per_pipe.dropna() * scale
    return {'median': r(d.median(), 2), 'p25': r(d.quantile(.25), 2), 'p75': r(d.quantile(.75), 2), 'n_pipelines': int(len(d))}


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
    (OUT / 'followup.json').write_text(json.dumps(n, indent=2) + '\n')
    print(json.dumps(n, indent=2))
