"""Follow-up analyses for the open GitHub issues; writes results/followup/:
  followup.json               the summary values, one key per issue
  slope_spread.csv            #2  per pipeline: SD of a over test-retest pairs, IQR of |a-1| across scanner pairs
  inter_scanner_pairs.csv     #7/#8  per scanner pair and protocol: median |a-1| over pipelines
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
    (OUT / 'followup.json').write_text(json.dumps(n, indent=2) + '\n')
    print(json.dumps(n, indent=2))
