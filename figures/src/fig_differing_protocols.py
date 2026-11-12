"""#14: the bridge-vs-local analyses on the local protocols that differ from the bridge (emtp.DIFFERENT_LOCALS, D8),
good pipelines (top 10% by simulation XSIM, D9). Pair fits from results/pair_fits.csv.gz, calibration from
results/followup/calibration_loso.csv.gz (analysis/followup.py)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
import emtp  # noqa: E402
from followup_plots import DATA_LABEL, GREY, INK, INK2, PURPLE, SCANNER, boxes, save  # noqa: E402

CLASS = lambda s: 'different' if s in emtp.DIFFERENT_LOCALS else 'borderline' if s in emtp.BORDERLINE_LOCALS else 'similar'


def make(fraction=emtp.GOOD_PIPELINE_FRACTION):
    good = emtp.good_pipelines(fraction)
    f = pd.read_csv(ROOT / 'results/pair_fits.csv.gz', dtype={'excluded': str})
    ok = f[f.excluded.isna() & f.pipeline.isin(good)]
    pct = lambda d, col='abs_slope_dev': 100 * d.groupby('pipeline')[col].median()
    scanners = [s for s in SCANNER if s in set(f.scanner_x[f.comparison == 'inter_protocol'])]
    fig, axs = plt.subplots(1, 3, figsize=(9 + 1.5 * len(scanners), 4.8),
                            gridspec_kw={'width_ratios': [0.62 * len(scanners), 1.9, 1.0]})

    # A: protocol effect (local vs bridge) against the rescan floor, per scanner
    ip, tr = ok[ok.comparison == 'inter_protocol'], ok[ok.comparison == 'test_retest']
    groups = {}
    for s in scanners:
        n = ip[ip.scanner_x == s].subject_x.nunique()
        groups[f"{SCANNER[s]}\n{n} subj.\n{CLASS(s)}"] = {'local vs bridge': pct(ip[ip.scanner_x == s]),
                                                         'rescan, same scanner': pct(tr[tr.scanner_x == s])}
    # axis cap: 14 %, or higher so that no box (IQR) is cut (site 2 single-echo locals)
    q75 = max(float(np.nanpercentile(v, 75)) for g in groups.values() for v in g.values() if len(v))
    cap_a = max(14.0, float(np.ceil(q75 / 5) * 5))
    boxes(groups, '|a − 1| per pipeline (%)', ax=axs[0], ylim=(0, cap_a * 1.107), cap=cap_a,
          colors={'local vs bridge': PURPLE, 'rescan, same scanner': GREY},
          title='A  Protocol effect vs rescan floor (same scanner and session)')
    for t in axs[0].get_xticklabels():
        if 'different' in t.get_text():
            t.set_color(INK); t.set_fontweight('bold')

    # B: inter-scanner agreement, bridge vs local on the same pairs
    isc = ok[ok.comparison == 'inter_scanner']
    key = ['pipeline', 'subject_x', 'scanner_x', 'scanner_y', 'run_x', 'run_y']
    def matched(d):
        m = d.pivot_table(index=key, columns='protocol', values='abs_slope_dev').dropna()
        return {p: 100 * m[p].groupby('pipeline').median() for p in ('bridge', 'local')}
    diff = isc[isc.scanner_x.isin(emtp.DIFFERENT_LOCALS) | isc.scanner_y.isin(emtp.DIFFERENT_LOCALS)]
    sim_sc = set(scanners) - set(emtp.DIFFERENT_LOCALS) - set(emtp.BORDERLINE_LOCALS)
    sim = isc[isc.scanner_x.isin(sim_sc) & isc.scanner_y.isin(sim_sc)]
    np_ = lambda d: d[['scanner_x', 'scanner_y']].drop_duplicates().shape[0]
    plural = lambda n, w: f'{n} {w}' + ('s' if n != 1 else '')
    boxes({f'Pairs with a different\nlocal ({plural(np_(diff), "scanner pair")},\n{plural(diff.subject_x.nunique(), "subject")})': matched(diff),
           f'Similar locals only\n({plural(np_(sim), "scanner pair")},\n{plural(sim.subject_x.nunique(), "subject")})': matched(sim)},
          'inter-scanner |a − 1| per pipeline (%)', ax=axs[1], ylim=(0, 15.5), cap=14,
          title='B  Inter-scanner, bridge vs local on the same pairs')

    # C: local -> bridge calibration learned on the other subject(s) (different locals on scanners with 2+ subjects)
    c = pd.read_csv(ROOT / 'results/followup/calibration_loso.csv.gz')
    c = c[c.pipeline.isin(good) & c.scanner.isin(emtp.DIFFERENT_LOCALS)]
    calib = ', '.join(SCANNER[s] for s in sorted(c.scanner.unique()))
    boxes({'none': {'u': pct(c, 'none_abs_slope_dev')},
           'from the\nother\nsubject': {'c': pct(c, 'loso_abs_slope_dev')}},
          'local vs bridge |a − 1| (%)', ax=axs[2], ylim=(0, 15.5), cap=14, legend=False,
          colors={'u': PURPLE, 'c': GREY}, title=f'C  Local → bridge calibration as in #5, {calib}')
    fig.suptitle(f'Local protocols that differ from the bridge, top {fraction:.0%} pipelines by simulation XSIM '
                 f'({len(good)}), {DATA_LABEL}', x=0.06, ha='left', fontsize=12, color=INK)
    fig.text(0.06, -0.06, 'Box 25–75 %, whiskers 5–95 %, line = median over pipelines (printed); each pipeline is the '
             'median over its pairs. a: orthogonal fit of regional χ. Different locals (D8): S2 single echo 0.8 mm (three '
             'scanners, one subject), S4 Skyra 1×1×2 mm, S5 Prisma 0.75 mm in-plane. C needs 2+ subjects per scanner.',
             fontsize=8.5, color=INK2, va='top')
    for ax in axs:
        ax.title.set_y(1.03)
    fig.subplots_adjust(wspace=0.32)
    return fig


if __name__ == '__main__':
    save(make(), 'differing_protocols')
    save(make(emtp.GOOD_PIPELINE_FRACTION_SENSITIVITY), 'differing_protocols_top25')
