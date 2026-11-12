"""Check results/pair_fits.csv.gz (analysis/abstract_numbers.py) against QSM-CI's published per-pipeline medians.

QSM-CI's results/repro.json `summary` holds, per pipeline and comparison class, the median |a - 1| over the included
pairs. emtp.pair_fits reproduces QSM-CI's pairs, fit and exclusions, so the medians should agree to QSM-CI's 4-decimal
rounding.
  python scripts/check_qsmci_fits.py --qsmci ~/code/QSM-CI [--ref origin/main]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
import emtp  # noqa: E402
from extract_qsmci import read_qsmci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--qsmci', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--tol', type=float, default=1e-4)
    a = ap.parse_args()
    repro = read_qsmci(a.qsmci, a.ref, 'results/repro.json')
    s = repro['summary']
    fits = pd.read_csv(ROOT / 'results' / 'pair_fits.csv.gz', dtype={'excluded': str}).fillna({'excluded': ''})
    ours = fits[fits.excluded == ''].groupby(['pipeline', 'comparison']).abs_slope_dev.median().unstack()
    worst, bad = 0.0, []
    for comp in emtp.COMPARISONS:
        theirs = pd.Series(s[f'{comp}_median_abs_slope_dev'], index=repro['pipelines'], dtype=float)
        d = (ours[comp].reindex(theirs.index) - theirs).abs()
        only = d.index[d.isna() & theirs.notna()].tolist() + d.index[d.isna() & ours[comp].reindex(theirs.index).notna()].tolist()
        off = d[d > a.tol + 1e-9]
        print(f'{comp}: {int((d <= a.tol + 1e-9).sum())} of {len(theirs)} pipelines agree; {len(off)} differ; '
              f'{len(only)} with a value on one side only')
        if len(off):
            bad.append(comp)
            print('  differ:', off.sort_values(ascending=False).head(10).round(4).to_dict())
        if len(only):
            print('  one side only:', only[:10])
        worst = max(worst, float(np.nanmax(d.to_numpy())) if len(d) else 0.0)
    print(f'largest difference {worst:.5f}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
