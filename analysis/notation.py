"""Notation for followup.ipynb: the analysis written the way the maths is.

A quantity is a pandas Series (or DataFrame) whose index levels are its subscripts. For example the slope
a[pipeline, comparison, protocol, subject, scanner_x, scanner_y, ..., acquisition_x, acquisition_y] has one value per
pipeline and scan pair. A statistic names the subscripts that stay free (`each`) and reduces over all the others:

    m = median(abs(a - 1), each=('pipeline', 'protocol'))      # m_{p,π} = median_k |a_{p,π,k} − 1|

`restrict` keeps the values whose subscripts take given values (a subset of pairs or scans), `exclude` drops them,
and `by` splits a quantity into one series per value of a subscript. SDs are sample SDs (ddof = 1) throughout.
"""
import numpy as np
import pandas as pd

import emtp
from followup import fit_dicts

# ---------------------------------------------------------------- statistics over subscripts


def _reduce(x, each, how, **kw):
    each = [each] if isinstance(each, str) else list(each)
    if not each:
        return getattr(x, how)(**kw)
    return getattr(x.groupby(level=each, sort=True), how)(**kw)


def median(x, each=()):
    """Median over every subscript not in `each`."""
    return _reduce(x, each, 'median')


def mean(x, each=()):
    """Mean over every subscript not in `each`."""
    return _reduce(x, each, 'mean')


def sd(x, each=()):
    """Sample SD (ddof = 1) over every subscript not in `each`; undefined (NaN) for a single value."""
    return _reduce(x, each, 'std', ddof=1)


def quantile(x, q, each=()):
    return _reduce(x, each, 'quantile', q=q)


def iqr(x, each=()):
    """Interquartile range (75th − 25th percentile) over every subscript not in `each`."""
    return quantile(x, .75, each) - quantile(x, .25, each)


def total(x, each=()):
    """Sum over every subscript not in `each` (a count, for a true/false quantity)."""
    return _reduce(x, each, 'sum')


def _match(x, conds):
    keep = np.ones(len(x), bool)
    for level, v in conds.items():
        values = x.index.get_level_values(level)
        keep &= np.asarray(values.isin(v) if isinstance(v, (list, tuple, set)) else values == v)
    return keep


def restrict(x, **conds):
    """The values whose subscripts match, e.g. restrict(a, comparison='test_retest', protocol=['bridge'])."""
    return x[_match(x, conds)]


def exclude(x, **conds):
    """The values whose subscripts do not match all of `conds`."""
    return x[~_match(x, conds)]


def by(x, level):
    """{value: x restricted to that value of `level`, with that subscript dropped}."""
    return {k: g.droplevel(level) for k, g in x.groupby(level=level, sort=True)}


def having(x, other):
    """The values of x whose subscripts shared with `other` appear in `other` (e.g. pairs that have a transform)."""
    keys = list(other.index.names)
    idx = pd.MultiIndex.from_arrays([x.index.get_level_values(k) for k in keys]) if len(keys) > 1 \
        else x.index.get_level_values(keys[0])
    return x[np.asarray(idx.isin(other.index))]


# ---------------------------------------------------------------- the data as subscripted quantities

PAIR = ['pipeline', 'comparison', 'protocol', 'subject', 'scanner_x', 'scanner_y', 'scanner_pair', 'same_site',
        'run_x', 'run_y', 'with_run1', 'acquisition_x', 'acquisition_y']
SCAN = ['subject', 'scanner', 'protocol', 'run']


def pair_fits(reg):
    """The orthogonal fit of every pipeline on every scan pair (emtp.pair_fits, QSM-CI exclusions applied):
    a (slope), b (intercept, ppb), loa and loa_corrected (± ppb), indexed by the pair's subscripts."""
    f = emtp.pair_fits(reg)
    f = f[f.excluded == ''].rename(columns={'subject_x': 'subject'})
    site = lambda s: s.str.split('-').str[0]
    f['scanner_pair'] = f.scanner_x + ' – ' + f.scanner_y
    f['same_site'] = site(f.scanner_x) == site(f.scanner_y)
    f['with_run1'] = (f.run_x == 'run1') | (f.run_y == 'run1')
    out = pd.DataFrame({'a': f.slope.values, 'b': 1000 * f.intercept_ppm.values, 'loa': 1000 * f.ba_loa_ppm.values,
                        'loa_corrected': 1000 * f.ba_loa_corrected_ppm.values})
    out.index = pd.MultiIndex.from_frame(f[PAIR])
    return out


def chi_by_label(reg):
    """χ[pipeline, subject, scanner, protocol, run, region]: mean χ of each SynthSeg label (ppb, minus the
    whole-brain mean of that reconstruction)."""
    r = reg.merge(emtp.acquisitions(), on='acquisition')
    return pd.Series(1000 * r.mean_ppm.values, index=pd.MultiIndex.from_frame(r[['pipeline'] + SCAN + ['region']]), name='chi')


def chi_region(reg, region):
    """χ[pipeline, subject, scanner, protocol, run] of a bilateral region (ppb): the voxel-weighted mean over the
    hemispheres present."""
    b = emtp.bilateral(reg, region)
    return pd.Series(b[region].values, index=pd.MultiIndex.from_frame(b[['pipeline'] + SCAN]), name=region)


# ---------------------------------------------------------------- lines: fitting and undoing them


def fit_line(x, y):
    """Orthogonal fit y ≈ a x + b over the labels present in both (Series indexed by label, ppb), with the QSM-CI
    exclusions (≥ 8 labels, not flat, |a − 1| ≤ 2). Returns (a, b in ppb), or None if excluded."""
    f = fit_dicts((x / 1000).to_dict(), (y / 1000).to_dict())
    return None if f is None else (f['slope'], 1000 * f['intercept_ppm'])


def undo_line(x, line):
    """(x − b) / a, where the line (a, b) is looked up by the subscripts it is indexed by (a subset of x's).
    Values without a line are dropped."""
    j = x.to_frame('x').join(line[['a', 'b']], on=list(line.index.names)).dropna(subset=['a'])
    return (j.x - j.b) / j.a


def _vectors(chi_lab):
    """{(pipeline, subject, scanner, protocol, run): χ over labels}"""
    keys = ['pipeline'] + SCAN
    return {k: g.droplevel(keys) for k, g in chi_lab.groupby(level=keys, sort=False)}


def fit_to_reference(chi_lab, ref):
    """For every pipeline p and scan: the line χ[p, scan, ·] ≈ a · ref[scan, ·] + b, fitted over the labels.
    `ref` is indexed by the scan subscripts and label. Returns (a, b) indexed by pipeline and scan."""
    refs = {k: g.droplevel(SCAN) for k, g in ref.groupby(level=SCAN, sort=False)}
    rows, idx = [], []
    for key, y in _vectors(chi_lab).items():
        f = fit_line(refs[key[1:]], y)
        if f:
            rows.append(f); idx.append(key)
    return pd.DataFrame(rows, columns=['a', 'b'], index=pd.MultiIndex.from_tuples(idx, names=['pipeline'] + SCAN))


def refit_local_into_bridge(chi_lab, pairs, line):
    """For each inter-protocol pair (bridge run k = x, local run k = y): map the local values into bridge space with
    the line for that pipeline, scanner and subject, y → (y − b) / a, and refit against the bridge. Returns |a' − 1|
    indexed like `pairs` (NaN where there is no line or the refit is excluded)."""
    vec = _vectors(chi_lab)
    out = []
    for k in pairs.index:
        d = dict(zip(pairs.index.names, k))
        key = (d['pipeline'], d['scanner_x'], d['subject'])
        if key not in line.index:
            out.append(np.nan); continue
        a, b = line.loc[key, ['a', 'b']]
        x = vec[(d['pipeline'], d['subject'], d['scanner_x'], 'bridge', d['run_x'])]
        y = vec[(d['pipeline'], d['subject'], d['scanner_y'], 'local', d['run_y'])]
        f = fit_line(x, (y - b) / a)
        out.append(abs(f[0] - 1) if f else np.nan)
    return pd.Series(out, index=pairs.index)


def leave_one_out(x, over):
    """x̂[…, s] = median over s' ≠ s of x[…, s'], all other subscripts fixed. Undefined (dropped) when s is the only
    value of `over` for those subscripts."""
    rest = [n for n in x.index.names if n != over]
    rows, idx = [], []
    for _, g in x.groupby(level=rest, sort=False):
        values = g.index.get_level_values(over)
        for i, s in enumerate(values):
            others = g[values != s]
            if len(others):
                rows.append(others.median()); idx.append(g.index[i])
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=x.index.names))
