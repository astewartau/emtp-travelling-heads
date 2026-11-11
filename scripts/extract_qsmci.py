"""QSM-CI harmonization results -> data/regional_chi.csv.gz, data/acquisitions.csv and data/pipelines.csv.

Inputs (see extraction/README.md):
  --rois        repro_rois.json from the QSM-CI run tree (results/repro_rois.json, written by harvest.sh; per
                pipeline and acquisition, the SynthSeg label statistics). Not committed to QSM-CI: it is large.
  --qsmci       a QSM-CI checkout; its results/repro.json (the published traveling-head payload: pipelines and
                acquisitions) and results/index.json (simulation scores) are read. Use --ref to read them from a
                git ref (e.g. origin/main) without switching the checkout's branch.
  --scanner-ids CSV qsmci_scanner,scanner mapping QSM-CI's scanner ids to the site<N>-<model> ids used here. Not
                in this repository; scanner ids already in site<N>-<model> form pass through unchanged.
Only the pipelines and acquisitions in the published payload are written (the run tree also holds runs the site
leaves out), so this repository analyses what QSM-CI shows.

pipelines.csv takes each pipeline's full-pipeline simulation scores (QSM Challenge 2.0 phantom, `sim`) from the
QSM-CI run whose slug is the pipeline's name; pipelines without such a run get none (emtp.good_pipelines leaves
them out). A step may carry a variant tag (`<method>.<tag>`, e.g. a parity or deconvolution setting);
`default_settings` is true when no step does.

  python scripts/extract_qsmci.py --rois repro_rois.json --qsmci ~/code/QSM-CI --ref origin/main \
      --scanner-ids scanner_ids.csv [--out data]
Prints a check report (missing acquisitions or pipelines, unmapped scanners, non-finite values, unscored
pipelines) and exits non-zero if the outputs are inconsistent.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROI_COLUMNS = ['pipeline', 'acquisition', 'label', 'region', 'n_voxels', 'mean_ppm', 'median_ppm', 'std_ppm',
               'brain_mean_ppm']
TRAVELING_HEADS = 's00'   # acquisition ids of the traveling heads (the pilot cohort's start otherwise)


def read_qsmci(qsmci, ref, path):
    if ref:
        return json.loads(subprocess.run(['git', '-C', str(qsmci), 'show', f'{ref}:{path}'], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads((Path(qsmci) / path).read_text())


def renamer(path):
    m = pd.read_csv(path)
    m = dict(zip(m.qsmci_scanner, m.scanner))

    def scanner(s):
        return m.get(s, s)

    def acquisition(a):
        subj, rest = a.split('-', 1)
        for old, new in m.items():
            if rest.startswith(old + '-'):
                return f'{subj}-{new}{rest[len(old):]}'
        return a
    return scanner, acquisition


def steps(pipeline):
    """(n_steps, field_mapping, background_removal, dipole_inversion): three steps are field mapping, background
    removal and dipole inversion; two are field mapping and a combined background removal + inversion; one is an
    end-to-end method (listed as the inversion)."""
    s = pipeline.split('+')
    if len(s) == 3:
        return 3, s[0], s[1], s[2]
    if len(s) == 2:
        return 2, s[0], '', s[1]
    return 1, '', '', s[0]


def sim_scores(index):
    """slug -> metrics of the simulation (QSM Challenge 2.0) runs; a composed run wins over an isolated one with
    the same slug (an end-to-end method appears as both, with the same scores)."""
    out = {}
    for r in index['runs']:
        if r.get('track') != 'sim' or r.get('phantom') != 'sim' or r.get('status') != 'ok':
            continue
        if r['slug'] not in out or r.get('mode') == 'composed':
            out[r['slug']] = r.get('metrics') or {}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rois', required=True)
    ap.add_argument('--qsmci', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--scanner-ids', required=True)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'data'))
    a = ap.parse_args()
    out = Path(a.out)
    scanner, acquisition = renamer(a.scanner_ids)
    repro = read_qsmci(a.qsmci, a.ref, 'results/repro.json')
    index = read_qsmci(a.qsmci, a.ref, 'results/index.json')
    problems, notes = [], []

    # acquisitions: the published payload's, renamed
    acq = pd.DataFrame([dict(acquisition=acquisition(k), subject=v['subject'], scanner=scanner(v['scanner']),
                             protocol=v['protocol'], run=v['run']) for k, v in repro['acquisitions'].items()])
    acq = acq.sort_values('acquisition').reset_index(drop=True)
    # scanner order within each subject, as QSM-CI pairs them (registry order: the reference-site session first,
    # then by session); inter-scanner pairs put the earlier scanner as x, so this fixes the direction of a
    ahead = {}
    for pr in repro['pairs']:
        if pr['class'] == 'inter_scanner':
            x, y = (repro['acquisitions'][pr[k]] for k in ('a', 'b'))
            ahead.setdefault(x['subject'], {}).setdefault(scanner(x['scanner']), set()).add(scanner(y['scanner']))
    rank = {(subj, sc): 1 + sum(sc in later for later in d.values()) for subj, d in ahead.items() for sc in
            set(d) | set().union(*d.values())}
    acq['scanner_order'] = [rank.get((r.subject, r.scanner), 1) for r in acq.itertuples()]
    bad = acq[~acq.scanner.str.match(r'^site\d-')]
    if len(bad):
        problems.append(f'scanner ids not mapped to site<N>-<model>: {sorted(bad.scanner.unique())}')
    want_acq, want_pipes = set(acq.acquisition), list(repro['pipelines'])

    # regional values
    rois = json.loads(Path(a.rois).read_text())
    labels = rois['roi_labels']
    rows, skipped = [], Counter()
    for run in rois['runs'].values():
        if not run['acq'].startswith(TRAVELING_HEADS):
            continue
        ac = acquisition(run['acq'])
        if run['pipeline'] not in repro['pipelines'] or ac not in want_acq:
            skipped[run['pipeline']] += 1
            continue
        st = run.get('roi_stats') or {}
        if set(st) != set(run['rois']):
            problems.append(f'{run["pipeline"]} {ac}: roi_stats and rois labels differ')
        for lab in sorted(run['rois'], key=int):
            s = st.get(lab, {})
            rows.append((run['pipeline'], ac, int(lab), labels[lab], s.get('n'), s.get('mean', run['rois'][lab]),
                         s.get('median'), s.get('std'), run.get('ref_mean')))
    reg = pd.DataFrame(rows, columns=ROI_COLUMNS).sort_values(['pipeline', 'acquisition', 'label'])
    if skipped:
        notes.append(f'{sum(skipped.values())} traveling-head runs of {len(skipped)} pipelines not in the published '
                     f'payload left out (e.g. {", ".join(sorted(skipped)[:5])})')
    num = reg[['n_voxels', 'mean_ppm', 'median_ppm', 'std_ppm', 'brain_mean_ppm']]
    nonfinite = (~np.isfinite(num.apply(pd.to_numeric, errors='coerce').astype(float))).sum()
    if nonfinite[['n_voxels', 'mean_ppm', 'brain_mean_ppm']].any():
        problems.append(f'non-finite or missing values: {nonfinite[nonfinite > 0].to_dict()}')
    elif nonfinite.any():
        notes.append(f'non-finite values (unused columns, diverged maps): {nonfinite[nonfinite > 0].to_dict()}')
    big = reg.mean_ppm.abs().groupby([reg.pipeline, reg.acquisition]).max()
    big = big[big > 10].reset_index()
    if len(big):
        where = big.acquisition.str.split('-').str[1:4].str.join('-').value_counts()
        notes.append(f'{len(big)} maps diverged (a region > 10 ppm from the brain mean; emtp leaves them out) in '
                     f'{big.pipeline.nunique()} pipelines; by scanner and protocol: {where.to_dict()}')
    dup = reg.duplicated(['pipeline', 'acquisition', 'label']).sum()
    if dup:
        problems.append(f'{dup} duplicated pipeline/acquisition/label rows')

    # coverage: every published pipeline x acquisition should have a reconstruction with regions
    have = reg[['pipeline', 'acquisition']].drop_duplicates()
    per_pipe = have.groupby('pipeline').size().reindex(want_pipes, fill_value=0)
    missing_pipes = per_pipe[per_pipe == 0].index.tolist()
    if missing_pipes:
        problems.append(f'{len(missing_pipes)} published pipelines with no regional values: {missing_pipes[:10]}')
    short = per_pipe[(per_pipe > 0) & (per_pipe < len(want_acq))]
    if len(short):
        notes.append(f'{len(short)} pipelines lack some acquisitions (no recon or no region >= 90% covered): '
                     + ', '.join(f'{p} {n}/{len(want_acq)}' for p, n in short.sort_values().head(15).items())
                     + (' ...' if len(short) > 15 else ''))
    missing_acq = sorted(want_acq - set(have.acquisition))
    if missing_acq:
        problems.append(f'acquisitions with no regional values at all: {missing_acq}')
    # QSM-CI's own no-recon count, for comparison
    s = repro.get('summary', {})
    if 'n_no_recon' in s:
        nr = dict(zip(repro['pipelines'], s['n_no_recon']))
        notes.append(f'QSM-CI summary: {sum(v > 0 for v in nr.values() if v == v)} pipelines with n_no_recon > 0')

    # pair plan: emtp.comparison_plan must give QSM-CI's pairs, in QSM-CI's direction
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
    import emtp
    ours = set(emtp.comparison_plan(acq))
    theirs = {(pr['class'], acquisition(pr['a']), acquisition(pr['b'])) for pr in repro['pairs']}
    if ours != theirs:
        problems.append(f'pair plan differs from QSM-CI: {len(theirs - ours)} QSM-CI pairs missing here '
                        f'(e.g. {sorted(theirs - ours)[:2]}), {len(ours - theirs)} extra')

    # pipelines
    sim = sim_scores(index)
    prow = []
    for p in sorted(want_pipes):
        n, fm, bfr, dip = steps(p)
        m = sim.get(p, {})
        prow.append(dict(pipeline=p, n_steps=n, field_mapping=fm, background_removal=bfr, dipole_inversion=dip,
                         default_settings=not any('.' in x for x in p.split('+')),
                         challenge2_xsim_full_pipeline=m.get('xsim'), challenge2_dgm_linearity=m.get('dgm_linearity')))
    pipes = pd.DataFrame(prow)
    unscored = pipes[pipes.challenge2_xsim_full_pipeline.isna()]
    notes.append(f'{len(pipes) - len(unscored)} of {len(pipes)} pipelines have a full-pipeline simulation score; '
                 f'unscored: {int((~unscored.default_settings).sum())} with a variant step, '
                 f'{int(unscored.default_settings.sum())} default ({", ".join(unscored[unscored.default_settings].pipeline)})')

    out.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out / 'regional_chi.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    acq.to_csv(out / 'acquisitions.csv', index=False)
    pipes.to_csv(out / 'pipelines.csv', index=False)
    print(f'{len(reg)} region rows, {reg.pipeline.nunique()} pipelines x {reg.acquisition.nunique()} acquisitions '
          f'({len(have)} maps); {len(acq)} acquisitions ({acq.subject.nunique()} subjects, {acq.scanner.nunique()} '
          f'scanners); labels per map {have.merge(reg).groupby(["pipeline", "acquisition"]).size().describe()[["min", "50%", "max"]].to_dict()}')
    for n_ in notes:
        print('note:', n_)
    for p_ in problems:
        print('PROBLEM:', p_)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
