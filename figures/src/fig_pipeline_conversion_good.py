"""#6: one conversion factor per pipeline, on good pipelines (top 10% / 25% by simulation XSIM, D9) and on all,
against the scanner and rescan levels. Data from analysis/followup.py (results/followup/pipeline_conversion_good_*)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
from followup_plots import DATA_LABEL, GREY, INK, INK2, PURPLE, boxes, save  # noqa: E402

RES = ROOT / 'results/followup'
SETS = {'good_top10': 'Top 10 %', 'good_top25': 'Top 25 %', 'all': 'All'}
CONV = {'none': 'no conversion', 'in_sample': 'one factor per pipeline, in-sample',
        'from_reference_bridge': 'one factor per pipeline, learned on the reference bridge'}
COLORS = {CONV['none']: GREY, CONV['in_sample']: '#b3a9dc', CONV['from_reference_bridge']: PURPLE}


def make():
    sd = pd.read_csv(RES / 'pipeline_conversion_good_sd.csv')
    pp = pd.read_csv(RES / 'pipeline_conversion_good_per_pipeline.csv.gz')
    summ = json.loads((RES / 'followup.json').read_text())['issue_6_pipeline_conversion_good']
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.4), sharey='row')
    for j, region in enumerate(['Caudate', 'Pallidum']):
        for i, (d, col, ylab, what) in enumerate([
                (sd, 'pipeline_sd_ppb', 'SD across pipelines (ppb)', 'one value per scan condition'),
                (pp, 'rms_from_consensus_ppb', 'distance from consensus (ppb, RMS)', 'one value per pipeline')]):
            ax = axs[i, j]
            groups = {}
            for s, lab in SETS.items():
                n = summ[s]['n_pipelines']
                groups[f'{lab}\n({n} pipelines)'] = {
                    CONV[c]: d[(d.pipelines == s) & (d.region == region) & (d.conversion == c)][col].values for c in CONV}
            boxes(groups, ylab, ax=ax, colors=COLORS, faint=set(), legend=False, fmt='{:.0f}',
                  ylim=(0, 49), cap=45, title=f'{region}: {what}')
            lv = summ['good_top10'][region]
            for key, lab, ls, va in (('scanner_sd_ppb', 'scanner', '--', 'bottom'), ('rescan_sd_ppb', 'rescan (floor)', ':', 'top')):
                ax.axhline(lv[key], color=INK2, lw=1, ls=ls)
                ax.annotate(f'{lab} {lv[key]:g}', (ax.get_xlim()[1], lv[key]), xytext=(5, 0), textcoords='offset points',
                            ha='left', va=va, fontsize=8, color=INK2, annotation_clip=False)
            ax.title.set_y(1.06)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor=c, alpha=0.5, label=l) for l, c in COLORS.items()], frameon=False,
               fontsize=9, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'Pipeline conversion (#6): one factor per pipeline, good pipelines vs all, {DATA_LABEL}',
                 x=0.06, ha='left', fontsize=12, color=INK)
    fig.text(0.06, -0.03, 'Good pipelines = top fraction by simulation XSIM. Consensus = median over the set\'s pipelines; '
             'factor a, b per pipeline fitted on regional χ against it. Evaluated on all scans except the reference '
             'bridge (site 1 Achieva).\nScan condition = subject, scanner and protocol (runs averaged). Scanner and rescan '
             'levels: SD of the same region for the top 10 % pipelines. Box 25–75 %, whiskers 5–95 %, median printed.',
             fontsize=8.5, color=INK2, va='top')
    fig.subplots_adjust(hspace=0.42, wspace=0.28)
    return fig


if __name__ == '__main__':
    save(make(), 'pipeline_conversion_good')
