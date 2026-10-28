"""Figure 4: sources of variability in mean caudate and pallidum susceptibility (SD, ppb), all pipelines, by protocol."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
import emtp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'fig4_sources_of_variability.png'

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = [('Bridge', '#2a78d6', 'Bridge protocol'), ('Local', '#eb6834', 'Local protocols')]
PROTOCOL_COL = '#8a8984'
LABELS = {
    'Rescan': ('Rescan', 'same session, different run'),
    'Protocol': ('Protocol', 'same scanner, local vs bridge'),
    'Scanner': ('Scanner', 'same protocol, different scanner'),
    'Pipeline': ('Pipeline', 'same scan, different pipeline'),
    'Between subjects': ('Between subjects', 'same pipeline and scanner'),
}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})


def make(reg=None, regions=('Caudate', 'Pallidum')):
    """The figure; `reg` is emtp.regions() (loaded if not given)."""
    reg = emtp.regions() if reg is None else reg
    T = {R: emtp.spread(reg, R).rename(columns=str.title) for R in regions}
    rows = list(LABELS)
    T = {R: t.rename(index=dict(zip(['rescan', 'protocol', 'scanner', 'pipeline', 'between_subjects'], rows))) for R, t in T.items()}
    xmax = max(t.max().max() for t in T.values()) * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor='white', sharey=True, gridspec_kw={'wspace': 0.08})
    H, GAP = 0.24, 0.04
    for ax, (R, t) in zip(axes, T.items()):
        for i, row in enumerate(rows):
            present = [('Both', PROTOCOL_COL)] if row == 'Protocol' else [(k, c) for k, c, _ in SERIES]
            n = len(present)
            for j, (k, col) in enumerate(present):
                y = i + (j - (n - 1) / 2) * (H + GAP)
                v = t.loc[row, k]
                ax.barh(y, v, height=H, color=col)
                ax.annotate(f'{v:.1f}', (v, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=9.5, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_title(R, fontsize=11.5, fontweight='bold', color=INK, loc='left')
        ax.set_xlabel(f'SD of mean {R.lower()} susceptibility (ppb)', color=INK)
        ax.xaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=8)
    ax = axes[0]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{LABELS[r][0]}\n' for r in rows], fontsize=10.5, fontweight='bold')
    for i, rr in enumerate(rows):
        ax.annotate(LABELS[rr][1], (0, i), xycoords=('axes fraction', 'data'), xytext=(-8, -7), textcoords='offset points',
                    ha='right', va='center', fontsize=8.8, color=INK2)
    ax.invert_yaxis()
    fig.legend(handles=[Patch(color=c, label=lab) for _, c, lab in SERIES], frameon=False, fontsize=9.5,
               loc='lower right', bbox_to_anchor=(0.9, 0.93), ncol=2, handlelength=1.2)
    return fig


if __name__ == '__main__':
    make().savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
