"""Figure 3: (A) orthogonal-regression example, (B) per-pipeline |a - 1| by comparison and protocol (all pipelines)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
import emtp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'fig3_agreement.png'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BRIDGE, LOCAL = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})



def make(reg=None, fits=None):
    """The figure; `reg` is emtp.regions() and `fits` emtp.pair_fits(reg) (computed if not given)."""
    reg = emtp.regions() if reg is None else reg
    fits = emtp.pair_fits(reg) if fits is None else fits
    pp = emtp.per_pipeline(fits)

    fig = plt.figure(figsize=(9.2, 4.3), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.38)

    # A: example fits for one representative pipeline (per-hemisphere regions present in both scans)
    PIPE, REF = 'romeo-qsmrs+pdf-qsmrs+whqsm-qsmrs', 's005002-site1-achieva-bridge-run1'
    ax = fig.add_subplot(gs[0])
    g = reg[reg.pipeline == PIPE]
    vals = {a: dict(zip(h.label, h.mean_ppm)) for a, h in g.groupby('acquisition')}
    lim = (-70, 190)
    ax.plot(lim, lim, color=GRID, lw=1.2, zorder=0)
    for acq, lab, col, mk in [('s005002-site1-achieva-bridge-run2', 'Rescan, same scanner', INK2, 'o'),
                              ('s005002-site5-prisma-bridge-run1', 'Other scanner (Siemens Prisma)', BRIDGE, 's')]:
        k = sorted(set(vals[REF]) & set(vals[acq]))
        x = 1000 * np.array([vals[REF][r] for r in k]); y = 1000 * np.array([vals[acq][r] for r in k])
        f = emtp.fit(x, y)
        ax.scatter(x, y, s=30, marker=mk, color=col, edgecolor='white', lw=0.8, zorder=3,
                   label=f'{lab}: slope a = {f["slope"]:.2f}')
        xx = np.array(lim); ax.plot(xx, f['slope'] * xx + f['intercept_ppm'], color=col, lw=1.6, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Reference scan: regional χ (ppb)'); ax.set_ylabel('Comparison scan: regional χ (ppb)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(-0.05, -0.2), handletextpad=0.3)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.03, 0.97, 'SynthSeg regions (L/R)\none subject, bridge protocol\nROMEO + PDF + WH-QSM', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color=INK2)

    # B: per-pipeline median |a - 1|
    ax = fig.add_subplot(gs[1])
    cols = [('test_retest', 'bridge'), ('test_retest', 'local'), ('inter_scanner', 'bridge'), ('inter_scanner', 'local')]
    pos, CAP = [0.8, 1.2, 2.3, 2.7], 25
    for (c, prot), p in zip(cols, pos):
        d = 100 * pp[(c, prot)].dropna().values
        col = BRIDGE if prot == 'bridge' else LOCAL
        q5, q25, q50, q75, q95 = np.percentile(d, [5, 25, 50, 75, 95])
        ax.add_patch(plt.Rectangle((p - 0.15, q25), 0.3, q75 - q25, facecolor=col, alpha=0.22, edgecolor=col, lw=1.2))
        ax.plot([p - 0.15, p + 0.15], [q50, q50], color=col, lw=2.4, solid_capstyle='butt')
        ax.plot([p, p], [q5, q25], color=col, lw=1.2); ax.plot([p, p], [q75, min(q95, CAP)], color=col, lw=1.2)
        rng = np.random.default_rng(0)
        ax.scatter(p + rng.uniform(-0.11, 0.11, len(d)), np.clip(d, None, CAP), s=3, color=col, alpha=0.25, lw=0, zorder=1)
        ax.annotate(f'{q50:.1f}', (p + 0.16, q50), fontsize=8.3, color=INK, va='center', bbox=dict(fc='white', ec='none', pad=0.6))
        n_over = (d > CAP).sum()
        if n_over:
            ax.annotate(f'+{n_over}', (p, CAP), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks([1, 2.5])
    ax.set_xticklabels(['Test–retest\n(same scanner)', 'Inter-scanner, same subject\n(bridge vs bridge;\nlocal vs local)'])
    ax.set_xlim(0.4, 3.1); ax.set_ylim(0, CAP + 3)
    ax.text(3.08, CAP + 2.2, f'(+n: pipelines above {CAP}%)', fontsize=7.5, color=INK2, ha='right')
    ax.set_ylabel('Slope deviation |a − 1| (%)\nper pipeline, median over scan pairs')
    ax.yaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.scatter([], [], marker='s', s=40, color=BRIDGE, alpha=0.6, label='Bridge protocol')
    ax.scatter([], [], marker='s', s=40, color=LOCAL, alpha=0.6, label='Local protocols')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', borderaxespad=0)

    for a, l in zip(fig.axes[:2], 'AB'):
        a.text(-0.02, 1.06, l, transform=a.transAxes, fontsize=14, fontweight='bold', color=INK, ha='right')
    return fig


if __name__ == '__main__':
    make().savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
