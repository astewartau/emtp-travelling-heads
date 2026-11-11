"""Plots for followup.ipynb. Each function draws one kind of chart from quantities computed in the notebook;
nothing is computed here except the percentiles a box shows.

Colours follow the series labels: a label containing 'bridge' is blue, 'local' orange, anything else grey unless
`colors` says otherwise. A series whose label says 'removed', 'without' or 'best case' is drawn faint (dashed box)
or, for bars, hatched.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parents[1] / 'followup'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BRIDGE, LOCAL, GREY, PURPLE = '#2a78d6', '#eb6834', '#8a8984', '#6b5bb3'
FAINT_WORDS = ('remov', 'without', 'best case')
SCANNER = {'site1-achieva': 'S1 Achieva', 'site1-elition': 'S1 Elition X', 'site2-prisma': 'S2 Prisma',
           'site2-cimax': 'S2 Cima.X', 'site2-skyra': 'S2 Skyra', 'site3-mr750w': 'S3 MR750w', 'site3-prisma': 'S3 Prisma',
           'site4-skyra': 'S4 Skyra', 'site5-mr750': 'S5 MR750', 'site5-prisma': 'S5 Prisma', 'site6-ingenia': 'S6 Ingenia'}
DATA_LABEL = 'v11 data'   # figure titles: which QSM-CI run the figures come from
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})


def colour_of(label, colors=None):
    if colors and label in colors:
        return colors[label]
    low = label.lower()
    return BRIDGE if 'bridge' in low else LOCAL if 'local' in low else GREY


def is_faint(label, faint=None):
    return label in faint if faint is not None else any(w in label.lower() for w in FAINT_WORDS)


def scanner_pair(label):
    """'site1-achieva – site6-ingenia' -> 'S1 Achieva –\\nS6 Ingenia'"""
    return ' –\n'.join(SCANNER[s] for s in label.split(' – '))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight', facecolor='white')


def _box(ax, x, values, col, faint=False, w=0.32, cap=None, fmt='{:.1f}'):
    """Box = 25th–75th percentile, thick line = median (printed), whiskers = 5th and 95th percentiles,
    +n = values above `cap`."""
    d = np.asarray(values, dtype=float); d = d[~np.isnan(d)]
    q5, q25, q50, q75, q95 = np.percentile(d, [5, 25, 50, 75, 95])
    ax.add_patch(plt.Rectangle((x - w / 2, q25), w, q75 - q25, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=0.10 if faint else 0.22, ls='--' if faint else '-'))
    ax.plot([x - w / 2, x + w / 2], [q50, q50], color=col, lw=2.4, solid_capstyle='butt', ls=(0, (2, 1)) if faint else '-')
    ax.plot([x, x], [q5, q25], color=col, lw=1.2)
    ax.plot([x, x], [q75, q95 if cap is None else min(q95, cap)], color=col, lw=1.2)
    ax.annotate(fmt.format(q50), (x + w / 2 + 0.02, q50), fontsize=8, color=INK, va='center',
                bbox=dict(fc='white', ec='none', pad=0.5))
    if cap is not None and (d > cap).sum():
        ax.annotate(f'+{(d > cap).sum()}', (x, cap), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)


def boxes(groups, ylabel, *, ax=None, title=None, ylim=None, cap=None, colors=None, faint=None, legend=True,
          figsize=None, fmt='{:.1f}', zero_line=False, name=None):
    """Box plots across pipelines. `groups` maps a group label (on the x-axis) to {series label: one value per
    pipeline}; each series is one box. `cap` cuts the axis and counts the pipelines above it (+n)."""
    if ax is None:
        n = sum(len(s) for s in groups.values())
        fig, ax = plt.subplots(figsize=figsize or (max(4.5, 0.62 * n + 2), 4), facecolor='white')
    step, gap, w = 0.62, 0.5, 0.32
    x, ticks, seen = 0.0, [], {}
    for group, series in groups.items():
        first = x
        for label, values in series.items():
            col, fa = colour_of(label, colors), is_faint(label, faint)
            _box(ax, x, values, col, fa, w=w, cap=cap, fmt=fmt)
            seen.setdefault(label, (col, fa))
            x += step
        ticks.append(((first + x - step) / 2, group))
        x += gap
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([g for _, g in ticks])
    ax.set_xlim(-0.5, x - gap - step + 0.75)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zero_line:
        ax.axhline(0, color=INK2, lw=1, ls='--')
    ax.set_ylabel(ylabel); ax.yaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, loc='left', color=INK2)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor=c, alpha=0.25 if f else 0.5, ls='--' if f else '-', label=l)
                           for l, (c, f) in seen.items()], frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    if name:
        save(ax.figure, name)
    return ax


def hbars(rows, xlabel, *, ax=None, title=None, colors=None, legend=True, figsize=(8, 4.8), name=None):
    """Horizontal bars, one row per item: `rows` maps a row label to {series label: value}. Values are printed at the
    bar ends; a series whose label says 'removed'/'removing' is hatched and named next to its value."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    h, seen = 0.26, {}
    for i, series in enumerate(rows.values()):
        n = len(series)
        for j, (label, v) in enumerate(series.items()):
            y = i + (j - (n - 1) / 2) * (h + 0.04)
            col, hatched = colour_of(label, colors), 'remov' in label.lower()
            ax.barh(y, v, height=h, color='white' if hatched else col, edgecolor=col, hatch='////' if hatched else None,
                    lw=1 if hatched else 0)
            ax.annotate(f'{v:.1f}' + (f'  {label}' if hatched else ''), (v, y), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=INK2 if hatched else INK)
            if not hatched:
                seen.setdefault(label, col)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(list(rows)); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.xaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, max(v for s in rows.values() for v in s.values()) * 1.18)
    if title:
        ax.set_title(title, fontsize=11, loc='left', fontweight='bold', color=INK)
    if legend:
        ax.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()], frameon=False, fontsize=8.5, loc='lower right',
                  bbox_to_anchor=(1, 1.0), ncol=len(seen))
    if name:
        save(ax.figure, name)
    return ax


def subject_boxes(values, ylabel, *, ax, ylim=None, title=None, name=None):
    """One box per subject, grouped by scanner. `values` is indexed by scanner, subject and pipeline."""
    x, ticks = 0, []
    for scanner, per_scanner in values.groupby(level=0, sort=True):
        first = x
        for subject, v in per_scanner.groupby(level=1, sort=True):
            _box(ax, x, v.values, GREY, w=0.55, fmt='{:+.1f}'); ticks.append((x, subject)); x += 1
        ax.annotate(SCANNER[scanner], ((first + x - 1) / 2, 0), xycoords=('data', 'axes fraction'), xytext=(0, -42),
                    textcoords='offset points', ha='center', fontsize=9.5, color=INK, fontweight='bold')
        x += 0.8
    ax.axhline(0, color=INK2, lw=1, ls='--')
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([s for _, s in ticks], fontsize=8, rotation=35, ha='right')
    ax.set_xlim(-0.7, x - 0.6)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.yaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, loc='left', color=INK)
    if name:
        save(ax.figure, name)


def run_pairs(r12, r13, r23, *, name=None):
    """One row per session, one panel per protocol: the two pairs with run 1 (dots) and run 2 vs run 3 (diamond).
    Each argument is indexed by subject, scanner and protocol."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor='white')
    for ax, (prot, col) in zip(axs, (('bridge', BRIDGE), ('local', LOCAL))):
        sessions = sorted(r23.xs(prot, level='protocol').index)
        for i, sess in enumerate(sessions):
            k = sess + (prot,)
            ax.plot([r12[k], r13[k], r23[k]], [i] * 3, color=GRID, lw=3, zorder=1)
            ax.scatter([r12[k], r13[k]], [i, i], s=80, color=col, edgecolors='white', linewidths=1.2, zorder=3)
            ax.scatter([r23[k]], [i], s=80, marker='D', facecolor='white', edgecolors=INK, linewidths=1.8, zorder=4)
        ax.set_yticks(range(len(sessions))); ax.set_yticklabels([f'{s} {SCANNER[c]}' for s, c in sessions]); ax.invert_yaxis()
        ax.set_title(prot.capitalize(), fontsize=12); ax.set_xlim(0, 7)
        ax.set_xlabel('Test-retest |a − 1| (%), median over pipelines'); ax.xaxis.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].scatter([], [], s=80, color=BRIDGE, label='pairs with run 1')
    axs[0].scatter([], [], s=80, marker='D', facecolor='white', edgecolors=INK, linewidths=1.8, label='run 2 vs run 3')
    axs[0].legend(frameon=False, fontsize=9, loc='upper right')
    if name:
        save(fig, name)


def example_fit(panels, *, name=None, show=('Left-Pallidum', 'Left-Putamen', 'Left-Caudate', 'Left-Cerebral-WM')):
    """Scatter of regional χ in two scans with the identity and the fitted line. `panels` is a list of
    (title, x label, y label, x, y, (a, b)) with x and y indexed by region."""
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.6), sharex=True, sharey=True, facecolor='white')
    for ax, (title, xl, yl, x, y, (a, b)) in zip(np.atleast_1d(axs), panels):
        lo, hi = min(x.min(), y.min()) - 10, max(x.max(), y.max()) + 10
        ax.plot([lo, hi], [lo, hi], color='#b9b6ae', ls='--', label='identity (a = 1)')
        ax.plot([lo, hi], [a * lo + b, a * hi + b], color=INK, label='orthogonal fit')
        ax.scatter(x, y, s=40, color=BRIDGE, edgecolors='white', zorder=3)
        for region in show:
            if region in x.index:
                ax.annotate(region.replace('Left-', 'L '), (x[region], y[region]), xytext=(5, -10), textcoords='offset points', fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_xlabel(xl); ax.set_ylabel(yl)
    np.atleast_1d(axs)[0].legend(frameon=False)
    if name:
        save(fig, name)


def pair_design(acq, counts, *, scanners=None, name=None):
    """The three kinds of scan pair. `counts` gives the numbers for the titles (from emtp.comparison_plan);
    `scanners` maps scanner id -> label for panel D (default: the six analysed in v10)."""
    scanners = scanners or SCANNER
    light, edge = {'B': '#d6e6f8', 'L': '#fbdccd'}, {'B': BRIDGE, 'L': LOCAL}

    def scan(ax, x, y, lab, w=0.72, h=0.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor=light[lab[0]], edgecolor=edge[lab[0]], lw=1.4, zorder=3))
        ax.text(x, y, lab, ha='center', va='center', fontsize=11, fontweight='bold', color=INK, zorder=4)

    def arc(ax, x0, x1, y, col, up=True, height=None):
        w = abs(x1 - x0)
        ax.add_patch(Arc(((x0 + x1) / 2, y), w, 2 * (height if height is not None else 0.42 * w),
                         theta1=0 if up else 180, theta2=180 if up else 360, color=col, lw=2, zorder=2))

    def session(ax, arrow_y=-0.55):
        for i, lab in enumerate(('B1', 'L1', 'B2', 'L2', 'B3', 'L3')):
            scan(ax, i, 0, lab)
        ax.annotate('', (5.6, arrow_y), (-0.4, arrow_y), arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
        ax.text(5.6, arrow_y - 0.15, 'time in the session', ha='right', va='top', fontsize=8.5, color=INK2)

    def clean(ax, xlim, ylim):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')

    fig = plt.figure(figsize=(12.5, 8.6), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.12, height_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    session(ax, arrow_y=-1.6)
    for i, j in ((0, 2), (0, 4), (2, 4)):
        arc(ax, i, j, 0.25, BRIDGE)
    for i, j in ((1, 3), (1, 5), (3, 5)):
        arc(ax, i, j, -0.25, LOCAL, up=False, height=0.28 * (j - i))
    ax.text(5.9, 1.2, 'bridge:\n1–2, 1–3, 2–3', color=BRIDGE, fontsize=9, va='center')
    ax.text(5.9, -1.2, 'local:\n1–2, 1–3, 2–3', color=LOCAL, fontsize=9, va='center')
    clean(ax, (-0.6, 7.4), (-2.1, 2.4))
    ax.set_title(f"A  Test-retest: every two runs, same protocol, same session\n"
                 f"3 pairs per session × {counts['sessions']} sessions = {counts['test_retest']} per protocol", loc='left', fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    session(ax)
    for k in range(3):
        arc(ax, 2 * k, 2 * k + 1, 0.25, PURPLE, height=0.9)
        ax.text(2 * k + 0.5, 1.2, f'B{k + 1}–L{k + 1}', ha='center', fontsize=9, color=PURPLE)
    ax.text(5.9, 0, 'bridge = x\nlocal = y', fontsize=9, color=INK2, va='center')
    clean(ax, (-0.6, 7.4), (-1.9, 2.4))
    ax.set_title(f"B  Inter-protocol: bridge run k vs local run k, same session\n"
                 f"3 pairs per session × {counts['sessions']} sessions = {counts['inter_protocol']}", loc='left', fontsize=10.5)

    ax = fig.add_subplot(gs[1, 0])
    for k in range(3):
        scan(ax, 1.4 * k + 1.6, 2.2, f'B{k + 1}'); scan(ax, 1.4 * k + 1.6, 0, f'B{k + 1}')
    for i in range(3):
        for j in range(3):
            ax.plot([1.4 * i + 1.6, 1.4 * j + 1.6], [1.95, 0.25], color=BRIDGE, lw=1.4, alpha=0.75, zorder=1)
    ax.text(0.9, 2.2, 'Site-1 scanner\n(day 1) = x', ha='right', va='center', fontsize=9.5)
    ax.text(0.9, 0, 'Other scanner\n(another day) = y', ha='right', va='center', fontsize=9.5)
    ax.text(3.0, -0.6, 'Every run × every run = 9 pairs (different days, so\nrun numbers don’t correspond). Bridge shown;\n'
            'local is paired the same way, separately.', ha='center', va='top', fontsize=8.5, color=INK2)
    clean(ax, (-1.9, 6.4), (-1.7, 2.8))
    ax.set_title(f"C  Inter-scanner: same subject and protocol, two scanners\n"
                 f"9 pairs per session pair × {counts['session_pairs']} session pairs = {counts['inter_scanner']} per protocol",
                 loc='left', fontsize=10.5)

    ax = fig.add_subplot(gs[1, 1])
    labels, scanners, subjects = list(scanners.values()), list(scanners), sorted(acq.subject.unique())
    n_multi = int((acq.groupby('subject').scanner.nunique() > 1).sum())
    sessions = acq[['subject', 'scanner']].drop_duplicates()
    for subj, g in sessions.groupby('subject'):
        y, xs = subjects.index(subj), sorted(scanners.index(c) for c in g.scanner)
        for i, a in enumerate(xs):
            for b in xs[i + 1:]:
                arc(ax, a, b, y, BRIDGE, up=False, height=min(0.12 + 0.08 * (b - a), 0.1 + 0.035 * len(scanners) * (b - a) / 6 + 0.05))
        n = len(xs) * (len(xs) - 1) // 2
        ax.text(len(scanners), y, (f'{n} pair' + ('s' if n > 1 else '')) if n else 'site 1 only', fontsize=9, va='center',
                color=INK if n else INK2)
    for r in sessions.itertuples():
        ax.scatter(scanners.index(r.scanner), subjects.index(r.subject), s=110, color=INK, zorder=3)
    for i in range(len(subjects)):
        ax.axhline(i, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(len(scanners))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(subjects))); ax.set_yticklabels(subjects, fontsize=9)
    ax.set_xlim(-0.5, len(scanners) + 1); ax.set_ylim(len(subjects) - 0.4, -0.7)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"D  Which sessions are paired: dot = session, curve = session pair\n"
                 f"{counts['session_pairs']} session pairs from {n_multi} subjects, {counts['scanner_pairs']} distinct scanner pairs",
                 loc='left', fontsize=10.5)
    if name:
        save(fig, name)
