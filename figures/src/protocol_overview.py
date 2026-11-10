"""#15: acquisition protocols of all 17 sessions, one row per scanner and protocol (data/protocols.csv), with the
local protocols counted as "different" from the bridge for #14 (emtp.DIFFERENT_LOCALS, decision D8)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
import emtp  # noqa: E402
from followup_plots import BRIDGE, GREY, GRID, INK, INK2, LOCAL, save  # noqa: E402

NAME = {'site1-achieva': 'S1 Philips Achieva', 'site1-elition': 'S1 Philips Elition X', 'site2-prisma': 'S2 Siemens Prisma',
        'site2-cimax': 'S2 Siemens Cima.X', 'site2-skyra': 'S2 Siemens Skyra', 'site3-mr750w': 'S3 GE MR750w',
        'site3-prisma': 'S3 Siemens Prisma', 'site4-skyra': 'S4 Siemens Skyra', 'site5-mr750': 'S5 GE MR750',
        'site5-prisma': 'S5 Siemens Prisma', 'site6-ingenia': 'S6 Philips Ingenia'}
WHY = {'site2-prisma': 'single echo, 0.8 mm (½ voxel vol.)', 'site2-cimax': 'single echo, 0.8 mm (½ voxel vol.)',
       'site2-skyra': 'single echo, 0.8 mm (½ voxel vol.)', 'site4-skyra': '2 mm slices (2× voxel vol.)',
       'site5-prisma': '0.75 mm in-plane (0.56× vol.), 8 echoes', 'site5-mr750': '8 echoes to 36 ms, 0.94 mm'}
SEQ = ['n_echoes', 'te_first_ms', 'te_last_ms', 'delta_te_ms', 'tr_ms', 'flip_angle_deg', 'voxel_x_mm', 'voxel_y_mm',
       'voxel_z_mm', 'matrix', 'acquisition_duration_s']
HEAD = ['Echoes', 'TE (ms)', 'ΔTE', 'TR', 'Flip', 'Voxel (mm)', 'Vol. (mm³)', 'FOV (mm)', 'Min']
KEY = [['n_echoes'], ['te_first_ms', 'te_last_ms'], ['delta_te_ms'], ['tr_ms'], ['flip_angle_deg'],
       ['voxel_x_mm', 'voxel_y_mm', 'voxel_z_mm'], ['voxel_mm3'], ['fov_x_mm', 'fov_y_mm', 'fov_z_mm'], ['acquisition_duration_s']]
CW = [0.75, 1.15, 0.55, 0.65, 0.5, 1.25, 1.1, 1.3, 0.55]
X0 = 3.4


def overview():
    """One row per scanner and protocol; runs (and subjects) share every sequence parameter."""
    p = pd.read_csv(emtp.DATA / 'protocols.csv')
    parts = p.acquisition.str.split('-')  # s<subject>-site<N>-<model>-<protocol>-run<N>
    p['subject'], p['scanner'], p['protocol'] = parts.str[0], parts.str[1] + '-' + parts.str[2], parts.str[3]
    g = p.groupby(['scanner', 'protocol'])
    assert (g[SEQ].nunique(dropna=False) <= 1).all().all(), 'runs differ within a scanner and protocol'
    ov = g[SEQ].first()
    ov['n_subjects'] = g.subject.nunique()
    m = ov.matrix.str.split('x', expand=True).astype(int)
    for i, ax in enumerate('xyz'):
        ov[f'fov_{ax}_mm'] = (m[i] * ov[f'voxel_{ax}_mm']).round(0)
    ov['voxel_mm3'] = (ov.voxel_x_mm * ov.voxel_y_mm * ov.voxel_z_mm).round(3)
    return ov


def cells(r):
    te = f'{r.te_first_ms:g}–{r.te_last_ms:g}' if r.n_echoes > 1 else f'{r.te_first_ms:g}'
    vox = '×'.join(f'{round(v, 2):g}' for v in (r.voxel_x_mm, r.voxel_y_mm, r.voxel_z_mm))
    return [str(r.n_echoes), te, f'{r.delta_te_ms:g}' if r.n_echoes > 1 else '–', f'{r.tr_ms:g}',
            f'{r.flip_angle_deg:g}', vox, f'{r.voxel_mm3:.2f}', '×'.join(f'{v:.0f}' for v in (r.fov_x_mm, r.fov_y_mm, r.fov_z_mm)),
            f'{r.acquisition_duration_s / 60:.1f}' if pd.notna(r.acquisition_duration_s) else '·']


def differs(r, b, cols):
    return any(pd.notna(r[c]) and pd.notna(b[c]) and abs(float(r[c]) - float(b[c])) > 1e-6 * max(1, abs(float(b[c])))
               for c in cols)


def make():
    ov = overview()
    nrows = 2 * len(NAME)
    fig, ax = plt.subplots(figsize=(14.5, 9.6)); ax.axis('off')
    right = X0 + sum(CW)
    ax.set_xlim(0, right + 3.0); ax.set_ylim(nrows + 2.6, -1.2)
    xs = np.cumsum([X0] + CW[:-1])
    for x, h in zip(xs, HEAD):
        ax.text(x + 0.05, -0.3, h, fontsize=9.5, weight='bold', color=INK, va='center')
    ax.text(right + 0.1, -0.3, 'For #14', fontsize=9.5, weight='bold', color=INK, va='center')
    for i, sc in enumerate(NAME):
        b = ov.loc[(sc, 'bridge')]
        for j, proto in enumerate(('bridge', 'local')):
            r, y = ov.loc[(sc, proto)], 2 * i + j
            if proto == 'bridge':
                n = int(r.n_subjects)
                ax.text(0.05, y + 1.0, NAME[sc], fontsize=10, color=INK, va='center', weight='bold')
                ax.text(0.05, y + 1.55, f"{n} subject{'s' if n > 1 else ''}", fontsize=8, color=INK2, va='center')
            ax.text(2.3, y + 0.5, proto, fontsize=9.5, color=BRIDGE if proto == 'bridge' else LOCAL, va='center', weight='bold')
            for x, w, txt, k in zip(xs, CW, cells(r), KEY):
                if proto == 'local' and differs(r, b, k):
                    ax.add_patch(plt.Rectangle((x, y + 0.08), w - 0.06, 0.84, color=LOCAL, alpha=0.16, lw=0))
                ax.text(x + 0.05, y + 0.5, txt, fontsize=9.5, color=GREY if txt == '·' else INK, va='center')
            if proto == 'local':
                cl = ('different' if sc in emtp.DIFFERENT_LOCALS else
                      'borderline (sensitivity check)' if sc in emtp.BORDERLINE_LOCALS else 'similar')
                ax.text(right + 0.1, y + 0.5, cl + (f': {WHY[sc]}' if sc in WHY else ''), fontsize=9, va='center',
                        color=INK if cl == 'different' else INK2 if cl.startswith('border') else GREY,
                        weight='bold' if cl == 'different' else 'normal')
        ax.plot([0, right + 2.6], [2 * i + 2] * 2, color=GRID, lw=0.8)
    ax.set_title('Acquisition protocols, all 17 sessions (one row per scanner and protocol)', loc='left', fontsize=12, color=INK)
    ax.text(0.05, nrows + 0.5, 'Shaded: differs from the bridge on the same scanner. Voxel = reconstructed voxel. '
            'Min = acquisition time (· = not in the sidecar). All slabs axial; runs 1–3 identical in every session.\n'
            'A local protocol counts as "different" if it is single-echo or its voxel volume is ≤ 0.67× or ≥ 1.5× the '
            "bridge's; TR, flip angle, FOV, acceleration and bandwidth are not counted.", fontsize=8.5, color=INK2, va='top')
    return fig


if __name__ == '__main__':
    save(make(), 'protocol_overview')
