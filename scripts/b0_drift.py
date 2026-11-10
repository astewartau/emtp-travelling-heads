"""Field drift between runs of the same protocol within a session -> data/b0_drift.csv.

Reads the QSM-CI input folders (magnitude.nii.gz, phase.nii.gz in radians, mask.nii.gz, params.json).
For each acquisition, the frequency difference to the same session's run 3 of the same protocol, per voxel
from the multi-echo phase difference: dphi(TE) = angle(z_run * conj(z_run3)), unwrapped along TE and
fitted as 2*pi*df*TE + phi0. No registration: within a session there is no repositioning (motion
~0.5 mm, data/head_pose.csv). Over the voxels inside both brain masks: median df, the linear gradient
(plane fit, Hz/cm along the scanner axes) and the residual SD.
  python scripts/b0_drift.py inputs_map.csv data/b0_drift.csv
inputs_map.csv: acquisition, path (the QSM-CI input folder of that acquisition).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def load(folder):
    folder = Path(folder)
    te = np.array(json.loads((folder / 'params.json').read_text())['TE'])
    mag = nib.load(str(folder / 'magnitude.nii.gz'))
    z = mag.get_fdata() * np.exp(1j * nib.load(str(folder / 'phase.nii.gz')).get_fdata())
    return te, z, nib.load(str(folder / 'mask.nii.gz')).get_fdata() > 0, mag.affine


def drift(te, za, zb, mask, affine):
    dphi = np.unwrap(np.angle(za[mask] * np.conj(zb[mask])), axis=1)   # voxels x echoes
    A = np.stack([2 * np.pi * te, np.ones_like(te)], 1)
    df = np.linalg.lstsq(A, dphi.T, rcond=None)[0][0]                  # Hz per voxel
    xyz = (affine[:3, :3] @ np.argwhere(mask).T).T / 10.0               # cm along scanner axes
    P = np.column_stack([xyz, np.ones(len(xyz))])
    coef = np.linalg.lstsq(P, df, rcond=None)[0]
    return {'median_hz': np.median(df), 'gradient_x_hz_per_cm': coef[0], 'gradient_y_hz_per_cm': coef[1],
            'gradient_z_hz_per_cm': coef[2], 'residual_sd_hz': np.std(df - P @ coef), 'n_voxels': int(mask.sum())}


if __name__ == '__main__':
    imap = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2])
    imap[['session', 'protocol', 'run']] = imap.acquisition.str.extract(r'^(s\d+-site\d-[a-z0-9]+)-(bridge|local)-(run\d)$')
    path = dict(zip(imap.acquisition, imap.path))
    rows = []
    for (session, prot), g in imap.groupby(['session', 'protocol']):
        ref = f'{session}-{prot}-run3'
        te, zr, mr, aff = load(path[ref])
        for a in sorted(set(g.acquisition) - {ref}):
            te_a, za, ma, _ = load(path[a])
            if za.shape != zr.shape or not np.allclose(te_a, te, atol=1e-6):
                print('skip', a, flush=True)
                continue
            d = drift(te, za, zr, ma & mr, aff)
            rows.append({'acquisition': a, 'reference': ref, **{k: round(float(v), 4) for k, v in d.items()}})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
