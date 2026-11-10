"""Drift during a run, estimated from the echo-to-echo image shift -> data/echo_shift.csv.

A field that drifts by D Hz over a 3D GRE acquisition, sampled in sequential phase-encoding order, adds a
phase ramp of 2*pi*D*TE across k-space along the slowest phase-encoding axis, which shifts that echo's
image by D*TE voxels along it. So the shift of each echo relative to the first, against TE, has a slope of
D (Hz) along that axis, and about zero along the others. Shifts are measured on the brain-masked magnitude
by phase cross-correlation (upsampled to 1/100 voxel); the slope is fitted through the origin at echo 1.
Which axis is the slow one, and whether the ordering is sequential, depends on the vendor and protocol, so
all three voxel axes are reported; contrast changes between echoes add the same spurious shift to every run
of a protocol, so runs are compared within a session and protocol.
  python scripts/echo_shift.py inputs_map.csv data/echo_shift.csv [n_processes]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation


def measure(args):
    acq, folder = args
    folder = Path(folder)
    te = np.array(json.loads((folder / 'params.json').read_text())['TE'])
    mag = np.asanyarray(nib.load(str(folder / 'magnitude.nii.gz')).dataobj).astype(np.float32)
    mask = nib.load(str(folder / 'mask.nii.gz')).get_fdata() > 0
    w = ndi.gaussian_filter(ndi.binary_dilation(mask, iterations=3).astype(np.float32), 2)
    ref = mag[..., 0] * w
    shifts = np.zeros((len(te), 3))
    for k in range(1, len(te)):
        # phase_cross_correlation returns the shift that registers moving onto reference
        shifts[k] = -phase_cross_correlation(ref, mag[..., k] * w, upsample_factor=100, normalization=None)[0]
    dte = te - te[0]
    slope = (dte[:, None] * shifts).sum(0) / (dte ** 2).sum()
    row = {'acquisition': acq, 'n_echoes': len(te)}
    for i in range(3):
        row[f'drift_hz_axis{i}'] = round(float(slope[i]), 3)
        row[f'last_echo_shift_vox_axis{i}'] = round(float(shifts[-1, i]), 4)
    print(row, flush=True)
    return row


if __name__ == '__main__':
    imap = pd.read_csv(sys.argv[1])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    with Pool(n) as p:
        rows = p.map(measure, list(zip(imap.acquisition, imap.path)))
    pd.DataFrame(rows).sort_values('acquisition').to_csv(sys.argv[2], index=False)
