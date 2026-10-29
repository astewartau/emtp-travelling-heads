"""Image-quality indicators of motion or field drift during a run -> data/image_quality.csv.

From the QSM-CI input folders (magnitude.nii.gz, mask.nii.gz), first echo:
  ghost_ratio   mean signal in the air outside the head (head = magnitude above 10% of its 99th
                percentile, holes filled, dilated 8 voxels; 6-voxel border excluded) / mean brain signal.
                Motion or drift during a 3D GRE run spreads signal along the phase-encoding axes as ghosts.
  sharpness     mean gradient magnitude inside the brain mask / mean brain signal (per mm). Motion
                during a run blurs edges.
Values are only compared within a session and protocol (same matrix and voxel size).
  python scripts/image_quality.py inputs_map.csv data/image_quality.csv
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def measures(folder):
    folder = Path(folder)
    img = nib.load(str(folder / 'magnitude.nii.gz'))
    mag = np.asanyarray(img.dataobj)[..., 0].astype(np.float64)
    brain = nib.load(str(folder / 'mask.nii.gz')).get_fdata() > 0
    vox = nib.affines.voxel_sizes(img.affine)
    head = ndi.binary_fill_holes(ndi.binary_closing(mag > 0.1 * np.percentile(mag, 99), iterations=3))
    head = ndi.binary_dilation(head, iterations=8)
    border = np.zeros_like(head)
    border[6:-6, 6:-6, 6:-6] = True
    air = ~head & border
    s = mag[brain].mean()
    grad = np.sqrt(sum((g / v) ** 2 for g, v in zip(np.gradient(mag), vox)))
    return {'ghost_ratio': mag[air].mean() / s, 'sharpness_per_mm': grad[brain].mean() / s,
            'n_air_voxels': int(air.sum())}


if __name__ == '__main__':
    imap = pd.read_csv(sys.argv[1])
    rows = []
    for r in imap.itertuples():
        rows.append({'acquisition': r.acquisition, **{k: round(float(v), 6) for k, v in measures(r.path).items()}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
