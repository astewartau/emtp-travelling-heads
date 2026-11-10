"""Acquisition parameters per acquisition from the raw BIDS-style sidecars -> data/protocols.csv.

Needs a session map (not in the repo; it holds the raw folder names): CSV with columns
acquisition (e.g. s004001-site1-achieva-bridge-run1) and path (the raw run/protocol folder).
Only sequence parameters are written; no names, dates, institutions or serial numbers.
  python scripts/extract_protocols.py session_map.csv data/protocols.csv
Rows already in the output for acquisitions not in the session map are kept. The folders may hold only the
sidecars and each NIfTI's 352-byte header (<name>.hdr) instead of the images.
"""
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ECHO_RE = re.compile(r"_e(\d+)(?:_(ph|real|imaginary))?\.(?:nii\.gz|hdr)$")
SKIP = ('_GRE_mag', '_GRE_phase')


def magnitude_echoes(folder):
    """echo number -> one image per echo: the magnitude, or the real part where only real/imaginary was saved, or
    the one non-phase image of a single-echo scan (the 4D *_GRE_{mag,phase} stacks are ignored). A file may be
    the NIfTI itself or just its 352-byte header (*.hdr)."""
    files = [p for p in sorted(Path(folder).iterdir()) if p.name.endswith(('.nii.gz', '.hdr'))
             and not any(k in p.name for k in SKIP)]
    by_kind = {}
    for p in files:
        m = ECHO_RE.search(p.name)
        if m:
            by_kind.setdefault(m.group(2), {})[int(m.group(1))] = p
    for kind in (None, 'real'):
        if by_kind.get(kind):
            return by_kind[kind]
    single = [p for p in files if not ECHO_RE.search(p.name) and not p.name.replace('.nii.gz', '').replace('.hdr', '').endswith('_ph')]
    if len(single) == 1:
        return {1: single[0]}
    raise ValueError(f'no magnitude echoes in {folder}')


def header(path):
    if path.name.endswith('.hdr'):
        return nib.Nifti1Header.from_fileobj(open(path, 'rb'))
    return nib.load(str(path)).header


def sidecar(nii):
    return json.loads(nii.with_name(nii.name.replace('.nii.gz', '').replace('.hdr', '') + '.json').read_text())


def describe(folder):
    mags = magnitude_echoes(folder)
    metas = [sidecar(mags[e]) for e in sorted(mags)]
    m = metas[0]
    tes = sorted(1000 * x['EchoTime'] for x in metas)
    hdr = header(mags[min(mags)])
    affine, shape = hdr.get_best_affine(), hdr.get_data_shape()
    vox = np.sqrt((affine[:3, :3] ** 2).sum(0))
    # slab normal (third voxel axis) in scanner coordinates, and its angle to B0 (z)
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    opts = m.get('ScanOptions', '')
    return {
        'manufacturer': m.get('Manufacturer'), 'model': m.get('ManufacturersModelName'),
        'software': m.get('SoftwareVersions'),
        'acquisition_type': m.get('MRAcquisitionType'),
        'n_echoes': len(tes), 'te_first_ms': round(tes[0], 2), 'te_last_ms': round(tes[-1], 2),
        'delta_te_ms': round(float(np.median(np.diff(tes))), 2) if len(tes) > 1 else None,
        'tr_ms': round(1000 * m['RepetitionTime'], 2) if 'RepetitionTime' in m else None,
        'flip_angle_deg': m.get('FlipAngle'),
        'voxel_x_mm': round(vox[0], 3), 'voxel_y_mm': round(vox[1], 3), 'voxel_z_mm': round(vox[2], 3),
        'matrix': 'x'.join(str(s) for s in shape[:3]),
        'pixel_bandwidth_hz': m.get('PixelBandwidth'),
        'parallel_factor_inplane': m.get('ParallelReductionFactorInPlane'),
        'parallel_technique': m.get('ParallelAcquisitionTechnique'),
        'partial_fourier': m.get('PartialFourier'),
        'percent_sampling': m.get('PercentSampling'),
        'scan_options': '|'.join(opts) if isinstance(opts, list) else opts,
        'sequence_variant': '|'.join(m['SequenceVariant']) if isinstance(m.get('SequenceVariant'), list) else m.get('SequenceVariant'),
        'acquisition_duration_s': m.get('AcquisitionDuration'),
        'slab_angle_to_b0_deg': round(float(np.degrees(np.arccos(abs(normal[2])))), 1),
    }


if __name__ == '__main__':
    smap, out = pd.read_csv(sys.argv[1]), Path(sys.argv[2])
    rows = pd.DataFrame([{'acquisition': r.acquisition, **describe(r.path)} for r in smap.itertuples()])
    if out.exists():  # keep the rows of acquisitions not in this session map
        old = pd.read_csv(out)
        rows = pd.concat([old[~old.acquisition.isin(rows.acquisition)], rows], ignore_index=True)[old.columns]
    rows.sort_values('acquisition').to_csv(out, index=False)
    print(f'{len(smap)} acquisitions extracted, {len(rows)} rows -> {out}')
