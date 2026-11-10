"""Head pose per acquisition from rigid registration of the first-echo magnitude -> data/head_pose.csv.

Two references per acquisition (both the bridge run 1 scan):
  session    the same session: motion between runs (no repositioning within a session)
  subject    the subject's reference-site session: repositioning between sessions
For each, the rotation angle and translation of the rigid transform, and the B0 direction expressed in
the reference head frame (b0_x, b0_y, b0_z; unit vector, = (0, 0, 1) for the reference itself), so
the change in head orientation relative to B0 between any two acquisitions is arccos(b0_i . b0_j).
Needs the private session map (see extract_protocols.py).
  python scripts/head_pose.py session_map.csv data/head_pose.csv [n_threads]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ECHO1 = re.compile(r"_e1\.nii\.gz$")


def echo1(folder):
    return next(p for p in sorted(Path(folder).glob('*.nii.gz')) if ECHO1.search(p.name))


def load(path):
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    return sitk.RescaleIntensity(img, 0, 1)


def register(fixed, moving):
    """Rigid (Euler) registration, Mattes mutual information, three resolution levels.
    Returns the transform mapping fixed physical points to moving physical points."""
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.05, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(init, inPlace=False)
    t = reg.Execute(fixed, moving)
    return sitk.Euler3DTransform(t.GetNthTransform(0) if hasattr(t, 'GetNthTransform') else t), reg.GetMetricValue()


def pose(t):
    R = np.array(t.GetMatrix()).reshape(3, 3)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    b0 = R.T @ np.array([0, 0, 1.0])  # scanner z (B0) of the moving scan, in the reference head frame
    return angle, np.linalg.norm(t.GetTranslation()), b0


if __name__ == '__main__':
    smap = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2])
    if len(sys.argv) > 3:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(sys.argv[3]))
    parts = smap.acquisition.str.extract(r'^(s\d+)-(site\d-[a-z0-9]+)-(bridge|local)-(run\d)$')
    smap[['subject', 'scanner', 'protocol', 'run']] = parts
    ref_session = {s: g.scanner[g.scanner.str.startswith('site1-')].iloc[0] for s, g in smap.groupby('subject')}
    path = dict(zip(smap.acquisition, smap.path))
    cache = {}
    get = lambda a: cache.setdefault(a, load(echo1(path[a])))
    rows = []
    for r in smap.itertuples():
        row = {'acquisition': r.acquisition}
        for lab, sc in (('session', r.scanner), ('subject', ref_session[r.subject])):
            ref = f'{r.subject}-{sc}-bridge-run1'
            if ref == r.acquisition:
                angle, shift, b0, metric = 0.0, 0.0, np.array([0, 0, 1.0]), np.nan
            else:
                t, metric = register(get(ref), get(r.acquisition))
                angle, shift, b0 = pose(t)
            row.update({f'{lab}_reference': ref, f'{lab}_rotation_deg': round(angle, 3),
                        f'{lab}_translation_mm': round(shift, 3), f'{lab}_metric': round(metric, 4),
                        **{f'{lab}_b0_{k}': round(v, 5) for k, v in zip('xyz', b0)}})
        rows.append(row)
        print(r.acquisition, row['session_rotation_deg'], row['subject_rotation_deg'], flush=True)
    pd.DataFrame(rows).sort_values('acquisition').to_csv(out, index=False)
