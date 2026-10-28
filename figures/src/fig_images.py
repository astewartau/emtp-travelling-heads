"""Figure 2: representative processing stages for one subject on three vendors (bridge protocol, run 1)."""
import SimpleITK as sitk, numpy as np, matplotlib
from scipy import ndimage
matplotlib.use('Agg'); import matplotlib.pyplot as plt

A = '../../data/figure2/align'
IN = '../../data/figure2/inputs'
KEEP = '../../data/figure2/recons'
SUB = 's005002'
ACQS = [('Philips Achieva', 'site1-achieva'), ('GE MR750', 'site5-mr750'), ('Siemens Prisma', 'site5-prisma')]

fixed = sitk.ReadImage(f'{A}/{SUB}/target_mag_e1.nii.gz', sitk.sitkFloat32)
seg = sitk.GetArrayFromImage(sitk.ReadImage(f'{A}/{SUB}/dseg.nii.gz'))
z = int(np.argmax(((seg == 13) | (seg == 52)).sum((1, 2)))) + 2  # through pallidum
head = sitk.GetArrayFromImage(fixed)[z] > np.percentile(sitk.GetArrayFromImage(fixed)[z], 60)
ys, xs = np.where(head); pad = 4
crop = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))


# SynthSeg regions in the subject frame (same for every scanner), FreeSurfer colours, left/right merged
segslice = np.flipud(seg[z][crop])
REGIONS = [('Cortex', (3, 42), (205, 62, 78)), ('White matter', (2, 41), (245, 245, 245)),
           ('Caudate', (11, 50), (122, 186, 220)), ('Putamen', (12, 51), (236, 13, 176)),
           ('Pallidum', (13, 52), (12, 48, 255)), ('Thalamus', (10, 49), (0, 118, 14)),
           ('Other regions', (7, 8, 16, 17, 18, 26, 28, 46, 47, 53, 54, 58, 60), (230, 148, 34))]
overlay = np.zeros(segslice.shape + (4,))
for _, labs, rgb in REGIONS:
    overlay[np.isin(segslice, labs)] = [*(np.array(rgb) / 255), 0.6]
# thin white boundaries between regions: a second cue besides colour (colour-vision deficiency)
group = np.zeros(segslice.shape, int)
for i, (_, labs, _) in enumerate(REGIONS, 1):
    group[np.isin(segslice, labs)] = i
edge = np.zeros_like(group, bool)
edge[:-1] |= group[:-1] != group[1:]; edge[1:] |= group[:-1] != group[1:]
edge[:, :-1] |= group[:, :-1] != group[:, 1:]; edge[:, 1:] |= group[:, :-1] != group[:, 1:]
edge &= group > 0
overlay[edge] = [1, 1, 1, 0.9]


def resample(img, acq, interp=sitk.sitkLinear):
    tfm = sitk.ReadTransform(f'{A}/{acq}.tfm')
    arr = sitk.GetArrayFromImage(sitk.Resample(img, fixed, tfm, interp, 0.0))
    return np.flipud(arr[z][crop])


def echo(path, e):
    im = sitk.ReadImage(path, sitk.sitkFloat32)
    return sitk.Extract(im, list(im.GetSize()[:3]) + [0], [0, 0, 0, e])


rows = []
for name, sc in ACQS:
    acq = f'{SUB}-{sc}-bridge-run1'
    mag = resample(echo(f'{IN}/{acq}/magnitude.nii.gz', 0), acq)
    mag /= np.percentile(mag[mag > 0], 99)
    mask = resample(sitk.ReadImage(f'{IN}/{acq}/mask.nii.gz', sitk.sitkFloat32), acq, sitk.sitkNearestNeighbor) > 0.5
    ph = resample(echo(f'{IN}/{acq}/phase.nii.gz', 2), acq, sitk.sitkNearestNeighbor)
    # head mask for phase: magnitude above noise, holes filled (scanner-masked / out-of-FOV voxels set to 0)
    head = ndimage.binary_opening(ndimage.binary_fill_holes(mag > 0.1), iterations=2)
    head = ndimage.binary_fill_holes(ndimage.binary_closing(head, iterations=3))
    ph[~head] = 0.0
    tf = resample(sitk.ReadImage(f'../../data/figure2/fields/{acq}__romeo-qsmrs__totalfield.nii.gz', sitk.sitkFloat32), acq)
    lf = resample(sitk.ReadImage(f'../../data/figure2/fields/{acq}__romeo-qsmrs_pdf-qsmrs__localfield.nii.gz', sitk.sitkFloat32), acq)
    qsm = resample(sitk.ReadImage(f'{KEEP}/romeo-qsmrs~pdf-qsmrs~whqsm-qsmrs-{acq}.nii.gz', sitk.sitkFloat32), acq)
    for a in (tf, lf, qsm):
        a[~mask] = 0.0  # outside the mask shown as 0, the neutral value of every colour scale
    rows.append((name, [mag, ph, tf, lf, qsm, qsm], [None, head, mask, mask, mask, None]))

COLS = [('Magnitude\n(echo 1)', 'gray', 0, 1, None),
        ('Phase\n(echo 3, wrapped)', 'gray', -np.pi, np.pi, 'rad'),
        ('Total field\n(ROMEO)', 'RdBu_r', -0.8, 0.8, 'ppm'),
        ('Local field\n(PDF)', 'gray', -0.03, 0.03, 'ppm'),
        ('Susceptibility\n(WH-QSM)', 'gray', -0.1, 0.15, 'ppm'),
        ('Regions\n(SynthSeg)', 'gray', -0.1, 0.15, 'regions')]
INK = '#1F2933'
fig, ax = plt.subplots(len(rows), len(COLS), figsize=(2.3 * len(COLS) + 0.8, 2.75 * len(rows) + 0.9), facecolor='white',
                       gridspec_kw=dict(wspace=0.04, hspace=0.06))
for r, (name, imgs, masks) in enumerate(rows):
    for c, (title, cmap, lo, hi, unit) in enumerate(COLS):
        a = ax[r, c]
        cm = plt.get_cmap(cmap)
        im = a.imshow(imgs[c], cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
        if unit == 'regions':
            a.imshow(overlay, interpolation='nearest')
        if masks[c] is not None:  # outline the mask on masked maps
            a.contour(masks[c].astype(float), levels=[0.5], colors='#52514e', linewidths=0.7)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if r == 0: a.set_title(title, fontsize=11.5, color=INK, pad=6)
        if c == 0: a.set_ylabel(name, fontsize=12, color=INK, labelpad=6)
        if r == len(rows) - 1 and unit == 'regions':
            from matplotlib.patches import Patch
            a.legend(handles=[Patch(facecolor=np.array(rgb) / 255, edgecolor='#9b9a95', lw=0.5, label=n) for n, _, rgb in REGIONS],
                     loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=7.6,
                     handlelength=1, handletextpad=0.4, columnspacing=0.8, labelspacing=0.3)
        elif r == len(rows) - 1 and unit:
            cax = a.inset_axes([0.1, -0.11, 0.8, 0.045])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[lo, 0, hi] if lo < 0 else [lo, hi])
            cb.ax.tick_params(labelsize=8.5, colors=INK, length=2)
            cb.ax.set_xticklabels([f'{v:g}' if unit != 'rad' else s for v, s in zip([lo, 0, hi], ['−π', '0', 'π'])])
            cb.set_label(unit, fontsize=9, color=INK, labelpad=1)
            cb.outline.set_visible(False)
fig.savefig('../fig2_representative_images.png', dpi=300, bbox_inches='tight', facecolor='white')
print('slice', z)
