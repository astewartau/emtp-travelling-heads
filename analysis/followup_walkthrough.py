# %% [markdown]
# # Follow-up analyses: how each figure was made
#
# This notebook rebuilds every figure from the follow-up analyses after the ISMRM 2027 abstract, starting from the regional values in `data/`. For each one it explains what goes in, the exact calculation, how to read the plot, and the caveats someone may ask about.
#
# The code is written to read like the maths (see **Notation** below). Each statistic says which subscripts stay free, and each plot call says which quantity goes in which box or bar. Data loading is in `analysis/notation.py` and drawing in `figures/src/followup_plots.py`.
#
# Each section ends with a **check** that the numbers recomputed here equal the ones in `results/followup/followup.json`, which is what was reported to the co-authors. If a check fails, the notebook stops.
#
# **Data version.** QSM-CI harmonization run with QSMxT v10.0.0 (October 2026): 6 subjects, 12 sessions, 72 acquisitions, 638 pipelines. The five sessions acquired in October and the re-run on QSMxT v11 are not included, so these numbers will change.
#
# | Fig. | Question | One-line answer |
# |---|---|---|
# | 1 | Should scan-rescan use the SD of *a* instead of median \|a − 1\|? | Same conclusion either way: bridge 3.3% vs local 2.2% (SD), 2.2% vs 1.4% (median). For a normal spread centred on 1 the two differ by a fixed factor of 0.674. |
# | 2 | How much does the inter-scanner deviation vary between scanner pairs? | The IQR across the 7 scanner pairs (≈ 6%) is about as large as the median deviation itself, but each pair is a different subject. |
# | 3 | Should the limits of agreement be taken after removing the scaling? | Removing each pair's fitted line narrows the LoA by only 5–8%, so most of the disagreement is regions scattering independently. |
# | 4 | What is the "Protocol" bar in Fig. 4 of the abstract? | The SD of two numbers, the bridge and local run-averaged means. Removing a line fitted on the same scans takes it to roughly rescan level, but that is a best case. |
# | 5 | Can pipelines be converted with one factor each? | Partly. One line per pipeline, learned on site-1 bridge scans, halves the spread between pipelines (pallidum 29–34 → 16–17 ppb). What remains is still larger than the scanner or protocol spread. |
# | 6 | Does a local-to-bridge calibration help a new subject? | No (4.0% → 4.2%). The local-to-bridge slope changes 3–5% from one subject to the next, and the protocols are already close (median slope 0.99). |
# | 7 | Why is the bridge scan-rescan noisier? | Mostly because of bridge run 1. In 9 of 12 sessions, run 2 vs run 3 is the best-agreeing of the three bridge pairs (p ≈ 0.004). |
# | 8 | Does dropping run 1 change the results? | Test-retest: bridge 1.6% vs local 1.5%. Inter-scanner: essentially unchanged. |

# %% [markdown]
# ## Setup and notation
#
# Every quantity is a table whose **row labels are its subscripts**. A statistic names the subscripts that stay **free** and takes the median, SD, … over all the others, just as in the formula.
#
# | Maths | Code |
# |---|---|
# | $a_{p,k}$: slope of pipeline $p$ on scan pair $k$ | `fits.a`, labelled by pipeline and the pair's subscripts (comparison, protocol, subject, scanner_x, scanner_y, run_x, run_y, …) |
# | $\chi_{p,s,c,\pi,r}$: mean χ of a region for pipeline, subject, scanner, protocol, run (ppb) | `chi['Pallidum']` |
# | $\operatorname{median}_k\, x_{p,\pi,k}$ (free: $p, \pi$) | `median(x, each=('pipeline', 'protocol'))` |
# | the same with SD, mean, IQR or a quantile | `sd`, `mean`, `iqr`, `quantile(x, q, each=…)` |
# | median over everything | `median(x)` |
# | only the test-retest pairs $k \in \mathcal T$ | `restrict(x, comparison='test_retest')` |
# | all pairs except those with run 1 | `restrict(x, with_run1=False)` |
# | $x_{\cdot,\text{bridge}}$ and $x_{\cdot,\text{local}}$ as two series | `by(x, 'protocol')` |
# | $(x - b)/a$, with a line $(a, b)$ per pipeline (or per pipeline and scanner, …) | `undo_line(x, line)` |
#
# SDs are sample SDs (ddof = 1). χ is in ppb, relative to each reconstruction's whole-brain mean.

# %%
import json
import sys

sys.path.insert(0, 'analysis'); sys.path.insert(0, 'figures/src')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

import emtp
import followup_plots as plots
from notation import (by, chi_by_label, chi_region, exclude, fit_line, fit_to_reference, having, iqr, leave_one_out,
                      mean, median, pair_fits, quantile, refit_local_into_bridge, restrict, sd, total, undo_line)

pd.set_option('display.precision', 2)
FU = json.load(open('results/followup/followup.json'))   # the reported follow-up values


def check(name, got, expected, tol=0.051):
    """Stop if a recomputed value differs from results/followup/followup.json."""
    assert abs(got - expected) <= tol, f'{name}: recomputed {got:.3f}, followup.json {expected:.3f}'
    print(f'check ok  {name}: {got:.2f} (followup.json {expected:.2f})')


reg = emtp.regions()                 # data/regional_chi.csv.gz
acq = emtp.acquisitions()            # acquisition -> subject, scanner, protocol, run
chi_lab = chi_by_label(reg)          # χ[pipeline, subject, scanner, protocol, run, region], 25 regions (per hemisphere)
chi = {R: chi_region(reg, R) for R in ('Caudate', 'Pallidum')}   # χ[pipeline, subject, scanner, protocol, run], bilateral
print(f'{reg.pipeline.nunique()} pipelines, {acq.shape[0]} acquisitions, {reg.region.nunique()} regions')
acq.groupby(['subject', 'scanner']).size().unstack(fill_value=0)

# %% [markdown]
# **Design.** Each cell above is one session: 3 bridge runs + 3 local runs, always in the order **B1 L1 B2 L2 B3 L3**. Every subject was scanned on the reference scanner of site 1, except s005001, who was scanned on the site-1 Elition instead. Only four subjects have more than one scanner (s004001, s005001, s005002, s006001), so all the inter-scanner results rest on those four people.
#
# **Regional values** (`data/regional_chi.csv.gz`, extracted by QSM-CI; see `extraction/README.md`). Every reconstruction is rigidly registered to the subject's reference scan and segmented once with SynthSeg on the reference magnitude. For each of 25 labels (left and right hemispheres separate: deep grey matter, thalamus, hippocampus, amygdala, cortex, WM, cerebellum, brainstem, ventral DC) we take the mean χ **minus the whole-brain mean of that reconstruction**. A label is dropped from a scan if under 90% of it is inside the reconstruction.

# %% [markdown]
# ## The building block: one orthogonal fit per scan pair
#
# Every agreement number here comes from one kind of calculation. Take one pipeline and two scans of the same subject, x (the reference) and y. Each region $i$ gives a point $(x_i, y_i)$: its mean χ in scan x and in scan y. We use only regions present in both scans, which is 21–25 of them.
#
# We fit $y = a\,x + b$ by **orthogonal (total least squares) regression**. This minimises the squared *perpendicular* distances of the points from the line, so x and y are treated the same way (both scans are noisy). The solution is the first principal component of the centred point cloud. With $s_{xx}, s_{yy}, s_{xy}$ the sample (co)variances:
#
# $$a = \frac{s_{yy} - s_{xx} + \sqrt{(s_{yy} - s_{xx})^2 + 4 s_{xy}^2}}{2 s_{xy}}, \qquad b = \bar y - a\,\bar x$$
#
# - **Slope deviation** $|a - 1|$, in %: how far the second scan is scaled relative to the first. Perfect agreement gives $a = 1$. Swapping x and y gives exactly $1/a$, so the direction only matters at second order.
# - **Bland–Altman limits of agreement** (half-width, ppb): $\mathrm{LoA} = 1.96\,\mathrm{SD}_i(y_i - x_i)$.
# - **Line-corrected LoA**: $1.96\,\mathrm{SD}_i\big(y_i - (a x_i + b)\big)$, the scatter left after removing this pair's own fitted line.
#
# QSM-CI's exclusion rules: no fit with fewer than 8 shared regions, with a flat map, or when $|a - 1| > 200\%$ (a failed reconstruction, not a reproducibility value).
#
# Here is one example. The session is s005001 on the site-1 Elition, bridge protocol, the clearest run-1 case (Fig. 7). The pipeline is ROMEO + V-SHARP + MEDI. The code below is the formulas above, line by line.

# %%
def tls_slope(x, y):
    """The closed form above."""
    sxx, syy, sxy = x.var(), y.var(), x.cov(y)
    return (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)


session = restrict(chi_lab, pipeline='romeo-qsmrs+vsharp-qsmrs+medi-qsmrs', subject='s005001', scanner='site1-elition', protocol='bridge')
run = {k: by(session, 'run')[f'run{k}'].droplevel(['pipeline', 'subject', 'scanner', 'protocol']) for k in (1, 2, 3)}   # χ_i per region

panels = []
for i, j in ((1, 2), (2, 3)):
    regions = run[i].index.intersection(run[j].index)              # regions present in both scans
    x, y = run[i][regions], run[j][regions]
    a, b = tls_slope(x, y), y.mean() - tls_slope(x, y) * x.mean()
    assert np.isclose((a, b), fit_line(x, y)).all()               # = the fit used for every pair
    assert np.isclose(tls_slope(y, x), 1 / a)                       # swapping the scans gives 1/a
    loa, loa_line = 1.96 * sd(y - x), 1.96 * sd(y - (a * x + b))
    print(f'run {j} vs run {i}: {len(regions)} regions, a = {a:.3f} (|a − 1| = {100 * abs(a - 1):.1f}%), b = {b:.1f} ppb, '
          f'LoA ±{loa:.1f} ppb, line-corrected LoA ±{loa_line:.1f} ppb')
    panels.append((f'Bridge run {j} vs run {i}:  a = {a:.3f}', f'Run {i}: regional χ (ppb)', f'Run {j}: regional χ (ppb)', x, y, (a, b)))
plots.example_fit(panels, name='example_fit')

# %% [markdown]
# Note what drives the slope: a few high-χ regions (pallidum, putamen) sit far from the rest, so they have most of the leverage. The cortex, WM and cerebellum cluster near zero. A slope deviation is therefore mostly saying "deep grey matter is scaled up or down relative to the rest of the brain".
#
# ### The pairs
#
# `emtp.comparison_plan` defines which scans get compared (the same as QSM-CI). Pairs never cross subjects.
#
# | Comparison | Pairs | Per pipeline and protocol |
# |---|---|---|
# | **test_retest** | same session and protocol, every two runs (1–2, 1–3, 2–3); x = earlier run | 12 sessions × 3 = **36** |
# | **inter_scanner** | same subject and protocol, every run on one scanner against every run on the other (different days, so run numbers don't correspond); x = the site-1 scanner | 8 session pairs × 9 = **72** (7 distinct scanner pairs) |
# | **inter_protocol** | same session, bridge run k vs local run k (acquired one after the other); x = bridge | 12 × 3 = **36** |
#
# All 638 pipelines are fitted on every pair (this takes about a minute). The result `fits` holds $a$, $b$ and both LoA for every pipeline and pair, labelled by the pair's subscripts.

# %%
fits = pair_fits(reg)
fitted = fits.a.notna()
pd.DataFrame({'fits': total(fitted, each=('comparison', 'protocol')),
              'pairs per pipeline': median(total(fitted, each=('pipeline', 'comparison', 'protocol')), each=('comparison', 'protocol'))})

# %% [markdown]
# The diagram draws the three kinds of pair. The counts in its titles come from `emtp.comparison_plan`, and panel D from the acquisition table.

# %%
plan = pd.DataFrame(emtp.comparison_plan(acq), columns=['comparison', 'x', 'y'])
plan['protocol'] = plan.x.map(acq.set_index('acquisition').protocol)
per_protocol = plan[plan.protocol == 'bridge'].comparison.value_counts()
plots.pair_design(acq, name='pairs_design', counts={
    'sessions': acq.groupby(['subject', 'scanner']).ngroups,
    'test_retest': per_protocol['test_retest'], 'inter_scanner': per_protocol['inter_scanner'],
    'inter_protocol': (plan.comparison == 'inter_protocol').sum(),
    'session_pairs': per_protocol['inter_scanner'] // 9,
    'scanner_pairs': restrict(fits.a, comparison='inter_scanner').index.get_level_values('scanner_pair').nunique()})

# %% [markdown]
# Panel D shows why the inter-scanner results rest on four people: s002001 and s003001 were scanned only at site 1. It also shows that S5 MR750 – S5 Prisma is the only scanner pair with two subjects, and that the two site-1 scanners never share a subject.
#
# ### How every box plot is summarised
#
# Each pipeline gives **one value**, usually the median of $|a-1|$ over its pairs of the kind in question. The box plots then show how that value is distributed across the 638 pipelines:
#
# - **box** = 25th–75th percentile across pipelines, **thick line** = median (the number printed next to it);
# - **whiskers = 5th and 95th percentiles** (not Tukey's 1.5 × IQR);
# - **+n** above a capped axis = number of pipelines beyond the cap;
# - **faint, dashed** boxes are the "after correction" or "without run 1" versions.
#
# So "bridge 2.2%" means "the median pipeline's median scan-rescan |a − 1| is 2.2%". The same 72 scans are behind all 638 pipelines, so the pipelines are not independent samples. The real sample size is subjects and sessions (see the caveats at the end).

# %% [markdown]
# ## Figure 1 · Scan-rescan: median |a − 1| or SD of a?
#
# **Question ([#2](https://github.com/astewartau/emtp-travelling-heads/issues/2)).** Within one scanner and protocol there should be no systematic scaling, so the spread of $a$ (its SD) is the natural measure. Median $|a-1|$ (the median absolute error) is meant for cases with a systematic error.
#
# **Calculation.** $\mathcal T_\pi$ = the test-retest pairs of protocol $\pi$: 36 per pipeline (12 sessions × 3 run pairs). For each pipeline $p$:
#
# $$m_{p,\pi} = \operatorname{median}_{k \in \mathcal T_\pi} |a_{p,k} - 1|, \qquad s_{p,\pi} = \operatorname{SD}_{k \in \mathcal T_\pi}\, a_{p,k}, \qquad d_{p,\pi} = \operatorname{mean}_{k \in \mathcal T_\pi} (a_{p,k} - 1)$$
#
# Left boxes: $m$, as in the abstract. Right boxes: $s$, the suggested measure. $d$ checks that there is no systematic error.
#
# **Why the two agree.** If $a - 1 \sim \mathcal N(0, \sigma^2)$, then $\mathrm{median}\,|a-1| = 0.674\,\sigma$. Here $0.674 \times 3.29 = 2.2$ and $0.674 \times 2.19 = 1.5$. So the two measures carry the same information and rank the protocols the same way. We kept median |a − 1| as the primary measure (decision D1 in `docs/decisions.md`) so test-retest and inter-scanner use the same scale and match QSM-CI.

# %%
a_T = 100 * restrict(fits.a, comparison='test_retest')       # a_{p,k} in %, test-retest pairs only

m = median(abs(a_T - 100), each=('pipeline', 'protocol'))     # m_{p,π}
s = sd(a_T, each=('pipeline', 'protocol'))                    # s_{p,π}
d = mean(a_T - 100, each=('pipeline', 'protocol'))            # d_{p,π}

display(pd.DataFrame({'median_p m': median(m, each='protocol'), 'median_p s': median(s, each='protocol'),
                      'median_p d': median(d, each='protocol'),
                      'ratio': median(m, each='protocol') / median(s, each='protocol')}).round(2))
for π in ('bridge', 'local'):
    check(f'{π} SD of a', median(s, each='protocol')[π], FU['issue_2_slope_spread'][π]['test_retest_sd_slope_pct']['median'])
    check(f'{π} mean a − 1', median(d, each='protocol')[π], FU['issue_2_slope_spread'][π]['test_retest_mean_slope_minus_1_pct']['median'])

plots.boxes({'Median |a − 1|\n(as in the abstract)': by(m, 'protocol'),
             'SD of a\n(suggested)':                 by(s, 'protocol')},
            ylabel='Scan–rescan, per pipeline (%)', ylim=(0, 7.5), name='fig1_sd_of_a');

# %% [markdown]
# **How to say it.** "The mean of a is 1 to within half a percent, so there is no systematic error, as expected. The SD of a gives bridge 3.3% vs local 2.2%, the same conclusion as the median |a − 1|. For a normal spread the two differ by a fixed factor of 0.67."
#
# **Possible questions.**
# - *Are the 36 pairs independent?* No. The three pairs in a session share scans: run 1 is in both 1–2 and 1–3. One odd scan therefore inflates two of the three pairs, which is exactly what happens with bridge run 1 (Fig. 7).
# - *Why is the bridge SD higher?* Mostly bridge run 1 (Figs. 7–8).

# %% [markdown]
# ## Figure 2 · Inter-scanner deviation for each scanner pair
#
# **Question ([#2](https://github.com/astewartau/emtp-travelling-heads/issues/2)).** Report the IQR too, to show how much the deviation varies between scanners.
#
# **Calculation.** $\mathcal I_\pi$ = the inter-scanner pairs of protocol $\pi$ (72 per pipeline); $\mathcal I_{\pi,c}$ = those of scanner pair $c$ (9 run combinations, or 18 for S5 MR750 – S5 Prisma, which has two subjects).
#
# $$M_{p,\pi} = \operatorname{median}_{k \in \mathcal I_\pi} |a_{p,k}-1| \;\text{(headline)}, \qquad M_{p,\pi,c} = \operatorname{median}_{k \in \mathcal I_{\pi,c}} |a_{p,k}-1| \;\text{(each box)}, \qquad \mathrm{IQR}_c\, M_{p,\pi,c} \;\text{(spread across the 7 pairs)}$$
#
# Each box shows $M_{p,\pi,c}$ across the 638 pipelines. Pairs are ordered by their median.

# %%
dev_I = 100 * abs(restrict(fits.a, comparison='inter_scanner') - 1)    # |a_{p,k} − 1| in %, inter-scanner pairs

M = median(dev_I, each=('pipeline', 'protocol'))                       # M_{p,π}
M_c = median(dev_I, each=('pipeline', 'protocol', 'scanner_pair'))     # M_{p,π,c}
spread_c = iqr(M_c, each=('pipeline', 'protocol'))                     # IQR over scanner pairs c

display(pd.DataFrame({'median_p M': median(M, each='protocol'), '75th pct': quantile(M, .75, each='protocol'),
                      '95th pct': quantile(M, .95, each='protocol'), 'median_p IQR_c': median(spread_c, each='protocol')}).round(2))
for π in ('bridge', 'local'):
    check(f'{π} inter-scanner median', median(M, each='protocol')[π], FU['issue_2_slope_spread'][π]['inter_scanner_median_abs_slope_dev_pct']['median'])
    check(f'{π} IQR across scanner pairs', median(spread_c, each='protocol')[π], FU['issue_2_slope_spread'][π]['inter_scanner_iqr_across_scanner_pairs_pct']['median'])

# each scanner pair: median over pipelines
median(M_c, each=('scanner_pair', 'protocol')).unstack().round(2)

# %%
order = mean(median(M_c, each=('scanner_pair', 'protocol')), each='scanner_pair').sort_values().index
plots.boxes({plots.scanner_pair(c): by(restrict(M_c, scanner_pair=c), 'protocol') for c in order},
            ylabel='Inter-scanner |a − 1| (%)\nper pipeline, median over run pairs', ylim=(0, 30), cap=25, figsize=(11, 4.2),
            title='One subject per scanner pair (two for S5 MR750 – S5 Prisma) · +n: pipelines above 25%', name='fig2_scanner_pairs');

# %% [markdown]
# **How to say it.** "The variation between scanner pairs is as large as the deviation itself: an IQR of about 6% against a median of 6–7%. But every pair is a different subject, so this mixes scanner and subject. With these data we cannot say whether it's the scanners or the people."
#
# **Same site vs different sites.** The one same-site pair, S5 MR750 – S5 Prisma (GE vs Siemens, different local protocols), agrees no better than the pairs between sites:

# %%
median(median(dev_I, each=('pipeline', 'protocol', 'same_site')), each=('same_site', 'protocol')).unstack().round(2)

# %% [markdown]
# **Is bridge better than local across scanners?** Not at the median: 6.4% vs 7.0%. Before the abstract it was 6.7% vs 6.8%, using matched runs only. Per pipeline it's close to a coin flip. The bootstrap over subjects (resampling the 6 subjects with replacement, 500 times; rough with only four multi-scanner subjects) puts the median difference's 95% interval across zero. The bridge advantage is in the tail: fewer pipelines go badly wrong.

# %%
M_bridge, M_local = by(M, 'protocol').values()
print(f"share of pipelines with M_bridge < M_local: {((M_bridge - M_local).dropna() < 0).mean():.2f}")
b = FU['bootstrap_over_subjects']
pd.DataFrame({k.replace('inter_scanner_', ''): {'estimate': b[k]['estimate'], 'ci95 low': b[k]['ci95'][0], 'ci95 high': b[k]['ci95'][1]}
              for k in ('inter_scanner_median_bridge_minus_local_pct', 'inter_scanner_p75_bridge_minus_local_pct',
                        'inter_scanner_p95_bridge_minus_local_pct')}).T

# %% [markdown]
# ## Figure 3 · Limits of agreement before and after removing the scaling
#
# **Question ([#3](https://github.com/astewartau/emtp-travelling-heads/issues/3)).** If the premise is a global scaling error, the LoA should be taken after correcting for it. The uncorrected LoA stays useful as a sanity check, since the scaling assumption may not hold for every protocol and pipeline.
#
# **Calculation.** For each pair $k$, from its fit over up to 25 regions: $\mathrm{LoA}_k = 1.96\,\mathrm{SD}_i(y_i - x_i)$ (solid boxes, as in the abstract) and $\mathrm{LoA}^{\text{line}}_k = 1.96\,\mathrm{SD}_i\big(y_i - (a_k x_i + b_k)\big)$ (faint boxes). For each pipeline, the median over its pairs of each kind:
#
# $$L_{p,\kappa,\pi} = \operatorname{median}_{k} \mathrm{LoA}_k, \qquad L^{\text{line}}_{p,\kappa,\pi} = \operatorname{median}_{k} \mathrm{LoA}^{\text{line}}_k$$
#
# with $\kappa$ the comparison (scan–rescan or inter-scanner).
#
# **The size of the scaling effect.** Variances add, so $1 - (L^{\text{line}}/L)^2$ is roughly the share of the variance of the regional differences explained by the pair's line. That share comes out at about 10–15%.

# %%
L = median(fits.loa, each=('pipeline', 'comparison', 'protocol'))                 # L_{p,κ,π}
L_line = median(fits.loa_corrected, each=('pipeline', 'comparison', 'protocol'))   # L^line_{p,κ,π}

table = pd.DataFrame({'median_p L (±ppb)': median(L, each=('comparison', 'protocol')),
                      'median_p L_line (±ppb)': median(L_line, each=('comparison', 'protocol')),
                      'variance explained by the line (%)': median(100 * (1 - (L_line / L) ** 2), each=('comparison', 'protocol'))})
table['narrower by (%)'] = 100 * (1 - table['median_p L_line (±ppb)'] / table['median_p L (±ppb)'])
for (κ, π), row in restrict(table, comparison=['test_retest', 'inter_scanner']).iterrows():
    check(f'{κ} {π} LoA', row.iloc[0], FU['issue_3_corrected_loa'][f'{κ}_{π}']['loa_ppb']['median'])
    check(f'{κ} {π} line-corrected LoA', row.iloc[1], FU['issue_3_corrected_loa'][f'{κ}_{π}']['loa_corrected_ppb']['median'])
restrict(table, comparison=['test_retest', 'inter_scanner']).round(1)

# %%
def loa_boxes(κ):
    return {f'{π}{label}': restrict(X, comparison=κ, protocol=π)
            for π in ('bridge', 'local') for label, X in (('', L), (', scale removed', L_line))}

plots.boxes({'Scan–rescan': loa_boxes('test_retest'), 'Inter-scanner': loa_boxes('inter_scanner')},
            ylabel='Limits of agreement, ± ppb\nper pipeline, median over scan pairs', ylim=(0, 55), cap=45, figsize=(8, 4.2),
            title='Faint, dashed: after removing each pair’s fitted line · +n: pipelines beyond 45 ppb', name='fig3_loa_corrected');

# %% [markdown]
# **How to say it.** "Removing each pair's own scaling only narrows the inter-scanner LoA from ±19 to ±18 ppb (bridge) and ±17.5 to ±16 ppb (local). So most of the disagreement is regions moving independently, not a global scale."
#
# **Why so little, when |a − 1| is 6–7%?** A 6% scale error moves a region at +150 ppb (pallidum) by 9 ppb, but a region near 0 (WM, cortex) hardly at all. Across 25 regions, most of which sit near zero, the scaling adds only a small part of the variance next to the region-by-region scatter.
#
# **Possible questions.**
# - *Is the corrected LoA optimistic?* Slightly. The line is fitted to the same ~25 points it corrects (2 parameters out of ~25), so it can only shrink the scatter. That makes "only 5–8%" a generous upper bound on what scale correction buys.
# - *Bias?* Values are relative to each map's whole-brain mean, so the bias $\overline{y - x}$ is small. The LoA shown is the half-width $1.96\,\mathrm{SD}$ around the bias.
# - *Why ddof = 1?* That is the sample SD. QSM-CI uses ddof = 0, so its LoA are about 2.5% narrower for a 20-region fit (`extraction/README.md`).

# %% [markdown]
# ## Figure 4 · Sources of variability, with the protocol bar corrected for scaling
#
# **Question ([#4](https://github.com/astewartau/emtp-travelling-heads/issues/4)).** What does "Protocol" in Fig. 4 of the abstract mean, and shouldn't it be corrected for $a$?
#
# **The bars.** $\chi_{p,s,c,\pi,r}$ is the bilateral mean χ of one region for pipeline $p$, subject $s$, scanner $c$, protocol $\pi$ and run $r$; $\bar\chi_{p,s,c,\pi} = \operatorname{mean}_r \chi_{p,s,c,\pi,r}$ averages the three runs. Each bar is an SD over **one** subscript, with the others fixed, followed by medians:
#
# | Bar | Formula |
# |---|---|
# | Rescan | $\operatorname{median}_p \operatorname{median}_{s,c} \operatorname{SD}_r\, \chi_{p,s,c,\pi,r}$ |
# | Protocol | $\operatorname{median}_p \operatorname{median}_{s,c} \operatorname{SD}_\pi\, \bar\chi_{p,s,c,\pi}$ (two values, bridge and local) |
# | Scanner | $\operatorname{median}_p \operatorname{median}_{s} \operatorname{SD}_c\, \bar\chi_{p,s,c,\pi}$ (subjects with ≥ 2 scanners) |
# | Pipeline | $\operatorname{median}_{s,c} \operatorname{SD}_p\, \bar\chi_{p,s,c,\pi}$ |
# | Between subjects | $\operatorname{median}_p \operatorname{SD}_s\, \bar\chi_{p,s,c_0,\pi}$, with $c_0$ the site-1 Achieva |
#
# **Protocol in plain words.** For two values the SD is $|\Delta|/\sqrt 2$. So "protocol = 3.0 ppb" in the pallidum means the run-averaged bridge and local means typically differ by about 4.2 ppb.

# %%
def sources(chi):
    """The Fig. 4 bars (SD, ppb) of one region: one column per protocol, and the protocol bar under 'both'."""
    chibar = mean(chi, each=('pipeline', 'subject', 'scanner', 'protocol'))                     # χ̄: mean over runs
    bars = pd.DataFrame({
        'rescan':           median(median(sd(chi, each=('pipeline', 'subject', 'scanner', 'protocol')), each=('pipeline', 'protocol')), each='protocol'),
        'scanner':          median(median(sd(chibar, each=('pipeline', 'subject', 'protocol')), each=('pipeline', 'protocol')), each='protocol'),
        'pipeline':         median(sd(chibar, each=('subject', 'scanner', 'protocol')), each='protocol'),
        'between subjects': median(sd(restrict(chibar, scanner='site1-achieva'), each=('pipeline', 'protocol')), each='protocol'),
    }).T
    bars.loc['protocol', 'both'] = median(median(sd(chibar, each=('pipeline', 'subject', 'scanner')), each='pipeline'))
    return bars


bars = {R: sources(chi[R]) for R in chi}
for R in bars:   # the same as the abstract's Fig. 4 (emtp.spread)
    ref = emtp.spread(reg, R).rename(index={'between_subjects': 'between subjects'})
    rows = ['rescan', 'scanner', 'pipeline', 'between subjects']
    assert np.allclose(bars[R].loc[rows, ['bridge', 'local']], ref.loc[rows, ['bridge', 'local']])
    print(f'\n{R} (SD, ppb): same as the abstract Fig. 4'); display(bars[R].round(1))

# %% [markdown]
# **Correction (hatched bar).** For each pipeline $p$ and scanner $c$, take the line $(a_{p,c}, b_{p,c})$ as the median of that scanner's local-vs-bridge fits (bridge = x) over all its subjects and runs. Map every local value into bridge space, $\chi \to (\chi - b_{p,c})/a_{p,c}$, and recompute the protocol bar. The line is estimated **on the same scans it corrects**, so this is a best case. The honest, out-of-sample version is Fig. 6.

# %%
line_pc = median(restrict(fits[['a', 'b']], comparison='inter_protocol'), each=('pipeline', 'scanner_x')).rename_axis(['pipeline', 'scanner'])
print(f"local-vs-bridge slope a_(p,c): median {median(line_pc.a):.3f}, IQR {quantile(line_pc.a, .25):.3f}–{quantile(line_pc.a, .75):.3f}")

protocol_after = {}
for R, x in chi.items():
    corrected = pd.concat([restrict(x, protocol='bridge'), undo_line(restrict(x, protocol='local'), line_pc)])
    protocol_after[R] = sources(corrected).loc['protocol', 'both']
    check(f'{R} protocol SD', bars[R].loc['protocol', 'both'], FU['issue_4_protocol_spread'][R]['protocol_sd_ppb'])
    check(f'{R} protocol SD after removing the line', protocol_after[R], FU['issue_4_protocol_spread'][R]['protocol_sd_corrected_ppb'])

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'wspace': 0.08})
for ax, (R, B) in zip(axs, bars.items()):
    plots.hbars({'Rescan\nsame session, different run':         {'bridge': B.bridge['rescan'], 'local': B.local['rescan']},
                 'Protocol\nsame scanner, local vs bridge':      {'both protocols': B.both['protocol'], 'after removing scaling (best case)': protocol_after[R]},
                 'Scanner\nsame protocol, different scanner':    {'bridge': B.bridge['scanner'], 'local': B.local['scanner']},
                 'Pipeline\nsame scan, different pipeline':      {'bridge': B.bridge['pipeline'], 'local': B.local['pipeline']},
                 'Between subjects\nsame pipeline and scanner':  {'bridge': B.bridge['between subjects'], 'local': B.local['between subjects']}},
                xlabel=f'SD of mean {R.lower()} susceptibility (ppb)', ax=ax, title=R, legend=(R == 'Pallidum'))
axs[1].set_yticklabels([])
plots.save(fig, 'fig4_protocol_corrected')

# %% [markdown]
# **How to say it.** "The protocol bar is the bridge–local difference itself, written as an SD. It ignores a. If I first map local into bridge space with each scanner's median local-to-bridge fit, it drops from 2.1 to 1.8 ppb in the caudate and from 3.0 to 2.1 ppb in the pallidum, roughly rescan level. But that line was fitted on the same scans, so it's the best case."
#
# **Possible questions.**
# - *Is the bar inflated by rescan noise?* A little. Each of the two values is a mean of 3 runs, so rescan noise adds about $\mathrm{SD}_{\text{rescan}}/\sqrt 3$ to each one.
# - *Why are the bars not corrected in the figure?* Decision D4: the corrected bars are in-sample. They will be corrected once there are enough sessions to estimate the line out of sample.

# %% [markdown]
# ## Figure 5 · One conversion line per pipeline
#
# **Question ([#6](https://github.com/astewartau/emtp-travelling-heads/issues/6)).** Are the differences between pipelines a global factor that could be converted away?
#
# **Calculation.** Write $t$ for a scan (subject, scanner, protocol, run) and $i$ for a region.
# 1. **Typical pipeline** (consensus): $\tilde\chi_{t,i} = \operatorname{median}_p \chi_{p,t,i}$.
# 2. For every pipeline and scan, fit $\chi_{p,t,\cdot} \approx a_{p,t}\,\tilde\chi_{t,\cdot} + b_{p,t}$ over the regions with the same orthogonal fit.
# 3. **Conversion line of a pipeline:** $(a_p, b_p) = \operatorname{median}_{t \in \mathcal R} (a_{p,t}, b_{p,t})$ over the **site-1 Achieva bridge scans** $\mathcal R$ (5 subjects × 3 runs). Apply $\chi \to (\chi - b_p)/a_p$ to all of that pipeline's scans.
# 4. Recompute the Fig. 4 "Pipeline" bar on the **held-out scans** (everything not in $\mathcal R$). This tests whether a line learned on one scanner and protocol carries over to the others.
#
# The other bars (rescan, protocol, scanner, between subjects) are the uncorrected Fig. 4 values, there for scale.

# %%
consensus = median(chi_lab, each=('subject', 'scanner', 'protocol', 'run', 'region'))     # χ̃_{t,i}
line_pt = fit_to_reference(chi_lab, consensus)                                             # (a, b)_{p,t}
line_p = median(restrict(line_pt, scanner='site1-achieva', protocol='bridge'), each='pipeline')   # (a, b)_p

held_out = lambda x: exclude(x, scanner='site1-achieva', protocol='bridge')
pipeline_bar = {R: {'as is': sources(held_out(x)).loc['pipeline'], 'one line': sources(held_out(undo_line(x, line_p))).loc['pipeline']}
                for R, x in chi.items()}
for R, v in pipeline_bar.items():
    print(f"{R}: pipeline SD on held-out scans  bridge {v['as is'].bridge:.1f} → {v['one line'].bridge:.1f} ppb,  local {v['as is'].local:.1f} → {v['one line'].local:.1f} ppb")
    for π in ('bridge', 'local'):
        check(f'{R} {π} pipeline SD after one line', v['one line'][π], FU['issue_6_pipeline_conversion'][R]['global_from_reference_bridge']['pipeline_sd_held_out_scans_ppb'][π])

# %% [markdown]
# **How stable is a pipeline's line?** Within each pipeline, take the SD of $\ln a_{p,t}$ (≈ relative SD of $a$) across runs, protocols, scanners or subjects, and compare it with how much the pipelines' $a_p$ differ from each other. With $\overline{\ln a}$ averaged over runs:

# %%
ln_a = np.log(line_pt.a.clip(lower=1e-3))
ln_a_bar = mean(ln_a, each=('pipeline', 'subject', 'scanner', 'protocol'))
within = pd.Series({
    'runs':      median(median(sd(ln_a, each=('pipeline', 'subject', 'scanner', 'protocol')), each='pipeline')),
    'protocols': median(median(sd(ln_a_bar, each=('pipeline', 'subject', 'scanner')), each='pipeline')),
    'scanners':  median(median(sd(ln_a_bar, each=('pipeline', 'subject', 'protocol')), each='pipeline')),
    'subjects':  median(median(sd(restrict(ln_a_bar, scanner='site1-achieva'), each=('pipeline', 'protocol')), each='pipeline'))})
print('SD of a within a pipeline (%):', (100 * within).round(1).to_dict())
for k in within.index:
    check(f'within-pipeline SD of a across {k}', 100 * within[k], FU['issue_6_pipeline_conversion']['sd_of_slope_within_pipeline_pct'][k]['median'])
a_p = median(line_pt.a, each='pipeline')
print(f'between pipelines: a_p has IQR {quantile(a_p, .25):.2f}–{quantile(a_p, .75):.2f} (about ±15%)')

# %%
B, conv = bars['Pallidum'], pipeline_bar['Pallidum']
plots.hbars({'Rescan':                           {'bridge': B.bridge['rescan'], 'local': B.local['rescan']},
             'Protocol':                         {'both protocols': B.both['protocol']},
             'Scanner':                          {'bridge': B.bridge['scanner'], 'local': B.local['scanner']},
             'Pipeline, one line\nper pipeline': {'bridge': conv['one line'].bridge, 'local': conv['one line'].local},
             'Between subjects':                 {'bridge': B.bridge['between subjects'], 'local': B.local['between subjects']},
             'Pipeline, as is':                  {'bridge': conv['as is'].bridge, 'local': conv['as is'].local}},
            xlabel='SD of mean pallidum χ (ppb)', name='fig5_pipeline_conversion');

# %% [markdown]
# **How to say it.** "Each pipeline gets one slope and intercept relative to a typical pipeline, learned on the site-1 bridge scans. Applied to all the other scans, it roughly halves the spread between pipelines: 29–34 ppb becomes 16–17 ppb in the pallidum. So the lines carry over: a varies about 2% across protocols, 4% across scanners and 5% across subjects, while pipelines differ by about ±15%. But what's left is still much bigger than the scanner (3–7 ppb) or protocol (2–3 ppb) spread. Using the same pipeline beats converting between pipelines."
#
# **Possible questions.**
# - *Why the median over pipelines as the reference?* It's robust to the failed pipelines, and $a_p = 1$ then means "typical". The choice of reference doesn't change the spread *between* converted pipelines much, because every pipeline gets its own line.
# - *What is the remaining 16–17 ppb?* The part of each pipeline's deviation that isn't a straight line: region-specific differences such as background-removal erosion at the edges or under-estimated deep grey matter in some inversions. Those vary by region, not by a global scale.
# - *Does the conversion hurt the between-subject differences?* No: between subjects goes from 29.3 to 28.4 ppb (bridge), so the biology is kept.

# %% [markdown]
# ## Figure 6 · Does a local-to-bridge calibration help a new subject?
#
# **Question ([#5](https://github.com/astewartau/emtp-travelling-heads/issues/5)).** The framework idea is to learn, per scanner, how the local protocol maps to the bridge, and then use that to bring local-only data into bridge space.
#
# **Calculation (leave one subject out).** Use the scanners with at least two subjects: site-1 Achieva (5), site-5 MR750 (2) and site-5 Prisma (2).
# 1. Each subject's local-to-bridge line on scanner $c$: $(a_{p,c,s}, b_{p,c,s})$ = the median over its 3 inter-protocol fits (bridge run k = x, local run k = y).
# 2. For a new subject $s$, learn the line from the others: $(\hat a, \hat b)_{p,c,s} = \operatorname{median}_{s' \ne s} (a, b)_{p,c,s'}$. With only 2 subjects, that means the other one.
# 3. Map the subject's local scan into bridge space, $\chi \to (\chi - \hat b)/\hat a$, refit it against the bridge scan of the same run, and record $|a' - 1|$.
# 4. Compare three cases. *No correction*: $|a-1|$ as fitted. *Other subjects*: the honest test. *Same subject*: the subject's own line, the in-sample best case, impossible in practice.
#
# Each pipeline gives the median over its held-out pairs; the boxes show that across pipelines.

# %%
scanners = [c for c, n in acq.groupby('scanner').subject.nunique().items() if n >= 2]
fits_P = restrict(fits[['a', 'b']], comparison='inter_protocol', scanner_x=scanners)

line_s = median(fits_P, each=('pipeline', 'scanner_x', 'subject'))      # (a, b)_{p,c,s}
line_loo = leave_one_out(line_s, over='subject')                          # (â, b̂)_{p,c,s} = median_{s'≠s} (a, b)_{p,c,s'}
pairs = having(fits_P, line_loo)                                          # pairs whose subject has a learned line

dev_P = pd.DataFrame({'no correction': 100 * abs(pairs.a - 1),
                      'other subjects': 100 * refit_local_into_bridge(chi_lab, pairs, line_loo),
                      'same subject': 100 * refit_local_into_bridge(chi_lab, pairs, line_s)})
per_pipeline = median(dev_P, each='pipeline')
display(median(per_pipeline).round(2).to_frame('median_p (%)').T)
for col, key in (('no correction', 'none'), ('other subjects', 'loso'), ('same subject', 'in_sample')):
    check(f'calibration: {col}', median(per_pipeline[col]), FU['issue_5_calibration']['all'][f'{key}_abs_slope_dev']['median'])
print('SD_s a_(p,c,s), median over pipelines (%):', (100 * median(sd(line_s.a, each=('pipeline', 'scanner_x')), each='scanner_x')).round(1).to_dict())

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.2, 1.6], 'wspace': 0.3})
plots.boxes({'No\ncorrection':                                {'no correction': per_pipeline['no correction']},
             'Calibrated,\nlearned on\nother subjects':       {'other subjects': per_pipeline['other subjects']},
             'Calibrated,\nlearned on the\nsame subject':     {'same subject': per_pipeline['same subject']}},
            ylabel='Local vs bridge |a − 1| (%), same scanner\nper pipeline, median over subjects and runs', ax=left, ylim=(0, 10),
            colors={'no correction': plots.GREY, 'other subjects': plots.PURPLE, 'same subject': plots.PURPLE},
            faint={'same subject'}, legend=False, title='Does a local-to-bridge calibration help a new subject?')
plots.subject_boxes(100 * (median(fits_P.a, each=('scanner_x', 'subject', 'pipeline')) - 1),
                    ylabel='Local-to-bridge slope a − 1 (%)\nper pipeline, median over 3 runs', ax=right, ylim=(-16, 13),
                    title='Why not: each subject’s local-to-bridge slope')
plots.save(fig, 'fig6_calibration')

# %% [markdown]
# **How to say it.** "Learning the local-to-bridge mapping from the other subjects and applying it to a new one gives no gain: 4.0% before, 4.2% after. Using the subject's own mapping gets to 1.0%, but that's circular. The right panel shows why. The slope moves 3–5% from one subject to the next on the same scanner, around a median of 0.99, so there's no stable offset to learn. With protocols this similar there isn't much to calibrate."
#
# **Possible questions.**
# - *Isn't this just too few subjects?* Partly. Two of the three scanners have only two subjects, so "the others" is a single person. But the between-subject SD of the slope (3–5%) is as large as the effect itself on the 5-subject scanner too.
# - *Why is local vs bridge on the same scanner (4%) bigger than scan-rescan (1.4–2.2%)?* It compares two different acquisitions with different echo times, resolution and readout. Even matched runs (bridge k vs local k) differ by more than rescan noise. It's just not consistent across subjects.
# - *S5 Prisma, s005001 at −8.6%?* That subject's local protocol reads lower than its bridge on the Prisma for most pipelines. Another subject on the same scanner doesn't show it (−3.0%), which is the point.

# %% [markdown]
# ## Figure 7 · Bridge run 1 is the odd one out
#
# **Question ([#10](https://github.com/astewartau/emtp-travelling-heads/issues/10)).** Why does the bridge look noisier in scan-rescan?
#
# **Calculation.** For each session (subject × scanner), protocol and run pair: $r^{ij} = \operatorname{median}_p |a_{p,k} - 1|$, where $k$ is the pair (run $i$, run $j$). Each pipeline has exactly one fit per run pair. Dots = $r^{12}$ and $r^{13}$; diamond = $r^{23}$.
#
# **The count.** Run 1 is "the odd one out" when **both** pairs containing it are worse than run 2 vs run 3: $r^{12} > r^{23}$ and $r^{13} > r^{23}$. Equivalently, run 2 vs run 3 is the best-agreeing of the three pairs. Under the null hypothesis that run 1 is nothing special, each of the three pairs is equally likely to be the best one, so the chance is 1/3 per session. Treating the 12 sessions as independent: $P(X \ge 9 \mid n = 12, p = 1/3) \approx 0.004$.

# %%
dev_T = 100 * abs(restrict(fits.a, comparison='test_retest') - 1)
r = median(dev_T, each=('subject', 'scanner_x', 'protocol', 'run_x', 'run_y'))      # median over pipelines
r12, r13, r23 = (restrict(r, run_x=f'run{i}', run_y=f'run{j}').droplevel(['run_x', 'run_y']) for i, j in ((1, 2), (1, 3), (2, 3)))

run1_odd = (r12 > r23) & (r13 > r23)
odd = pd.DataFrame({'sessions with run 1 odd': total(run1_odd, each='protocol'), 'sessions': total(run1_odd > -1, each='protocol')})
odd['expected by chance'] = odd.sessions / 3
odd['p (binomial, one-sided)'] = binom.sf(odd['sessions with run 1 odd'] - 1, odd.sessions, 1 / 3)
display(odd.round(4))
check('bridge sessions with run 1 odd', odd.loc['bridge', 'sessions with run 1 odd'], FU['issue_10_first_run']['odd_run_per_session']['bridge']['run1'], tol=0)
restrict(pd.DataFrame({'r12': r12, 'r13': r13, 'r23': r23, 'run 1 odd': run1_odd}), protocol='bridge').round(2)

# %%
plots.run_pairs(r12, r13, r23, name='fig7_run_pairs_sessions')

# %% [markdown]
# **How to say it.** "Nearly all the extra bridge deviation comes from one scan per session: bridge run 1, the first QSM scan, which usually came after a structural. In 9 of 12 sessions, run 2 vs run 3 is the best-agreeing bridge pair. By chance you'd expect 4 (p ≈ 0.004). The local protocol, acquired in between, doesn't show it (4 of 12)."
#
# **Possible questions.**
# - *Some of the 9 are marginal.* Yes. s005001 on the MR750 is 1.46 / 1.85 vs 1.43. The clear cases are s002001, s004001 on both scanners, s005001 on the Elition and on the Prisma. s005002 on the Achieva and on the MR750 go the other way: there run 3 is the odd one.
# - *Are the sessions independent?* The p-value assumes so. Several sessions share a subject (s005001 has three). It is a descriptive test, not a formal one.
# - *What have we ruled out?* Head motion within the session (≈ 0.5° between runs, no larger for run 1), ghosting and edge sharpness (`data/image_quality.csv`), and field drift between runs (a few Hz, the same for both protocols, `data/b0_drift.csv`). Still open: head-support pillows settling over the first 3–5 minutes, scanner warm-up (at site 1, s002001 was the first scan of the day, and s004001 and s005001 came after a gap of more than 30 minutes), and physiological settling. Randomising or counterbalancing the run order, or adding a dummy scan first, would test this (decision D6).
# - *Bridge run 1 always comes first, so is this the first scan or the bridge protocol?* The design can't separate them. That is why counterbalancing is proposed.

# %% [markdown]
# ## Figure 8 · Results with and without run 1
#
# **Calculation.** As in Figs. 1–2: for each pipeline, the median $|a-1|$ over its pairs. "Without run 1" drops every pair in which either scan is run 1:
# - test-retest keeps 1 of 3 pairs per session (run 2 vs run 3), so 12 per pipeline;
# - inter-scanner keeps 4 of 9 run combinations (runs 2–3 × runs 2–3), so 32 per pipeline.
#
# Because test-retest keeps only 12 pairs, its "without run 1" boxes are wider. The median is taken over fewer values.

# %%
def per_pipeline_dev(comparison, **only):
    """median_k |a_{p,k} − 1| (%) over the pairs of one comparison, for each pipeline and protocol."""
    return median(100 * abs(restrict(fits.a, comparison=comparison, **only) - 1), each=('pipeline', 'protocol'))


groups = {κ: {f'{π}{label}': restrict(per_pipeline_dev(κ, **only), protocol=π)
              for π in ('bridge', 'local') for label, only in (('', {}), (', without run 1', {'with_run1': False}))}
          for κ in ('test_retest', 'inter_scanner')}
display(pd.DataFrame({κ: {k: median(v) for k, v in g.items()} for κ, g in groups.items()}).round(2))
for κ, g in groups.items():
    for π in ('bridge', 'local'):
        check(f'{κ} {π} without run 1', median(g[f'{π}, without run 1']), FU['issue_10_first_run'][κ][π]['pairs_without_run1_pct']['median'])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'wspace': 0.25})
plots.boxes({'Test-retest': groups['test_retest']}, ylabel='|a − 1| (%), one value per pipeline', ax=axs[0], ylim=(0, 5))
plots.boxes({'Inter-scanner': groups['inter_scanner']}, ylabel='', ax=axs[1], ylim=(0, 32), legend=False)
plots.save(fig, 'fig8_without_run1')

# %% [markdown]
# **How to say it.** "Leave run 1 out and bridge and local scan-rescan are the same, 1.6% vs 1.5%. The inter-scanner results barely move: bridge 6.4% either way, local 7.0% vs 6.8%. So the bridge may not be noisier as a protocol, and its length may matter less than we thought. The comparison across scanners doesn't depend on run 1."
#
# **Why doesn't inter-scanner change?** Inter-scanner differences (6–7%) are much bigger than the extra run-1 noise (~1%). Noise adds in quadrature: $\sqrt{6.4^2 + 1^2} \approx 6.5$, so removing it barely shows.

# %% [markdown]
# ## Caveats that apply to everything
#
# - **The sample size is people, not pipelines.** 638 pipelines × many pairs looks like a lot of data, but it all comes from 6 subjects and 12 sessions, and from only 4 subjects for inter-scanner. Pipelines, run pairs and run combinations share scans, so they aren't independent. The only formal tests here are the bootstrap over subjects (Fig. 2) and the binomial count over sessions (Fig. 7).
# - **Scanner and subject are confounded.** Most scanner pairs have one subject, and site 1's two scanners never share a subject.
# - **Head position between sessions.** The head's angle to B0 changes by 2.5–15° between sessions but only ~0.5° within one. Every between-session pair is also a different scanner, and the largest angles are all in the GE sessions. With 8 session pairs, head position and scanner can't be told apart (`data/head_pose.csv`).
# - **All 638 pipelines are included everywhere**, including poor ones. Medians and percentiles keep them from dominating, and the tails (p75, p95, +n) show them.
# - **Regions are per hemisphere and include cortex, WM and cerebellum.** The slope is driven mostly by the deep grey matter, which has the largest χ range.
# - **Data version.** QSMxT v10.0.0. The v11 re-run, with QSM.rs ports corrected to match their references, and the five October sessions (sites 2 and 3, including a single-echo 0.8 mm local protocol) will change these numbers.
