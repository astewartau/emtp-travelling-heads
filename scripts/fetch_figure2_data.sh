#!/bin/bash
# Figure 2 inputs (367 MB of NIfTI: one subject, bridge run 1, three scanners) are not in git.
# Copy them from Bunya (needs the ssh alias `bunya`).
set -e
cd "$(dirname "$0")/.."
rsync -a --info=progress2 bunya:/scratch/user/uqaste15/emtp-travelling-head/ismrm-abstract/data/figure2_sites/ data/figure2/
