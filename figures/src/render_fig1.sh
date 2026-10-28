#!/bin/bash
# Render fig1_study_design.html to ../fig1_study_design.png at 2.5x with headless Firefox
# (the SVG keeps its viewBox; only its pixel size is scaled).
set -e
cd "$(dirname "$0")"
read W H < <(grep -o 'viewBox="0 0 [0-9]* [0-9]*"' fig1_study_design.html | head -1 | tr -dc '0-9 ' | awk '{print $3, $4}')
SW=$((W * 5 / 2)); SH=$((H * 5 / 2))
sed "s#width=\"$W\" height=\"$H\"#width=\"$SW\" height=\"$SH\"#" fig1_study_design.html > .fig1_render.html
PROF=$(mktemp -d)
firefox --headless --profile "$PROF" --no-remote --window-size=$SW,$SH \
  --screenshot "$PWD/../fig1_study_design.png" "file://$PWD/.fig1_render.html" >/dev/null 2>&1
rm -rf "$PROF" .fig1_render.html
