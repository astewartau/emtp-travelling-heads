"""Figure 5: QSM-CI screenshot (harmonization pipeline matrix, ROMEO field mapping) with a light browser bar."""
from PIL import Image, ImageDraw, ImageFont
card = Image.open('../../data/qsmci_screenshot/matrix_subset.png').convert('RGB')
W = card.width; S = W / 900  # scale relative to CSS width of the card
BAR = int(64 * S); PAD = int(18 * S)
out = Image.new('RGB', (W + 2 * PAD, card.height + BAR + 2 * PAD), '#ffffff')
d = ImageDraw.Draw(out)
d.rounded_rectangle([0, 0, out.width - 1, out.height - 1], radius=int(14 * S), fill='#f3f4f6', outline='#d1d5db', width=max(2, int(1.5 * S)))
for i, c in enumerate(['#ff5f57', '#febc2e', '#28c840']):  # window dots
    cx, cy, r = int((22 + i * 18) * S), BAR // 2, int(6 * S)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
F = lambda sz, bold=False: ImageFont.truetype(f'fonts/DejaVuSans{"-Bold" if bold else ""}.ttf', int(sz * S))
ux0, ux1 = int(90 * S), out.width - int(24 * S)
d.rounded_rectangle([ux0, int(14 * S), ux1, BAR - int(14 * S)], radius=int(9 * S), fill='#ffffff', outline='#e5e7eb')
d.text((ux0 + int(14 * S), BAR // 2), 'qsmxt.github.io/QSM-CI', font=F(14, True), fill='#111827', anchor='lm')
tw = d.textlength('qsmxt.github.io/QSM-CI', font=F(14, True))
d.text((ux0 + int(14 * S) + tw + int(12 * S), BAR // 2), 'Results  ›  Harmonization (2026)  ›  Pipeline reproducibility',
       font=F(13), fill='#6b7280', anchor='lm')
out.paste(card, (PAD, BAR + PAD // 2))
out.save('../fig5_qsmci.png', dpi=(300, 300))
print(out.size)
