const { chromium } = require('playwright');
const ROWS = ['WH-QSM (QSM.rs)', 'HD-QSM (QSM.rs)', 'MEDI (QSM.rs)', 'iLSQR (STI Suite)', 'RTS (QSM.rs)', 'TKD (QSM.rs)', 'LPCNN (PyTorch)', 'QSMnet+ (TensorFlow)'];
const COLS = ['SHARP (QSM.rs)', 'V-SHARP (STI Suite)', 'PDF (QSM.rs)', 'RESHARP (QSM.rs)', 'LBV (QSM.rs)'];
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({viewport:{width:1440,height:1000}, colorScheme:'light', deviceScaleFactor:3});
  await ctx.addInitScript(() => { try { localStorage.setItem('qsmci-theme','light'); } catch(e){} });
  const p = await ctx.newPage();
  await p.goto('https://qsmxt.github.io/QSM-CI/results.html', {waitUntil:'networkidle', timeout:90000});
  await p.getByText('Harmonization (2026)', {exact:true}).first().click(); await p.waitForTimeout(3000);
  await p.getByText('Pipeline reproducibility', {exact:true}).first().click(); await p.waitForTimeout(6000);
  const sels = p.locator('select'); const n = await sels.count();
  for (let i = 0; i < n; i++) { const o = (await sels.nth(i).locator('option').allTextContents()).find(t => /ROMEO/.test(t));
    if (o && await sels.nth(i).isVisible()) { await sels.nth(i).selectOption({label: o}); break; } }
  await p.waitForTimeout(4000);
  await p.addStyleTag({content: 'header, nav, .sticky, [class*="sticky"], [class*="fixed"] {position: static !important}'});
  const res = await p.evaluate(({ROWS, COLS}) => {
    const lab = [...document.querySelectorAll('div')].find(e => e.children.length === 0 && e.textContent.trim() === 'WH-QSM (QSM.rs)');
    const grid = lab.parentElement.parentElement;
    const ncol = getComputedStyle(grid).gridTemplateColumns.split(' ').length;
    const kids = [...grid.children];
    // rows of the grid: either each child is a row wrapper, or the grid is flat (row-major cells)
    const start = kids.findIndex(c => c.textContent.trim() === 'SHARP (QSM.rs)') - 1;
    const rows = []; for (let i = start; i + ncol <= kids.length; i += ncol) rows.push(kids.slice(i, i + ncol));
    const header = rows[0];
    const hTexts = header.map(c => c.textContent.trim());
    const rowLab = (r) => { const leaf = [...r[0].querySelectorAll('*'), r[0]].find(e => e.children.length === 0 && e.textContent.trim()); return leaf ? leaf.textContent.trim() : ''; };
    const keepCols = [0, ...COLS.map(c => hTexts.indexOf(c)).filter(i => i > 0)];
    const dataRows = rows.filter(r => r !== header && r.length === header.length);
    const keepRows = ROWS.map(l => dataRows.find(r => rowLab(r) === l)).filter(Boolean);
    const nMoreCols = header.length - keepCols.length, nMoreRows = dataRows.length - keepRows.length;
    const ph = (txt, cls) => { const d = document.createElement('div'); d.textContent = txt;
      d.style.cssText = 'display:flex;align-items:center;justify-content:center;height:36px;border-radius:6px;border:1.5px dashed #d1d5db;color:#6b7280;font-size:12px;font-weight:600;background:#fafafa;' + (cls || ''); return d; };
    const first = grid.firstElementChild ? getComputedStyle(grid).gridTemplateColumns.split(' ')[0] : '160px';
    // keep original nodes: hide what isn't selected, insert placeholders in place
    kids.slice(0, Math.max(start, 0)).forEach(k => k.style.display = 'none');
    rows.forEach(r => { const keepRow = r === header || keepRows.includes(r);
      r.forEach((c, i) => { if (!keepRow || !keepCols.includes(i)) c.style.display = 'none'; }); });
    header.slice(1).forEach(h => h.querySelectorAll('*').forEach(x => { x.style.whiteSpace = 'normal'; x.style.overflow = 'visible'; x.style.textOverflow = 'clip'; }));
    header.forEach(h => { h.style.whiteSpace = 'normal'; h.style.overflow = 'visible'; h.style.textOverflow = 'clip'; h.style.textAlign = 'center'; });
    const lastOf = (r) => r[keepCols[keepCols.length - 1]];
    lastOf(header).after(ph(`+${nMoreCols} more`, 'border:none;background:none;height:auto;'));
    keepRows.forEach(r => lastOf(r).after(ph('…')));
    const tail = rows[rows.length - 1][ncol - 1];
    const lastLab = document.createElement('div'); lastLab.textContent = `+${nMoreRows} more`; lastLab.style.cssText = 'display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:12px;font-weight:700;color:#6b7280;height:36px;';
    const extra = [lastLab, ...keepCols.slice(1).map(() => ph('⋮')), ph('…')];
    let prev = tail; extra.forEach(e => { prev.after(e); prev = e; });
    // no in-cell accuracy bars: hide thin elements inside the heat cells
    grid.querySelectorAll('.heat-cell *').forEach(x => { const r = x.getBoundingClientRect(); if (r.height > 0 && r.height <= 5) x.style.display = 'none'; });
    grid.style.gridTemplateColumns = `${first} repeat(${keepCols.length}, minmax(0,1fr))`;
    grid.style.display = 'grid';
    // card: narrower, only title + one line + matrix + one footer line
    const card = grid.closest('.rounded-2xl');
    card.style.maxWidth = '900px';
    const block = (e) => { let s = e; while (s.parentElement && s.parentElement !== card) s = s.parentElement; return s; };
    const gridBlock = block(grid);
    const title = [...card.querySelectorAll('*')].find(e => e.children.length === 0 && e.textContent.trim() === 'Combination matrix');
    const titleBlock = block(title);
    [...card.children].forEach(c => { if (c !== gridBlock && c !== titleBlock) c.style.display = 'none'; });
    [...titleBlock.querySelectorAll('*')].forEach(e => { if (e !== title && !e.contains(title)) e.style.display = 'none'; });
    const sub = document.createElement('div');
    sub.textContent = 'Inter-scanner slope deviation |a − 1| for each pipeline: dipole inversion (rows) × background removal (columns), ROMEO field mapping.';
    sub.style.cssText = 'font-size:13px;color:#4b5563;margin-top:4px;';
    title.after(sub);
    if (gridBlock !== titleBlock) {
      let e = grid; while (e !== gridBlock) { [...e.parentElement.children].forEach(c => { if (c !== e) c.style.display = 'none'; }); e = e.parentElement; }
    }
    const foot = document.createElement('div');
    foot.innerHTML = `<span>Showing ${keepRows.length} of ${dataRows.length} dipole inversions × ${keepCols.length - 1} of ${header.length - 1} background removals · 638 pipelines in total on QSM-CI, including combined and end-to-end methods</span>
      <span style="display:inline-flex;align-items:center;gap:8px;margin-left:auto">better <span style="display:inline-block;width:90px;height:8px;border-radius:4px;background:linear-gradient(90deg,#b9dcc9,#e8dcc0,#e7b8bd)"></span> worse</span>`;
    foot.style.cssText = 'display:flex;align-items:center;gap:16px;font-size:12px;color:#6b7280;margin-top:14px;';
    grid.after(foot);
    const total = '';
    window.scrollTo(0, 0);
    return {hTexts, nRows: dataRows.length, kept: keepRows.map(rowLab), keptCols: keepCols.map(i => hTexts[i]), total, ncol};
  }, {ROWS, COLS});
  console.log(JSON.stringify(res));
  await p.waitForTimeout(800);
  const card = p.getByText('Combination matrix', {exact:true}).first().locator('xpath=ancestor::div[contains(@class,"rounded-2xl")][1]');
  await card.screenshot({path:'../../data/qsmci_screenshot/matrix_subset.png'});
  console.log('done'); await b.close(); })();
