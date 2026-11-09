"""Build and execute followup.ipynb from analysis/followup_walkthrough.py (python analysis/build_followup_notebook.py).

The source is a plain script in "percent" format: `# %%` starts a code cell, `# %% [markdown]` a markdown cell
whose lines are commented with `# `.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'analysis' / 'followup_walkthrough.py'


def cells(text):
    out, kind, buf = [], None, []

    def flush():
        body = '\n'.join(buf).strip('\n')
        if kind == 'markdown':
            out.append(nbf.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())))
        elif kind == 'code' and body:
            out.append(nbf.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()
    return out


nb = nbf.v4.new_notebook(cells=cells(SRC.read_text()))
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
nbf.write(nb, ROOT / 'followup.ipynb')
print('wrote followup.ipynb')
