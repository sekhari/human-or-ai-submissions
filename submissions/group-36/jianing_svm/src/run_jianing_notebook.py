"""Execute and save Jianing's notebook, preserving outputs even if a cell fails."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / '.jupyter/runtime'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / '.jupyter/ipython'))
# Download the pinned weights separately before offline execution.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
path = ROOT / 'notebooks/01_Jianing_TF_MiniLM_SVM.ipynb'
nb = nbformat.read(path, as_version=4)
def progress(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index + 1}/{len(nb.cells)}', flush=True)
client = NotebookClient(nb, timeout=3600, kernel_name='kaggle-jianing',
                        resources={'metadata': {'path': str(ROOT)}}, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Notebook execution finished and saved.', flush=True)
