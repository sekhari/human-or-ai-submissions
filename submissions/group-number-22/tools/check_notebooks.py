"""Check notebook files without running experiments."""
import ast
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
assert len(paths) == 20
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        assert "\u2014" not in cell.source
        if cell.cell_type == "code":
            ast.parse(cell.source)
print(f"Checked {len(paths)} notebooks. Experiments were not executed.")
