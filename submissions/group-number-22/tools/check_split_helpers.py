from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path.cwd()
private=root/'artifacts/02_split_metrics'
private.mkdir(parents=True,exist_ok=True)
source='''
from pathlib import Path
import tempfile, json
repo = Path.cwd()
before = {str(p): p.stat().st_mtime_ns for folder in ['configs', 'results'] for p in (repo / folder).rglob('*') if p.is_file()}
%run ./notebooks/02_split_metrics.ipynb
after = {str(p): p.stat().st_mtime_ns for folder in ['configs', 'results'] for p in (repo / folder).rglob('*') if p.is_file()}
assert before == after
assert 'parts' not in globals()
parts = load_partitions()
assert parts['train'].id.tolist() == parts['split']['train_ids']
assert parts['validation'].id.tolist() == parts['split']['validation_ids']
audit_path = ROOT / parts['split']['audit_path']
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / 'split.json'
    target.write_text('{}')
    try:
        freeze_split(audit_path, target=target)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed split accepted')
    assert target.read_text() == '{}'
    original_hash = file_hash
    def file_hash(path):
        if Path(path).name == 'train.csv':
            return 'changed'
        return original_hash(path)
    try:
        load_partitions()
    except ValueError:
        pass
    else:
        raise AssertionError('Changed raw data accepted')
    finally:
        file_hash = original_hash
print('Import had no file writes. Manifest row order passed. Changed split and raw bytes rejected.')
'''
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
try:
 NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
finally:
 nbformat.write(nb,private/'helper_checks.ipynb')
for out in nb.cells[0].outputs:
 if out.output_type=='stream': print(out.text)
