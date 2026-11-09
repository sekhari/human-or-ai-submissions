"""Export trusted fitted models and inference code without competition data/caches."""
import argparse,json,zipfile
from pathlib import Path

def package(directory,model,output):
    required=('model.joblib','training_complete.json','inference_check.json')
    for name in required:
        if not (model/name).is_file():raise ValueError(f'Model bundle is incomplete: {name}')
    if not json.loads((model/'inference_check.json').read_text()).get('passed'):raise ValueError('Inference check must pass before export.')
    if output.resolve().is_relative_to(model.resolve()):raise ValueError('Write the ZIP outside the model folder.')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path==output:continue
            relative=path.relative_to(directory)
            is_source=(path.parent==directory and path.suffix in ('.py','.txt','.md')) or (relative.parts[0]=='src' and path.suffix=='.py')
            is_evidence=relative.parts[0]=='robust_validation' and path.suffix=='.json'
            is_model=model in path.parents and path.suffix in ('.joblib','.json','.safetensors','.bin','.txt','.model')
            if path.name.startswith('_') and path.name!='__init__.py':continue
            if is_source or is_evidence or is_model:archive.write(path,relative)
    with zipfile.ZipFile(output) as archive:
        names=archive.namelist()
        for name in ('src/__init__.py','src/robust_ensemble.py','predict_robust.py',str(model.relative_to(directory)/'model.joblib')):
            if name not in names:raise AssertionError(f'Export omitted {name}')
    return {'bundle':str(output),'bytes':output.stat().st_size,'files':len(names)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(Path(__file__).resolve().parent,a.model.resolve(),a.output.resolve()),indent=2))

if __name__=='__main__':main()
