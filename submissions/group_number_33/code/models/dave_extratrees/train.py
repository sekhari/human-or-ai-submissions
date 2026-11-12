"""Reproduce Dave's retained model from source, the frozen recipe, and raw CSVs.

No parameter search is performed. The historical 8000/2000 benchmark and final
10000-row training are separate fits with exactly the same model configuration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .model import CHAMPION_PARAMS, DaveExtraTreesClassifier
from ..text_generalization.style_search import FEATURE_NAMES


ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
RECIPE_FILE=PACKAGE/'recipe.json'
FEATURE_SOURCE=ROOT/'models/text_generalization/style_search.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indices_sha256(indices):
    """Digest row order, using compact UTF-8 JSON of Python integer indices."""
    payload=json.dumps([int(i) for i in indices],separators=(',',':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _ids_valid(frame):
    return (pd.api.types.is_integer_dtype(frame.id.dtype)
            and not pd.api.types.is_bool_dtype(frame.id.dtype)
            and frame.id.notna().all() and frame.id.is_unique)


def validate_sources(train,test,sample):
    if (len(train),len(test),len(sample))!=(10000,3000,3000):
        raise ValueError('Expected 10000 training and 3000 test/submission rows')
    for frame,columns in ((train,['id','text','label']),(test,['id','text']),(sample,['id','label'])):
        if list(frame.columns)!=columns or not _ids_valid(frame):
            raise ValueError('Unexpected source columns or nonunique/noninteger IDs')
    for frame in (train,test):
        if frame.text.isna().any() or not frame.text.map(lambda text:isinstance(text,str)).all():
            raise ValueError('Source texts must be nonmissing strings')
    if (not pd.api.types.is_integer_dtype(train.label.dtype)
            or train.label.isna().any() or set(train.label)!={0,1}
            or train.label.value_counts().to_dict()!={0:5000,1:5000}):
        raise ValueError('Expected 5000 human=0 and 5000 AI=1 training labels')
    if set(sample.id)!=set(test.id) or not set(train.id).isdisjoint(test.id):
        raise ValueError('Source IDs are inconsistent across files')


def validate_submission(submission,test,sample):
    if list(submission.columns)!=['id','label'] or len(submission)!=3000:
        raise ValueError('Submission must have 3000 rows and exactly id,label columns')
    if not _ids_valid(submission) or set(submission.id)!=set(test.id):
        raise ValueError('Submission IDs must match every test ID exactly once')
    if not np.array_equal(submission.id.to_numpy(),sample.id.to_numpy()):
        raise ValueError('Submission ID order must match sample_submission.csv')
    if not pd.api.types.is_numeric_dtype(submission.label.dtype):
        raise ValueError('Submission probabilities must be numeric')
    values=submission.label.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Submission probabilities must be finite and within [0,1]')


def load_recipe():
    recipe=json.loads(RECIPE_FILE.read_text())
    if recipe['model_params']!=CHAMPION_PARAMS:
        raise ValueError('Recipe does not match the retained 800-tree model')
    if recipe['feature_count']!=145 or len(FEATURE_NAMES)!=145:
        raise ValueError('Expected the retained 145-feature definition')
    if sha256(FEATURE_SOURCE)!=recipe['feature_source_sha256']:
        raise ValueError('The retained feature-extraction source has changed')
    split=recipe['split']
    if (split['test_size'],split['random_state'],split['stratify'])!=(.2,12,'label'):
        raise ValueError('The fixed stratified 8000/2000 split has changed')
    return recipe


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data',help='Directory containing train.csv, test.csv and sample_submission.csv')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dave_extratrees',help='Parent directory for a new timestamped run; existing runs are never overwritten')
    args=parser.parse_args(argv)
    recipe_bytes=RECIPE_FILE.read_bytes()
    recipe=load_recipe()
    recipe_hash=sha256(RECIPE_FILE)
    source_files=[PACKAGE/'__init__.py',PACKAGE/'model.py',Path(__file__),FEATURE_SOURCE]
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    data_dir=args.data_dir.resolve()
    data_hashes={name:sha256(data_dir/name) for name in ('train.csv','test.csv','sample_submission.csv')}
    if data_hashes!=recipe['data_sha256']:
        raise ValueError('CSV hashes differ from the original competition data; refusing to redefine the benchmark')
    train=pd.read_csv(data_dir/'train.csv')
    test=pd.read_csv(data_dir/'test.csv')
    sample=pd.read_csv(data_dir/'sample_submission.csv')
    validate_sources(train,test,sample)
    tr,va=train_test_split(np.arange(len(train)),test_size=.2,random_state=12,stratify=train.label)
    split_hashes=dict(development_indices_sha256=indices_sha256(tr),validation_indices_sha256=indices_sha256(va))
    if any(value!=recipe['split'][name] for name,value in split_hashes.items()):
        raise ValueError('Reproduced row order does not match the original fixed split')
    for indices,expected_rows in ((tr,8000),(va,2000)):
        counts=train.label.iloc[indices].value_counts().to_dict()
        if len(indices)!=expected_rows or counts!={0:expected_rows//2,1:expected_rows//2}:
            raise ValueError('Unexpected stratified partition sizes')
    if not set(tr).isdisjoint(va) or set(tr)|set(va)!=set(range(10000)):
        raise ValueError('Training and validation row partitions overlap or omit examples')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=args.output_dir.resolve()/stamp
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'recipe.json').write_bytes(recipe_bytes)
    (destination/'feature_names.json').write_text(json.dumps(list(FEATURE_NAMES),indent=2)+'\n')
    try:git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):git_head=None
    metrics=dict(created_utc=datetime.now(timezone.utc).isoformat(),git_head=git_head,
        recipe_sha256=recipe_hash,source_sha256=source_hashes,data_sha256=data_hashes,
        split=recipe['split'],reproduced_split_sha256=split_hashes,model_params=recipe['model_params'],
        feature_count=145,benchmark_training_rows=8000,benchmark_validation_rows=2000,final_training_rows=10000,
        validation_scope='Historical fixed 2000-row benchmark; reused in earlier development, not an untouched test.',
        external_or_test_labels_used=False,unlabeled_data_used=False,
        reference=recipe['reference'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    template=DaveExtraTreesClassifier(params=recipe['model_params'])
    with threadpool_limits(limits=2):
        started=perf_counter()
        benchmark=clone(template).fit(train.text.iloc[tr],train.label.iloc[tr])
        metrics['benchmark_fit_seconds']=perf_counter()-started
        validation_probability=benchmark.predict_proba(train.text.iloc[va])[:,1]
        metrics['validation_auroc']=float(roc_auc_score(train.label.iloc[va],validation_probability))
        metrics['validation_auroc_delta_vs_reference']=metrics['validation_auroc']-recipe['reference']['validation_auroc']
        metrics['train_auroc']=float(roc_auc_score(train.label.iloc[tr],benchmark.predict_proba(train.text.iloc[tr])[:,1]))
        pd.DataFrame(dict(id=train.id.iloc[va].to_numpy(),label=train.label.iloc[va].to_numpy(),
                          probability=validation_probability)).to_csv(destination/'validation_predictions.csv',index=False)
        print(f'Historical 8000/2000 benchmark AUROC: {metrics["validation_auroc"]:.9f}',flush=True)
        # Keep the recipe fixed regardless of the benchmark result.
        started=perf_counter()
        final=clone(template).fit(train.text,train.label)
        metrics['final_fit_seconds']=perf_counter()-started
        p=final.predict_proba(test.text)[:,1]
        submission=pd.DataFrame(dict(id=sample.id,label=sample.id.map(pd.Series(p,index=test.id))))
        validate_submission(submission,test,sample)
        submission_path=destination/'submission.csv'
        submission.to_csv(submission_path,index=False)
        readback=pd.read_csv(submission_path)
        validate_submission(readback,test,sample)
        np.testing.assert_allclose(readback.label,submission.label,rtol=0,atol=1e-15)
        joblib.dump(final,destination/'model.joblib',compress=3)
        reloaded=joblib.load(destination/'model.joblib')
        again=reloaded.predict_proba(test.text)[:,1]
        batches=np.concatenate([reloaded.predict_proba(test.text.iloc[i:i+257])[:,1] for i in range(0,len(test),257)])
        np.testing.assert_allclose(again,p,rtol=0,atol=1e-12)
        np.testing.assert_allclose(batches,p,rtol=0,atol=1e-12)
        for fitted,rows in ((benchmark,8000),(final,10000)):
            if fitted.n_training_rows_!=rows or fitted.classifier_._n_samples!=rows:
                raise ValueError('Unexpected model fit row count')
            if fitted.classifier_.n_features_in_!=145:
                raise ValueError('Unexpected fitted feature count')
            if fitted.classifier_.get_params()!=_expanded_params(recipe['model_params']):
                raise ValueError('Fitted forest parameters differ from frozen recipe')
        metrics.update(submission_file='submission.csv',submission_rows=len(submission),
            submission_sha256=sha256(submission_path),model_sha256=sha256(destination/'model.joblib'),
            reload_max_abs_error=float(np.max(np.abs(again-p))),batch_max_abs_error=float(np.max(np.abs(batches-p))),
            probability_min=float(p.min()),probability_max=float(p.max()))
    if sha256(RECIPE_FILE)!=recipe_hash or any(sha256(ROOT/name)!=value for name,value in source_hashes.items()):
        raise ValueError('Source or recipe changed while training')
    if any(sha256(data_dir/name)!=value for name,value in data_hashes.items()):
        raise ValueError('Input data changed while training')
    metrics['verification']='PASS: canonical data/split, schema/IDs/ranges, fit counts/parameters, reload and batch invariance'
    (destination/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(destination=str(destination),submission=str(submission_path),
                         validation_auroc=metrics['validation_auroc'],verification=metrics['verification']),indent=2),flush=True)
    return destination


def _expanded_params(params):
    """Expand explicit recipe settings with installed sklearn defaults for QA."""
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(**params).get_params()


if __name__=='__main__':main()
