"""Q5: prespecified punctuation ablations of the selected Q1 model.

No grid search or validation-based candidate selection. See manifest.json.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

SEED = 42
NAMES = ['baseline', 'normalize_quotes', 'word_only']
QUOTE_MAP = str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"'})


def texts_for(name, texts):
    # Stateless transformation; standard vectorizer lowercase behavior is preserved.
    return np.asarray([t.translate(QUOTE_MAP) for t in texts], dtype=object) if name == 'normalize_quotes' else np.asarray(texts, dtype=object)


def model_for(name, base):
    model = clone(base)
    if name == 'word_only':
        model.set_params(vec__token_pattern=r'(?u)\b\w+\b')
    return model


def load_data(train_path, ids_path):
    data = pd.read_csv(train_path, dtype={'id': str})
    ids = pd.read_csv(ids_path, dtype={'id': str}).id
    if not {'id','text','label'}.issubset(data):
        raise ValueError('Expected id, text, label columns.')
    if len(data) != 10000 or not data.id.is_unique or data[['id','text','label']].isna().any().any():
        raise ValueError('Expected 10,000 complete rows with unique IDs.')
    if data.label.value_counts().to_dict() != {0:5000,1:5000}:
        raise ValueError('Expected 5,000 human and 5,000 AI labels.')
    if len(ids) != 2000 or not ids.is_unique or not ids.isin(data.id).all():
        raise ValueError('Expected the saved 2,000 unique validation IDs.')
    mask = data.id.isin(set(ids))
    tr, va = data.loc[~mask].reset_index(drop=True), data.loc[mask].reset_index(drop=True)
    if va.label.value_counts().to_dict() != {0:1000,1:1000}:
        raise ValueError('Validation must contain 1,000 examples of each class.')
    assert len(tr) == 8000 and set(tr.id).isdisjoint(va.id)
    return tr, va


def fit(model, x, y):
    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        start = time.perf_counter()
        model.fit(x, y)
        duration = time.perf_counter() - start
    return duration


def proba(model, texts):
    return model.predict_proba(texts)[:, list(model.classes_).index(1)]


def paired_ci(y, baseline, alternative, repeats=2000):
    rng = np.random.default_rng(SEED)
    by_class = [np.flatnonzero(y == label) for label in (0,1)]
    differences = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in by_class])
        differences.append(roc_auc_score(y[sample], alternative[sample]) - roc_auc_score(y[sample], baseline[sample]))
    low, high = np.quantile(differences, [.025,.975])
    return {'repeats': repeats, 'method': 'paired class-stratified percentile bootstrap, seed 42',
            'delta_auroc_ci95_low': float(low), 'delta_auroc_ci95_high': float(high),
            'interpretation': 'Descriptive uncertainty; same validation set motivated Q4/Q5 development.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--val-ids', type=Path, required=True)
    ap.add_argument('--q1-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tr, va = load_data(args.train, args.val_ids)
    sys.path.insert(0, str(args.q1_dir.resolve()))
    spec = importlib.util.spec_from_file_location('q1_factory', args.q1_dir/'q1_model_comparison.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _, base, _ = module.build_models()['A_word_tfidf_lr']
    base.set_params(vec__ngram_range=(1,2), vec__min_df=5, clf__C=30)
    train_views = {name: texts_for(name, tr.text) for name in NAMES}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(tr, tr.label))
    fold_ids = tr[['id']].copy()
    for number, (_, valid) in enumerate(folds, 1): fold_ids.loc[valid, 'cv_fold'] = number
    fold_ids['cv_fold'] = fold_ids.cv_fold.astype(int)
    fold_ids.to_csv(args.out/'cv_fold_ids.csv',index=False)
    va[['id']].to_csv(args.out/'val_ids.csv',index=False)
    manifest = {
        'source_commit': 'db769ff154a6fb5c49d9cc7be355d887af7c9d63',
        'train_csv_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        'q1_model_code_sha256': hashlib.sha256((args.q1_dir/'q1_model_comparison.py').read_bytes()).hexdigest(),
        'validation_ids_sha256': hashlib.sha256('\n'.join(sorted(va.id)).encode()).hexdigest(),
        'seed': SEED, 'train_rows':len(tr), 'validation_rows':len(va),
        'candidates': {'baseline':'Selected Q1 Model A, unchanged',
            'normalize_quotes':'Map U+2018/U+2019 to ASCII apostrophe and U+201C/U+201D to ASCII double quote before unchanged vectorizer',
            'word_only':r'Original text; token_pattern=(?u)\b\w+\b; drop standalone punctuation, retain 1-character words'},
        'selection_rule':'Best proposed variant by mean training-only five-fold CV AUROC. Adopt only if it exceeds baseline mean CV AUROC; ties favor baseline.',
        'timing':'Median of 3 TF-IDF + classifier fits per candidate; candidate order rotates; excludes stateless quote mapping, scoring, CV and bootstrap; threadpool limit 2',
        'caution':'Q4 inspected this validation set; Q5 is exploratory follow-up, not independent confirmation.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threadpoolctl':threadpoolctl.__version__},
        'host':{'machine':platform.machine(),'system':platform.system(),'processor':platform.processor()},
    }
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cv_records = []
    with threadpool_limits(limits=2):
        for name in NAMES:
            for number, (train_index, val_index) in enumerate(folds,1):
                model = model_for(name, base)
                elapsed = fit(model, train_views[name][train_index], tr.label.to_numpy()[train_index])
                scores = proba(model, train_views[name][val_index])
                cv_records.append({'configuration':name,'fold':number,
                    'auroc':roc_auc_score(tr.label.to_numpy()[val_index],scores),'fit_seconds':elapsed})
            print(name, 'mean CV AUROC',np.mean([r['auroc'] for r in cv_records if r['configuration']==name]),flush=True)
        cv = pd.DataFrame(cv_records)
        cv.to_csv(args.out/'cv_fold_scores.csv',index=False)
        means = cv.groupby('configuration').auroc.mean().to_dict()
        best_variant = max(NAMES[1:], key=lambda name:means[name])
        selected = best_variant if means[best_variant] > means['baseline'] else 'baseline'
        selection = {'best_proposed_variant':best_variant,'selected_configuration':selected,
                     'cv_mean_auroc':means,'selected_before_validation_scoring':True}
        (args.out/'selection.json').write_text(json.dumps(selection,indent=2))
        print('Selection before validation:',selection,flush=True)

        models = {}; timings = []
        for repeat in range(3):
            order = NAMES[repeat:]+NAMES[:repeat]
            for position,name in enumerate(order,1):
                model = model_for(name,base)
                duration = fit(model,train_views[name],tr.label)
                timings.append({'configuration':name,'repeat':repeat+1,'position':position,'fit_seconds':duration})
                models[name] = model
        pd.DataFrame(timings).to_csv(args.out/'fit_timings.csv',index=False)
        results = []; predictions = va[['id','label']].copy()
        predictions['word_count'] = va.text.str.split().str.len()
        for name in NAMES:
            scores = proba(models[name],texts_for(name,va.text))
            predictions[name+'_p_ai'] = scores
            tn,fp,fn,tp = confusion_matrix(va.label,scores>=.5,labels=[0,1]).ravel()
            row = {'configuration':name,'cv_mean_auroc':means[name],
                'cv_std_auroc':float(cv.loc[cv.configuration==name,'auroc'].std(ddof=1)),
                'train_auroc':roc_auc_score(tr.label,proba(models[name],train_views[name])),
                'validation_auroc':roc_auc_score(va.label,scores),
                'median_fit_seconds':float(np.median([r['fit_seconds'] for r in timings if r['configuration']==name])),
                'vocabulary_size':len(models[name].named_steps['vec'].vocabulary_),
                'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
            results.append(row)
        predictions.to_csv(args.out/'validation_predictions.csv',index=False)
        table = pd.DataFrame(results)
        table['delta_validation_auroc'] = table.validation_auroc-table.loc[table.configuration=='baseline','validation_auroc'].iloc[0]
        table.to_csv(args.out/'comparison.csv',index=False)
        print(table.to_string(index=False),flush=True)
        ci = paired_ci(va.label.to_numpy(),predictions.baseline_p_ai.to_numpy(),predictions[best_variant+'_p_ai'].to_numpy())
        ci['comparison'] = best_variant+' minus baseline'
        (args.out/'paired_bootstrap.json').write_text(json.dumps(ci,indent=2))
        print('Descriptive bootstrap:',ci,flush=True)
        selected_ids = ['5234','4947','3671','7767','8238','3062','9490','1524','4600','473']
        predictions.loc[predictions.id.isin(selected_ids)].to_csv(args.out/'q4_error_followup.csv',index=False)
        subgroups = []
        for label,mask in [('<150',predictions.word_count<150),('150-249',predictions.word_count.between(150,249)),('>=250',predictions.word_count>=250)]:
            for name in NAMES:
                subgroups.append({'bin_words':label,'configuration':name,'count':int(mask.sum()),
                    'validation_auroc':roc_auc_score(predictions.loc[mask,'label'],predictions.loc[mask,name+'_p_ai'])})
        pd.DataFrame(subgroups).to_csv(args.out/'length_subgroups.csv',index=False)


if __name__ == '__main__': main()
