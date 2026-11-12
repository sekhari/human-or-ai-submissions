"""Group 15 Model C: raw-text style features + Random Forest.

Run from submissions/group15/: python experiments/model_c_style.py [--tune] [--submit]
No separate style_features.py is needed. Requires numpy, pandas, scikit-learn,
joblib and (for plots in error_analysis.py) matplotlib.

The supplied common.py and splits/val_ids.txt are required. Training/evaluation
uses common.evaluate(model, train_part, val_part, mode="raw") directly.
Validation predictions follow the exact line order in val_ids.txt.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Make common.py importable when launching from experiments/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common as TEAM_COMMON
SEED = int(TEAM_COMMON.SEED)
DEFAULT_DEPTH = None
DEFAULT_LEAF = 2


def load_csv(data, name):
    """Only use original data columns; id is never a predictor."""
    frame = pd.read_csv(Path(data) / name)
    if 'id' not in frame or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError(f'{name}: IDs must be present, unique and nonmissing')
    if 'text' in frame and (frame.text.isna().any() or frame.text.str.strip().eq('').any()):
        raise ValueError(f'{name}: missing/empty text')
    return frame


def load_split(train, path, seed=SEED):
    """Use the real team split; reorder validation explicitly by the ID file.

    common.get_split preserves input CSV row order rather than ID-file order.
    The provided CSV and sorted ID file agree, but explicit reordering also
    prevents silent misalignment if a later ID file has another order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing. Copy the real team validation file; it will never be reconstructed here.')
    raw = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(raw) != 2000 or len(set(raw)) != 2000:
        raise ValueError('val_ids.txt must have exactly 2,000 unique IDs, one per line, no header')
    lookup = dict(zip(train.id.astype(str), train.index))
    if not set(raw) <= lookup.keys():
        raise ValueError('Some validation IDs are not in train.csv')
    val = train.loc[[lookup[k] for k in raw]].reset_index(drop=True)
    fit = train.loc[~train.id.astype(str).isin(raw)].reset_index(drop=True)
    if len(fit) != 8000 or set(val.label.unique()) != {0,1}:
        raise ValueError('Expected 8,000 training rows and both labels in validation')
    common_checked = path.resolve()==Path(TEAM_COMMON.SPLIT_FILE).resolve()
    if common_checked:
        common_fit, common_val = TEAM_COMMON.get_split(train)
        if set(common_fit.id)!=set(fit.id) or set(common_val.id)!=set(val.id):
            raise ValueError('Public get_split and the supplied validation IDs disagree')
        # Match the team's training row order exactly, while keeping ID-file
        # order authoritative for saved validation predictions.
        fit = common_fit.reset_index(drop=True)
    origin = {'origin': 'actual_team_ID_file', 'seed':seed,
              'common_get_split_verified':common_checked,
              'validation_order':'exact line order in supplied val_ids.txt',
              'val_id_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return fit, val, origin


def repeated_fraction(items):
    return (len(items)-len(set(items))) / max(len(items),1)


WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?")

def one_text(text):
    """Preserve raw punctuation, case and newlines; lowercase word tokens only.

    Sentence boundaries are an approximation based on .!? and line breaks.
    Word-count bins use whitespace words, separately from regex word tokens.
    Repetition is (total - unique) / total for each sequence type.
    """
    text = str(text)
    tokens = WORD.findall(text)
    words = [word.lower().replace('’', "'") for word in tokens]
    lengths = np.array([len(word) for word in words], dtype=float)
    count, chars = len(words), max(len(text), 1)
    alpha = sum(c.isalpha() for c in text)
    sentences = [s for s in re.split(r'[.!?]+|\n+', text) if WORD.search(s)]
    sentence_lengths = [len(WORD.findall(s)) for s in sentences]
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    freq = Counter(words)
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    out = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'mean_word_length': float(lengths.mean()) if count else 0.,
        'std_word_length': float(lengths.std()) if count else 0.,
        'short_word_fraction': sum(len(w) <= 3 for w in words) / max(count, 1),
        'long_word_fraction': sum(len(w) >= 8 for w in words) / max(count, 1),
        'sentence_count': len(sentences),
        'mean_sentence_words': float(np.mean(sentence_lengths)) if sentences else 0.,
        'std_sentence_words': float(np.std(sentence_lengths)) if sentences else 0.,
        'max_sentence_words': max(sentence_lengths, default=0),
        'type_token_ratio': len(freq) / max(count, 1),
        'singleton_token_fraction': sum(v == 1 for v in freq.values()) / max(count, 1),
        'most_common_word_fraction': max(freq.values(), default=0) / max(count, 1),
        'repeated_bigram_fraction': repeated_fraction(bigrams),
        'repeated_trigram_fraction': repeated_fraction(trigrams),
        'digit_fraction': sum(c.isdigit() for c in text) / chars,
        'uppercase_alpha_fraction': sum(c.isupper() for c in text) / max(alpha, 1),
        'uppercase_word_fraction': sum(w.isupper() and len(w) > 1 for w in tokens) / max(count, 1),
        'whitespace_fraction': sum(c.isspace() for c in text) / chars,
        'newline_fraction': text.count('\n') / chars,
        'blank_line_fraction': len(re.findall(r'\n\s*\n', text)) / chars,
        'mean_line_words': float(np.mean([len(s.split()) for s in lines])) if lines else 0.,
        'repeated_line_fraction': repeated_fraction(lines),
        'contraction_fraction': sum("'" in w for w in words) / max(count, 1),
    }
    for name, symbols in {
        'comma': ',', 'period': '.', 'semicolon': ';', 'colon': ':',
        'question': '?', 'exclamation': '!', 'quote': '\"“”',
        'apostrophe': "'’", 'dash': '-–—', 'parenthesis': '()',
    }.items():
        out[name + '_fraction'] = sum(text.count(c) for c in symbols) / chars
    return out


def extract_style_features(texts):
    frame = pd.DataFrame([one_text(t) for t in texts])
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Non-finite style features')
    return frame



def make_model(depth=DEFAULT_DEPTH, leaf=DEFAULT_LEAF, seed=SEED):
    """A raw-text estimator compatible with the real common.evaluate API."""
    return Pipeline([
        ('style', FunctionTransformer(extract_style_features, validate=False)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=depth,
               min_samples_leaf=leaf, max_features='sqrt', random_state=seed, n_jobs=1)),
    ])


def save_predictions(root, split, frame, probabilities, labeled=False):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all():
        raise ValueError('Prediction array shape/values are invalid')
    if not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError('Expected P(AI) in [0,1]')
    target = root/'preds'; target.mkdir(parents=True, exist_ok=True)
    np.save(target/f'model_c_{split}.npy', probabilities, allow_pickle=False)
    np.save(target/f'model_c_{split}_ids.npy', frame.id.to_numpy(), allow_pickle=False)
    data = {'id': frame.id, 'p_ai': probabilities}
    if labeled:
        data['y_true'] = frame.label
    pd.DataFrame(data).to_csv(target/f'model_c_{split}.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=TEAM_COMMON.DATA_DIR)
    p.add_argument('--val-ids', default=TEAM_COMMON.SPLIT_FILE)
    p.add_argument('--tune', action='store_true', help='Compare six settings on the fixed team validation set')
    p.add_argument('--submit', action='store_true', help='Refit 10,000 rows and replace test probabilities with final-model predictions')
    a = p.parse_args()
    root = Path.cwd()
    for folder in ['preds','models','experiments']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    # Use the exact team loader; the unlabeled file is loaded but not used to fit C.
    train, _unlabeled, test = TEAM_COMMON.load_data(a.data)
    if len(train)!=10000 or len(test)!=3000 or set(train.label.unique())!={0,1}:
        raise ValueError('Unexpected official data size or labels')
    for name,frame in [('train.csv',train),('test.csv',test)]:
        if frame.id.isna().any() or frame.id.duplicated().any() or frame.text.isna().any() or frame.text.str.strip().eq('').any():
            raise ValueError(f'{name}: missing/duplicate IDs or empty texts')
    fit,val,origin=load_split(train,a.val_ids)
    grid=[(depth,leaf) for depth in [5,10,None] for leaf in [2,10]] if a.tune else [(DEFAULT_DEPTH,DEFAULT_LEAF)]
    records,best,best_auc,chosen=[],None,-np.inf,None
    for depth,leaf in grid:
        model=make_model(depth,leaf)
        # common.evaluate handles raw-mode preprocessing, fitting, scoring and time.
        result=TEAM_COMMON.evaluate(model,fit,val,mode='raw')
        vp=np.asarray(result['val_pred'],dtype=float)
        if vp.shape!=(len(val),) or not np.isfinite(vp).all():
            raise ValueError('common.evaluate returned invalid validation predictions')
        if not np.isclose(roc_auc_score(val.label,vp),result['val_auc'],rtol=0,atol=1e-12):
            raise ValueError('Public evaluator AUROC does not match returned validation probabilities')
        row={'max_depth':depth,'min_samples_leaf':leaf,'n_estimators':300,
             'train_auc':float(result['train_auc']),'val_auc':float(result['val_auc']),
             'fit_seconds':float(result['fit_seconds']),'evaluation_backend':'common.evaluate'}
        records.append(row);print(json.dumps(row),flush=True)
        if row['val_auc']>best_auc:
            best,best_auc,chosen=model,row['val_auc'],dict(row)
    raw=lambda frame:frame.text.map(lambda t:TEAM_COMMON.clean_text(t,mode='raw'))
    for name,frame in [('train',fit),('val',val),('test',test)]:
        save_predictions(root,name,frame,TEAM_COMMON.get_scores(best,raw(frame)),name!='test')
    rf=best.named_steps['rf'];params=rf.get_params()
    feature_names=list(extract_style_features(['']).columns)
    pd.DataFrame({'feature':feature_names,'importance':rf.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(root/'experiments/model_c_feature_importance.csv',index=False)
    # Save just the fitted forest plus its feature schema. This avoids pickling a
    # FunctionTransformer referencing __main__; the public fitting API is still a Pipeline.
    artifact={'model':rf,'feature_names':feature_names,'phase':'development_8000',
              'params':params,'seed':SEED,'text_mode':'raw','split':origin}
    joblib.dump(artifact,root/'models/model_c_dev.joblib',compress=3)
    if a.submit:
        final=make_model(params['max_depth'],params['min_samples_leaf'])
        final.fit(raw(train),train.label)
        testp=TEAM_COMMON.get_scores(final,raw(test))
        save_predictions(root,'test',test,testp)
        sample=load_csv(a.data,'sample_submission.csv')
        if list(sample.columns)!=['id','label'] or set(sample.id)!=set(test.id):
            raise ValueError('Sample submission IDs/schema do not match test')
        pd.DataFrame({'id':test.id,'label':testp}).to_csv(root/'submission_model_c.csv',index=False)
        final_artifact=dict(artifact,model=final.named_steps['rf'],phase='final_10000')
        joblib.dump(final_artifact,root/'models/model_c_final.joblib',compress=3)
    metadata={'seed':SEED,'common_available':True,'evaluation_backend':'common.evaluate',
              'common_sha256':hashlib.sha256(Path(TEAM_COMMON.__file__).read_bytes()).hexdigest(),
              'text_mode':'raw','split':origin,'train_rows':len(fit),'val_rows':len(val),'test_rows':len(test),
              'validation_order':'exact line order in actual team val_ids.txt',
              'test_order':'original row order in test.csv','test_fit_rows':10000 if a.submit else 8000,
              'chosen':chosen,'features':feature_names,
              'fit_time_definition':'common.evaluate: includes style feature extraction and forest fitting',
              'data_sha256':{'train':hashlib.sha256((Path(a.data)/'train.csv').read_bytes()).hexdigest(),
                             'test':hashlib.sha256((Path(a.data)/'test.csv').read_bytes()).hexdigest()},
              'limitations':['Validation is used for model selection, not an independent final test.'],
              'python':sys.version.split()[0],'numpy':np.__version__,'sklearn':__import__('sklearn').__version__}
    (root/'experiments/model_c_metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    table=pd.DataFrame(records)
    table.to_csv(root/'experiments/model_c_tuning.csv',index=False)
    (root/'experiments/model_c_tuning.txt').write_text(table.to_string(index=False)+'\n\n'+json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Done: {best_auc:.6f} validation AUROC; test fit size={metadata["test_fit_rows"]}; actual common.evaluate API verified.',flush=True)


if __name__=='__main__':
    main()
