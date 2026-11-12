"""Fixed internal protocols and a common evaluator for text experiments."""
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/text_generalization'
PROTOCOL_FILE = OUT / 'protocol.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    protocol = json.loads(PROTOCOL_FILE.read_text())
    if sha256(ROOT / 'data/train.csv') != protocol['train_sha256']:
        raise ValueError('Training data changed since protocol freeze')
    raw = pd.read_csv(ROOT / 'data/train.csv')
    development = raw.iloc[protocol['development_indices']].reset_index(drop=True)
    return development, protocol


def decision_scores(model, texts):
    if hasattr(model, 'decision_function'):
        values = np.asarray(model.decision_function(texts), dtype=float)
    else:
        positive = int(np.flatnonzero(model.classes_ == 1).item())
        values = logit(np.clip(model.predict_proba(texts)[:, positive], 1e-7, 1-1e-7))
    if values.shape != (len(texts),) or not np.isfinite(values).all():
        raise ValueError('Invalid binary decision scores')
    return values


def run_candidates(family, configs, build_pipeline):
    """Fit on protocol training rows only; export heldout scores for fixed blends."""
    development, protocol = load_protocol()
    destination = OUT / family
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'frozen_candidates.json').write_text(json.dumps(configs, indent=2))
    cache = Memory(destination / '_cache', verbose=0)
    records = []
    with threadpool_limits(limits=2):
        for index, config in enumerate(configs):
            candidate = config.get('name', f'{family}_{index:02d}')
            record = dict(candidate=candidate, config=config)
            predictions = {}
            for split_name, split in protocol['protocols'].items():
                tr, va = np.asarray(split['train']), np.asarray(split['validation'])
                model = build_pipeline(config)
                if isinstance(model, Pipeline):
                    model.memory = cache
                start = perf_counter()
                model.fit(development.text.iloc[tr], development.label.iloc[tr])
                fit_seconds = perf_counter()-start
                train_score = decision_scores(model, development.text.iloc[tr])
                score = decision_scores(model, development.text.iloc[va])
                std = float(train_score.std())
                if std <= 0:
                    raise ValueError('Constant training decision scores')
                record[f'{split_name}_auroc'] = float(roc_auc_score(development.label.iloc[va], score))
                record[f'{split_name}_fit_seconds'] = fit_seconds
                record[f'{split_name}_train_score_mean'] = float(train_score.mean())
                record[f'{split_name}_train_score_std'] = std
                predictions[f'{split_name}_ids'] = development.id.iloc[va].to_numpy()
                predictions[f'{split_name}_labels'] = development.label.iloc[va].to_numpy()
                predictions[f'{split_name}_scores'] = score
                predictions[f'{split_name}_z'] = (score-train_score.mean())/std
                del model
                gc.collect()
            values = [record[f'{name}_auroc'] for name in protocol['protocols']]
            record['mean_auroc'] = float(np.mean(values))
            record['worst_auroc'] = float(min(values))
            records.append(record)
            np.savez_compressed(destination / f'{candidate}_predictions.npz', **predictions)
            (destination / 'results.json').write_text(json.dumps(records, indent=2, allow_nan=False))
            pd.DataFrame([{k:v for k,v in r.items() if k!='config'} for r in records]).to_csv(destination / 'scores.csv', index=False)
            print(f'{candidate}: mean={record["mean_auroc"]:.6f} worst={record["worst_auroc"]:.6f} '+
                  ' '.join(f'{name}={record[f"{name}_auroc"]:.6f}' for name in protocol['protocols']), flush=True)
    return records
