"""Analyze the published runner's frozen 8000/2000 validation benchmark.

No training or tuning is performed. Only train.csv and saved validation
predictions are read; the final model and test.csv are never loaded. Fixed
choices: P(AI) >= 0.5, regex-token bins <150/150-299/>=300, and five highest-
confidence errors per class, breaking probability ties by ascending ID.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .train import (
    FEATURE_SOURCE, PACKAGE, RECIPE_FILE, ROOT, _ids_valid,
    indices_sha256, load_recipe, sha256,
)
from ..text_generalization.style_search import TOKEN_RE, extract_style_features


THRESHOLD = 0.5
FEATURES = [
    'token_count', 'paragraph_count', 'line_count', 'blank_line_fraction',
    'markdown_list_line_fraction', 'markdown_heading_line_fraction',
    'code_fence_count', 'punctuation_char_fraction', 'digit_char_fraction',
    'word_type_token_ratio', 'sentence_tokens_mean', 'sentence_tokens_std',
    'token_length_mean', 'uppercase_char_fraction',
]
LENGTH_BINS = ['Short: <150 tokens', 'Medium: 150-299 tokens', 'Long: >=300 tokens']


def load_validation(run_dir, data_dir):
    """Check provenance and return exactly the saved validation rows and texts."""
    recipe = load_recipe()
    metrics_path = run_dir / 'metrics.json'
    predictions_path = run_dir / 'validation_predictions.csv'
    run_recipe_path = run_dir / 'recipe.json'
    data_path = data_dir / 'train.csv'
    metrics = json.loads(metrics_path.read_text())
    recipe_hash = sha256(RECIPE_FILE)
    if sha256(run_recipe_path) != recipe_hash or metrics['recipe_sha256'] != recipe_hash:
        raise ValueError('Run recipe and recorded recipe hash must match the current frozen recipe')
    if (metrics['model_params'] != recipe['model_params']
            or metrics['feature_count'] != 145
            or metrics['benchmark_training_rows'] != 8000
            or metrics['benchmark_validation_rows'] != 2000):
        raise ValueError('Run must describe the retained 145-feature 8000/2000 benchmark')
    source_files = [PACKAGE / '__init__.py', PACKAGE / 'model.py',
                    PACKAGE / 'train.py', FEATURE_SOURCE]
    for path in source_files:
        if metrics['source_sha256'].get(str(path.relative_to(ROOT))) != sha256(path):
            raise ValueError(f'Run source hash does not match current source: {path.name}')
    recorded_data = metrics['data_sha256']
    # The separate validation entrypoint reads only train.csv. The legacy
    # combined runner records all three files. Verify every recorded digest.
    if (sha256(data_path) != recipe['data_sha256']['train.csv']
            or recorded_data.get('train.csv') != recipe['data_sha256']['train.csv']
            or any(value != recipe['data_sha256'].get(name) for name, value in recorded_data.items())):
        raise ValueError('Expected the canonical competition train.csv and recorded data hashes')
    if metrics['split'] != recipe['split']:
        raise ValueError('Run split metadata differs from the fixed stratified seed-12 split')
    raw = pd.read_csv(data_path)
    predictions = pd.read_csv(predictions_path)
    if (list(predictions.columns) != ['id', 'label', 'probability']
            or not _ids_valid(predictions) or not _ids_valid(raw)):
        raise ValueError('Expected id,label,probability columns and unique integer IDs')
    train_idx, valid_idx = train_test_split(
        np.arange(len(raw)), test_size=.2, random_state=12, stratify=raw.label)
    if (len(train_idx), len(valid_idx), len(predictions)) != (8000, 2000, 2000):
        raise ValueError('Expected 8000 training rows and 2000 saved validation rows')
    split_hashes = dict(development_indices_sha256=indices_sha256(train_idx),
                        validation_indices_sha256=indices_sha256(valid_idx))
    if (any(value != recipe['split'][name] for name, value in split_hashes.items())
            or metrics['reproduced_split_sha256'] != split_hashes):
        raise ValueError('Validation row order does not reproduce the original fixed split')
    if (not np.array_equal(predictions.id, raw.id.iloc[valid_idx])
            or not np.array_equal(predictions.label, raw.label.iloc[valid_idx])):
        raise ValueError('Saved prediction IDs and labels must match the seed-12 validation order')
    if not pd.api.types.is_numeric_dtype(predictions.probability.dtype):
        raise ValueError('Saved probabilities must be numeric')
    probability = predictions.probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Saved probabilities must be finite and within [0,1]')
    auc = float(roc_auc_score(predictions.label, probability))
    if not np.isclose(auc, metrics['validation_auroc'], rtol=0, atol=1e-12):
        raise ValueError('Recomputed validation AUROC does not match run metrics')
    train_auc = metrics['train_auroc']
    if not np.isfinite(train_auc) or not 0 <= train_auc <= 1:
        raise ValueError('Saved training AUROC must be finite and within [0,1]')
    # The verified ID order makes positional text attachment unambiguous.
    frame = predictions.copy()
    frame['text'] = raw.text.iloc[valid_idx].to_numpy()
    paths = [data_path, metrics_path, predictions_path, run_recipe_path, RECIPE_FILE,
             *source_files, Path(__file__).resolve()]
    sources = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p)
               for p in paths}
    return frame, recipe, metrics, auc, sources


def analyze(frame, recipe, metrics, auc, sources):
    """Apply the original Q4 choices without changing the fitted model."""
    frame['predicted_label'] = (frame.probability >= THRESHOLD).astype(int)
    frame['error_type'] = np.select([
        (frame.label == 0) & (frame.predicted_label == 1),
        (frame.label == 1) & (frame.predicted_label == 0)], ['FP', 'FN'], default='correct')
    features = pd.DataFrame([extract_style_features(text) for text in frame.text])
    if features.shape != (2000, 145) or not np.isfinite(features.to_numpy()).all():
        raise ValueError('Expected 145 finite original style features for each validation row')
    for column in FEATURES:
        frame[column] = features[column].to_numpy()
    frame['token_count'] = frame.token_count.astype(int)
    if not np.array_equal(frame.token_count, frame.text.map(lambda text: len(TOKEN_RE.findall(text)))):
        raise ValueError('Feature token counts do not match the original TOKEN_RE')
    frame['length_bin'] = pd.cut(frame.token_count, bins=[-1, 149, 299, np.inf], labels=LENGTH_BINS)
    bins = []
    for name, group in frame.groupby('length_bin', observed=False, sort=True):
        if set(group.label) != {0, 1}:
            raise ValueError(f'Length-bin AUROC requires both classes: {name}')
        bins.append(dict(
            bin=str(name), count=len(group), human=int((group.label == 0).sum()),
            ai=int((group.label == 1).sum()),
            validation_auroc=float(roc_auc_score(group.label, group.probability)),
            false_positive=int((group.error_type == 'FP').sum()),
            false_negative=int((group.error_type == 'FN').sum())))
    if sum(group['count'] for group in bins) != 2000:
        raise ValueError('Length bins must cover all 2000 validation rows')
    fp = frame[frame.error_type == 'FP'].sort_values(['probability', 'id'], ascending=[False, True]).head(5)
    fn = frame[frame.error_type == 'FN'].sort_values(['probability', 'id'], ascending=[True, True]).head(5)
    if len(fp) != 5 or len(fn) != 5:
        raise ValueError('The fixed Q4 selection requires at least five FP and five FN examples')
    examples = pd.concat([fp, fn], ignore_index=True)
    patterns = []
    for label in (0, 1):
        for status in ('correct', 'FP' if label == 0 else 'FN'):
            group = frame[(frame.label == label) & (frame.error_type == status)]
            if group.empty:
                raise ValueError(f'Pattern medians require a nonempty group: label={label}, status={status}')
            patterns.append(dict(
                label=label, status=status, count=len(group),
                with_blank_lines=int((group.blank_line_fraction > 0).sum()),
                with_multiple_paragraphs=int((group.paragraph_count > 1).sum()),
                with_multiple_lines=int((group.line_count > 1).sum()),
                with_markdown_lists=int((group.markdown_list_line_fraction > 0).sum()),
                medians={key: float(group[key].median()) for key in FEATURES}))
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(), model=recipe['model_name'],
        model_params=recipe['model_params'], feature_count=145,
        train_auroc=metrics['train_auroc'], validation_auroc=auc,
        auroc_gap=metrics['train_auroc'] - auc, training_rows=8000, validation_rows=2000,
        train_auroc_source='Saved benchmark metrics; not recomputed or independently refitted by this analysis.',
        threshold=THRESHOLD, tokenization=TOKEN_RE.pattern, length_bins=bins,
        confusion_matrix=confusion_matrix(frame.label, frame.predicted_label, labels=[0, 1]).tolist(),
        confusion_matrix_order='Rows true label, columns predicted label: 0=human, 1=AI',
        error_counts={key: int((frame.error_type == key).sum()) for key in ['FP', 'FN', 'correct']},
        error_selection='Five highest P(AI) FP and five lowest P(AI) FN, ID ascending for ties',
        examples=examples[['id', 'label', 'probability', 'predicted_label', 'error_type', *FEATURES]].to_dict(orient='records'),
        observed_pattern_counts=patterns, sources=sources,
        scope='Frozen original 8000/2000 development split. Previously reused validation, not pristine independent test. '
              'No refit or tuning in this analysis. Bins/threshold fixed before subgroup scoring; highest-confidence '
              'errors are illustrative, not representative random samples. Original text is not exported.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Published training run containing metrics.json, recipe.json and validation_predictions.csv')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data',
                        help='Directory containing the canonical train.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/dave_extratrees/analysis',
                        help='Parent directory for a new UTC timestamped analysis; existing runs are never overwritten')
    args = parser.parse_args(argv)
    report = analyze(*load_validation(args.run_dir.resolve(), args.data_dir.resolve()))
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = args.output_dir.resolve() / stamp
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / 'analysis.json'
    output.write_text(payload)
    print(json.dumps(dict(analysis=str(output), validation_auroc=report['validation_auroc'],
                          error_counts=report['error_counts'], selected_examples=len(report['examples']))))
    return destination


if __name__ == '__main__':
    main()
