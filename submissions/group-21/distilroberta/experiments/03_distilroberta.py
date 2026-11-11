"""Distilroberta experiment."""

from pathlib import Path
import hashlib
import json
import random
import unicodedata
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

data_dir = Path(os.environ.get('DATA_DIR', 'data'))
out = Path(os.environ.get('OUT_DIR', 'outputs'))
run = out / 'distilroberta_v1'
run.mkdir(exist_ok=True, parents=True)
assert torch.cuda.is_available(), 'Choose a GPU runtime in Colab.'
train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
if not (out / 'val_ids.csv').exists():
    _, indices = train_test_split(np.arange(len(train)), test_size=2000,
        stratify=train.label, random_state=42)
    train[['id']].iloc[indices].to_csv(out / 'val_ids.csv', index=False)
val_ids = pd.read_csv(out / 'val_ids.csv')['id']
assert len(train) == 10000 and train.id.is_unique
assert train.label.value_counts().to_dict() == {0: 5000, 1: 5000}
assert len(val_ids) == 2000 and val_ids.is_unique and val_ids.isin(train.id).all()
assert len(test) == 3000 and set(test.id) == set(range(13001, 16001))
assert train.text.notna().all() and test.text.notna().all()
val = train.set_index('id').loc[val_ids].reset_index()
fit = train.loc[~train.id.isin(val_ids)].copy()
assert len(fit) == 8000 and set(fit.id).isdisjoint(val.id)
print('Train and validation counts', len(fit), len(val))

MODEL = 'distilbert/distilroberta-base'
SEED = 42
from huggingface_hub import model_info
revision_file = run / 'pretrained_revision.txt'
if revision_file.exists():
    revision = revision_file.read_text().strip()
else:
    revision = model_info(MODEL).sha
    revision_file.write_text(revision)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)

from distilroberta.features import Passages, ranking_margins

def train_or_load(frame, stage):
    set_seed(SEED)
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        frame[['id','text','label']], index=False).values.tobytes()).hexdigest()
    path = run / f'{stage}_{digest[:16]}'
    ready = (path / 'complete').is_file()
    model = (AutoModelForSequenceClassification.from_pretrained(path) if ready else
             AutoModelForSequenceClassification.from_pretrained(MODEL, revision=revision, num_labels=2))
    args = TrainingArguments(output_dir=str(run / 'trainer_tmp'), learning_rate=2e-5,
        num_train_epochs=2, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        fp16=True, seed=SEED, data_seed=SEED, save_strategy='no', report_to='none',
        logging_steps=50)
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer),
                      train_dataset=Passages(tokenizer, frame.text, frame.label) if not ready else None)
    if not ready:
        trainer.train()
        trainer.save_model(str(path))
        (path / 'complete').touch()
    if stage == 'final':
        tokenizer.save_pretrained(str(path))
        metadata = {'checkpoint': path.name, 'model': MODEL, 'revision': revision}
        pending = run / 'final_checkpoint.json.tmp'
        pending.write_text(json.dumps(metadata, indent=2))
        pending.replace(run / 'final_checkpoint.json')
    return trainer

def margins(trainer, texts):
    return ranking_margins(trainer, tokenizer, texts)

trainer = train_or_load(fit, 'validation')

def edit(text, passage_id, variant):
    if variant == 'original':
        return text
    if variant == 'lower_no_punct':
        return ''.join(c for c in text.lower() if not unicodedata.category(c).startswith('P'))
    rng = random.Random(f'{SEED}/{passage_id}/{variant}')
    words = text.split()
    if variant == 'delete_5pct':
        return ' '.join(w for w in words if rng.random() >= 0.05)
    if variant != 'clean_typos':
        raise ValueError(variant)
    for i, word in enumerate(words):
        swaps = [j for j in range(len(word)-1) if word[j:j+2].isalpha() and word[j] != word[j+1]]
        if swaps and rng.random() < 0.03:
            j = rng.choice(swaps)
            words[i] = word[:j] + word[j+1] + word[j] + word[j+2:]
    return ' '.join(words)

variants = ['original','clean_typos','delete_5pct','lower_no_punct']
rows = []
for variant in variants:
    edited = [edit(t, i, variant) for t, i in zip(val.text, val.id)]
    rows.append(pd.DataFrame(dict(id=val.id, label=val.label, variant=variant,
                                  text=edited, margin=margins(trainer, edited))))
probes = pd.concat(rows, ignore_index=True)
probes.to_csv(run / 'validation_probes.csv.gz', index=False)
aucs = {v: roc_auc_score(df.label, df.margin) for v, df in probes.groupby('variant')}
pd.Series({**aucs, 'worst': min(aucs.values())}).to_csv(run / 'validation_auc.csv', header=['AUROC'])
print(pd.Series(aucs).round(5), '\nWorst probe:', min(aucs.values()))

FINAL_FIT = os.environ.get("FINAL_FIT", "1") == "1"
if FINAL_FIT:
    import gc
    from scipy.special import expit
    if 'trainer' in globals():
        del trainer
    gc.collect()
    torch.cuda.empty_cache()
    final_model = train_or_load(train, 'final')
    test_scores = margins(final_model, test.text)
    submission = pd.DataFrame({'id': test.id, 'label': expit(test_scores)})
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    submission = submission.set_index('id').loc[sample.id].reset_index()
    assert submission.columns.tolist() == ['id','label']
    assert len(submission) == 3000 and submission.id.is_unique
    assert set(submission.id) == set(test.id) and submission.label.between(0, 1).all()
    submission.to_csv(run / 'submission_distilroberta.csv', index=False)
    print('Saved transformer candidate.', MODEL, revision)
