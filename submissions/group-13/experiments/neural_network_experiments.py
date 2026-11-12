r"""Train and explain two supervised bidirectional-GRU text classifiers.

Run in the existing Windows environment:
    .\.venv\Scripts\python.exe neural_network_experiments.py
Quick implementation checks, without fitting the dataset:
    .\.venv\Scripts\python.exe neural_network_experiments.py --check-only

EXPERIMENT QUESTION
Does learned attention pooling improve on uniform mean pooling when everything
else (tokenizer, split, dimensions, optimizer, seed, batch order) is held fixed?
Train BOTH versions at seeds 42, 43, 44. The data split always remains seed 42.
These are two variants of one model family, not six different architectures.

ARCHITECTURE
words + individual punctuation -> integer IDs -> trainable 128-D embedding
-> embedding dropout 0.1 -> one bidirectional GRU, 64 units per direction
-> 128-D contextual vector at each position -> mean OR attention pooling
-> Linear(128,64) -> ReLU -> dropout 0.3 -> Linear(64,1) -> raw logit.
Sigmoid is applied only for prediction; BCEWithLogitsLoss already includes it.
No autoencoder, pretrained embeddings, external labels, or Transformer is used.

Attention: k_t=tanh(W h_t+b), s_t=q^T k_t, alpha=softmax(s), c=sum(alpha*h).
W maps 128->64; q is one learned 64-D query shared by all passages. h_t is
the 128-D VALUE VECTOR, whereas alpha_t is a scalar mixture weight. Attention
weights are not word-level AI probabilities or causal feature explanations.
Mean pooling uses alpha_t=1/length for each valid position.

DATA AND PADDING
Only train.csv is read from the original zip. Reuse the exact saved 8000/2000
split. Learn a lowercase punctuation-preserving vocabulary from training only,
requiring document frequency >=2. PAD=0; UNK=1. Keep at most 768 tokens, chosen
from the prior training-only length audit; report any truncation in each split.
Right-pad each batch dynamically; pack sequences so neither GRU direction
processes padding; mask padding out of BOTH pooling methods.

TRAINING
AdamW lr=.001, weight_decay=.0001; batch=32; gradient-norm clipping=1;
maximum 20 epochs; stop after 3 epochs without higher validation AUROC.
Save the highest-AUROC epoch, not simply the last epoch. Balanced labels need
no class weights. Use full precision, fixed seeds, and deterministic PyTorch
operations. Exact reproducibility across hardware/versions is not guaranteed.
Shared layers start identically for matched seeds; attention adds parameters.
Model initialization and training/dropout RNG streams are separated so adding
the attention layer does not perturb initial shared weights or batch order.

ANALYSIS
Save every epoch and every seed, aggregate mean and sample standard deviation
over seeds, and compare paired predictions with the saved LR/SVM results.
Bootstrap intervals condition on trained models and the validation set; they
do not include checkpoint/architecture selection bias or training variation.
Three seeds provide a limited view of training variability. Validation is a
development set, not an untouched test set. Report train-vs-validation gaps,
length subgroups, 5 false positives/5 false negatives at probability .5, and
token attention examples selected by ID rather than by attractive weights.

The script automatically uses CUDA/ROCm when torch.cuda.is_available(), else
CPU. Explicit --device cuda fails clearly if unavailable. DirectML is not used:
the installed Windows DirectML backend failed the packed GRU forward/backward
probe. ROCm uses PyTorch's 'cuda' device name even on AMD GPUs.

Outputs are in results_neural/. Checkpoints contain state_dicts and config;
vocabulary.json is required for text inference. Completed runs can be reused
with --resume; a partial run restarts from its seed. No test predictions,
all-label refit, Kaggle upload, or unlabeled-data experiment is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

import logistic_regression_experiments as baseline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
TOKENIZER = re.compile(TOKEN_PATTERN)
SETTINGS = dict(embedding_dim=128, hidden_size=64, attention_dim=64,
                classifier_dim=64, embedding_dropout=0.1, classifier_dropout=0.3,
                min_document_frequency=2, max_tokens=768, batch_size=32,
                learning_rate=0.001, weight_decay=0.0001, gradient_clip_norm=1.0,
                max_epochs=20, patience=3)


def tokenize(text):
    return TOKENIZER.findall(text.lower())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BiGRUClassifier(nn.Module):
    """The two versions differ only in how contextual token vectors are pooled."""

    def __init__(self, vocabulary_size, pooling, config):
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(pooling)
        self.pooling = pooling
        self.embedding = nn.Embedding(vocabulary_size, config["embedding_dim"], padding_idx=0)
        self.embedding_dropout = nn.Dropout(config["embedding_dropout"])
        self.gru = nn.GRU(config["embedding_dim"], config["hidden_size"],
                          num_layers=1, batch_first=True, bidirectional=True)
        width = 2 * config["hidden_size"]
        # Initialize all SHARED layers before attention, to match them by seed.
        self.classifier = nn.Sequential(nn.Linear(width, config["classifier_dim"]),
                                        nn.ReLU(), nn.Dropout(config["classifier_dropout"]),
                                        nn.Linear(config["classifier_dim"], 1))
        if pooling == "attention":
            self.key = nn.Linear(width, config["attention_dim"])
            self.query = nn.Linear(config["attention_dim"], 1, bias=False)

    def forward(self, token_ids, lengths, return_attention=False):
        # token_ids: [batch, padded_length]; lengths always stays on CPU.
        embedded = self.embedding_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_h, _ = self.gru(packed)
        contextual, _ = pad_packed_sequence(packed_h, batch_first=True,
                                            total_length=token_ids.shape[1])
        valid = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :] < lengths.to(token_ids.device)[:, None]
        if self.pooling == "attention":
            keys = torch.tanh(self.key(contextual))          # [B,L,64]
            scores = self.query(keys).squeeze(-1)           # [B,L]
            scores = scores.masked_fill(~valid, -torch.inf)
            weights = torch.softmax(scores, dim=1)
        else:
            weights = valid.to(contextual.dtype) / lengths.to(token_ids.device)[:, None]
        passage = torch.sum(contextual * weights.unsqueeze(-1), dim=1)  # [B,128]
        logits = self.classifier(passage).squeeze(-1)        # [B]
        return (logits, weights) if return_attention else logits


def batch_at(sequences, labels, indices, device):
    rows = [sequences[int(i)] for i in indices]
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    tokens = pad_sequence(rows, batch_first=True, padding_value=0).to(device)
    targets = torch.tensor(labels[np.asarray(indices)], dtype=torch.float32, device=device)
    return tokens, lengths, targets


@torch.no_grad()
def predict(model, sequences, labels, device, batch_size=64):
    model.eval()
    chunks = []
    for start in range(0, len(sequences), batch_size):
        tokens, lengths, _ = batch_at(sequences, labels, np.arange(start, min(start + batch_size, len(sequences))), device)
        chunks.append(model(tokens, lengths).cpu().numpy().astype(np.float64))
    logits = np.concatenate(chunks)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite model predictions")
    return logits


def score(labels, logits):
    probabilities = expit(logits)
    return dict(auc=float(roc_auc_score(labels, logits)),
                loss=float(log_loss(labels, probabilities, labels=[0, 1])),
                brier=float(brier_score_loss(labels, probabilities)),
                accuracy=float(accuracy_score(labels, probabilities >= 0.5)))


def check_model(device):
    """Meaningful invariants: shared initialization, padding, masks, gradients."""
    models = {}
    for pooling in ("mean", "attention"):
        seed_everything(123)
        models[pooling] = BiGRUClassifier(30, pooling, SETTINGS).to(device)
    mean_state = models["mean"].state_dict()
    attention_state = models["attention"].state_dict()
    assert all(torch.equal(value, attention_state[name]) for name, value in mean_state.items())
    tokens = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 0, 0]], device=device)
    lengths = torch.tensor([5, 3])
    for model in models.values():
        model.eval()
        logits, weights = model(tokens, lengths, True)
        assert tuple(logits.shape) == (2,) and tuple(weights.shape) == (2, 5)
        torch.testing.assert_close(weights.sum(1), torch.ones(2, device=device))
        assert torch.count_nonzero(weights[1, 3:]).item() == 0
        extended = nn.functional.pad(tokens, (0, 9), value=0)
        torch.testing.assert_close(logits, model(extended, lengths), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(logits[1:], model(tokens[1:2, :3], lengths[1:]), atol=2e-6, rtol=2e-6)
        if model.pooling == "mean":
            torch.testing.assert_close(weights[0], torch.full((5,), 0.2, device=device))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.], device=device))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        assert torch.count_nonzero(model.embedding.weight.grad[0]).item() == 0
        if model.pooling == "attention":
            assert model.query.weight.grad.abs().sum().item() > 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        optimizer.step()
    assert tokenize("Hi—there! A-b.") == ["hi", "—", "there", "!", "a", "-", "b", "."]
    print("PASS: shared initialization, punctuation, shape, padding invariance, normalized/masked attention, backward gradients and optimizer step.", flush=True)


def prepare_data(args, config, write_outputs=True):
    train, validation, assignments = baseline.load_and_split(args.zip, 42)
    references = {}
    for directory in ("results", "results_punctuation", "results_svm"):
        path = baseline.PROJECT_DIR / directory
        pd.testing.assert_frame_equal(assignments, pd.read_csv(path / "split_assignments.csv"))
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["source_zip_sha256"] == baseline.sha256(args.zip)
        references[directory] = {file.name: baseline.sha256(file) for file in path.iterdir() if file.is_file()}
    token_lists = {"train": [tokenize(text) for text in train.text],
                   "validation": [tokenize(text) for text in validation.text]}
    counts = Counter(token for row in token_lists["train"] for token in set(row))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    vocabulary.update({token: index + 2 for index, token in enumerate(sorted(
        token for token, count in counts.items() if count >= config["min_document_frequency"]))})
    sequences, audit = {}, {}
    for name, rows in token_lists.items():
        lengths = np.array([len(row) for row in rows])
        assert (lengths > 0).all()
        kept = [row[:config["max_tokens"]] for row in rows]
        sequences[name] = [torch.tensor([vocabulary.get(token, 1) for token in row], dtype=torch.long) for row in kept]
        audit[name] = dict(rows=len(rows), minimum_tokens=int(lengths.min()), median_tokens=float(np.median(lengths)),
                           maximum_tokens=int(lengths.max()), truncated_rows=int((lengths > config["max_tokens"]).sum()),
                           removed_tokens=int(np.maximum(lengths-config["max_tokens"], 0).sum()),
                           unknown_tokens=sum(token not in vocabulary for row in kept for token in row),
                           retained_tokens=sum(len(row) for row in kept))
        audit[name]["unknown_token_fraction"] = audit[name]["unknown_tokens"] / audit[name]["retained_tokens"]
    if write_outputs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)
        (args.output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified 8,000/2,000 split; vocabulary={len(vocabulary):,} including PAD/UNK. Token audit: {audit}", flush=True)
    return train, validation, sequences, vocabulary, audit, references, token_lists


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def fit_one(pooling, seed, config, vocabulary, sequences, y_train, y_val, device, output, fingerprint, resume):
    name = f"bigru_{pooling}_seed{seed}"
    run_dir = output / name
    run_dir.mkdir(exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    summary_path = run_dir / "summary.json"
    if resume and summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Cannot resume {name}: configuration/source/data changed; use a fresh output directory.")
        print(f"Reusing completed run {name}", flush=True)
        return saved
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists; use --resume or a new output directory.")
    seed_everything(seed)
    model = BiGRUClassifier(len(vocabulary), pooling, config).to(device)
    # Reset after initialization so adding attention does not shift dropout RNG.
    seed_everything(seed + 10000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"], foreach=False)
    loss_function = nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(seed + 20000)
    records, best_auc, best_epoch, stale = [], -np.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        order = generator.permutation(len(y_train))
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            tokens, lengths, targets = batch_at(sequences["train"], y_train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss in {name}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True, foreach=False)
            optimizer.step()
            total_loss += loss.detach().item() * len(indices)
            if (start // config["batch_size"] + 1) % 100 == 0:
                print(f"  {name} epoch {epoch}: {start + len(indices)}/{len(order)} training passages", flush=True)
        val_metrics = score(y_val, predict(model, sequences["validation"], y_val, device))
        record = dict(epoch=epoch, online_training_loss=total_loss / len(y_train),
                      validation_auc=val_metrics["auc"], validation_loss=val_metrics["loss"],
                      epoch_seconds=time.perf_counter() - epoch_start)
        records.append(record)
        pd.DataFrame(records).to_csv(run_dir / "history.csv", index=False)
        improved = val_metrics["auc"] > best_auc
        if improved:
            best_auc, best_epoch, stale = val_metrics["auc"], epoch, 0
            torch.save(dict(state_dict=cpu_state(model), config=config, pooling=pooling,
                            vocabulary_size=len(vocabulary), seed=seed, epoch=epoch,
                            validation_auc=best_auc, fingerprint=fingerprint), checkpoint_path)
        else:
            stale += 1
        print(f"{name} epoch {epoch:02}: train loss={record['online_training_loss']:.4f}, "
              f"val loss={val_metrics['loss']:.4f}, val AUC={val_metrics['auc']:.6f}, "
              f"seconds={record['epoch_seconds']:.1f}{' BEST' if improved else ''}", flush=True)
        if stale >= config["patience"]:
            break
    fit_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    train_logits = predict(model, sequences["train"], y_train, device)
    validation_logits = predict(model, sequences["validation"], y_val, device)
    train_metrics, val_metrics = score(y_train, train_logits), score(y_val, validation_logits)
    assert abs(val_metrics["auc"] - best_auc) < 1e-10
    np.savez_compressed(run_dir / "logits.npz", train=train_logits, validation=validation_logits)
    # Reload a second instance, not just the existing object, and verify outputs.
    reloaded = BiGRUClassifier(len(vocabulary), pooling, config).to(device)
    reloaded.load_state_dict(checkpoint["state_dict"])
    check_logits = predict(reloaded, sequences["validation"][:96], y_val[:96], device)
    np.testing.assert_allclose(check_logits, validation_logits[:96], atol=3e-6, rtol=3e-6)
    result = dict(experiment=name, pooling=pooling, seed=seed, best_epoch=best_epoch,
                  epochs_run=len(records), early_stopped=len(records) < config["max_epochs"],
                  parameters=sum(p.numel() for p in model.parameters()), train_auc=train_metrics["auc"],
                  validation_auc=val_metrics["auc"], auc_gap=train_metrics["auc"]-val_metrics["auc"],
                  train_loss=train_metrics["loss"], validation_loss=val_metrics["loss"],
                  validation_brier=val_metrics["brier"], validation_accuracy_at_05=val_metrics["accuracy"],
                  fit_seconds=fit_seconds, fingerprint=fingerprint, saved_model_reload_verified=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FINISHED {name}: best epoch {best_epoch}, train AUC {train_metrics['auc']:.6f}, val AUC {best_auc:.6f}", flush=True)
    return result


def analysis_outputs(metrics, train, validation, sequences, vocabulary, token_lists, device, args, config):
    output = args.output_dir
    y_val = validation.label.to_numpy()
    predictions = validation[["id", "label"]].reset_index(drop=True)
    train_predictions = train[["id", "label"]].reset_index(drop=True)
    score_columns, score_matrix = [], []
    for row in metrics.itertuples():
        with np.load(output / row.experiment / "logits.npz") as saved:
            logits = saved["validation"]
            predictions[row.experiment] = expit(logits)
            train_predictions[row.experiment] = expit(saved["train"])
            score_columns.append(row.experiment)
            score_matrix.append(logits)
            assert np.isclose(roc_auc_score(y_val, logits), row.validation_auc, atol=1e-12)
    references = [
        ("logistic_words", "results", "validation_predictions.csv", "unigrams_bigrams_C10"),
        ("logistic_punctuation", "results_punctuation", "validation_predictions.csv", "unigrams_bigrams_C10"),
        ("svm_punctuation", "results_svm", "validation_decision_scores.csv", "punctuation__unigrams_bigrams_C10")]
    for name, directory, filename, column in references:
        reference = pd.read_csv(baseline.PROJECT_DIR / directory / filename).set_index("id").loc[validation.id]
        assert np.array_equal(reference.label.to_numpy(), y_val)
        score_columns.append(name)
        score_matrix.append(reference[column].to_numpy())
    draws = baseline.paired_bootstrap(y_val, np.column_stack(score_matrix), args.bootstrap_repetitions, 43)
    for index in metrics.index:
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        metrics.loc[index, "auc_ci_low"] = low
        metrics.loc[index, "auc_ci_high"] = high
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    train_predictions.to_csv(output / "training_predictions.csv", index=False)
    summary = metrics.groupby("pooling", sort=False).agg(
        seeds=("seed", "count"), validation_auc_mean=("validation_auc", "mean"),
        validation_auc_sd=("validation_auc", "std"), validation_auc_min=("validation_auc", "min"),
        validation_auc_max=("validation_auc", "max"), train_auc_mean=("train_auc", "mean"),
        validation_loss_mean=("validation_loss", "mean"), fit_seconds_mean=("fit_seconds", "mean")).reset_index()
    summary.to_csv(output / "architecture_summary.csv", index=False)
    lookup = {name: i for i, name in enumerate(score_columns)}
    comparisons = []
    for seed in args.seeds:
        a, b = f"bigru_attention_seed{seed}", f"bigru_mean_seed{seed}"
        if a not in lookup or b not in lookup:
            continue
        difference_draws = draws[:, lookup[a]] - draws[:, lookup[b]]
        comparisons.append(dict(model_a=a, model_b=b,
                                difference=roc_auc_score(y_val, score_matrix[lookup[a]])-roc_auc_score(y_val, score_matrix[lookup[b]]),
                                ci_low=float(np.quantile(difference_draws, .025)), ci_high=float(np.quantile(difference_draws, .975))))
    # Average the PER-SEED AUCs; this is not an ensemble of predictions.
    family_draws, family_auc = {}, {}
    for pooling in args.pooling:
        indices = [lookup[f"bigru_{pooling}_seed{seed}"] for seed in args.seeds]
        family_draws[pooling] = draws[:, indices].mean(axis=1)
        family_auc[pooling] = metrics.loc[metrics.pooling.eq(pooling), "validation_auc"].mean()
        for reference_name, *_ in references:
            d = family_draws[pooling] - draws[:, lookup[reference_name]]
            comparisons.append(dict(model_a=f"{pooling}_mean_AUC_over_seeds", model_b=reference_name,
                                    difference=family_auc[pooling]-roc_auc_score(y_val, score_matrix[lookup[reference_name]]),
                                    ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975))))
    if len(family_draws) == 2:
        d = family_draws["attention"] - family_draws["mean"]
        comparisons.append(dict(model_a="attention_mean_AUC_over_seeds", model_b="mean_mean_AUC_over_seeds",
                                difference=family_auc["attention"]-family_auc["mean"],
                                ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975))))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)

    # Diagnostics use a predesignated seed, not the luckiest seed.
    preferred_pooling = "attention" if "attention" in args.pooling else "mean"
    diagnostic_name = f"bigru_{preferred_pooling}_seed{args.seeds[0]}"
    probabilities = predictions[diagnostic_name].to_numpy()
    lengths_train = np.array([len(row) for row in token_lists["train"]])
    lengths_val = np.array([len(row) for row in token_lists["validation"]])
    boundaries = np.quantile(lengths_train, [1/3, 2/3])
    bins = np.digitize(lengths_val, boundaries, right=True)
    subgroup_records = []
    for group, label in enumerate(("short", "medium", "long")):
        mask = bins == group
        subgroup_records.append(dict(model=diagnostic_name, length_group=label, rows=int(mask.sum()),
                                     human=int((y_val[mask] == 0).sum()), AI=int((y_val[mask] == 1).sum()),
                                     min_tokens=int(lengths_val[mask].min()), max_tokens=int(lengths_val[mask].max()),
                                     auc=float(roc_auc_score(y_val[mask], probabilities[mask]))))
    pd.DataFrame(subgroup_records).to_csv(output / "length_subgroups.csv", index=False)
    mistakes = validation[["id", "label", "text"]].reset_index(drop=True).copy()
    mistakes["probability_AI"] = probabilities
    fp = mistakes.loc[(y_val == 0) & (probabilities >= .5)].nlargest(5, "probability_AI").assign(error="false_positive")
    fn = mistakes.loc[(y_val == 1) & (probabilities < .5)].nsmallest(5, "probability_AI").assign(error="false_negative")
    pd.concat([fp, fn]).to_csv(output / "error_examples.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"mean": "#0072B2", "attention": "#D55E00"}
    for pooling in args.pooling:
        for seed in args.seeds:
            history = pd.read_csv(output / f"bigru_{pooling}_seed{seed}" / "history.csv")
            label = f"{pooling.title()}, seed {seed}"
            axes[0].plot(history.epoch, history.validation_auc, label=label, alpha=.8, color=colors[pooling], linestyle={0:"-", 1:"--", 2:":"}[args.seeds.index(seed)%3])
            axes[1].plot(history.epoch, history.validation_loss, label=label, alpha=.8, color=colors[pooling], linestyle={0:"-", 1:"--", 2:":"}[args.seeds.index(seed)%3])
    axes[0].axhline(roc_auc_score(y_val, score_matrix[lookup["svm_punctuation"]]), color="#555555", linewidth=1, linestyle="-.", label="Best linear SVM")
    axes[0].set(ylabel="Validation AUROC", title="Ranking quality across epochs")
    axes[1].set(ylabel="Validation binary cross-entropy", title="Probability loss across epochs")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Bidirectional GRU: mean versus attention pooling")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    names = ["LR: words", "LR: punctuation", "Linear SVM"] + [f"GRU: {p} pooling" for p in args.pooling]
    values = [roc_auc_score(y_val, score_matrix[lookup[n]]) for n, *_ in references] + [family_auc[p] for p in args.pooling]
    axis.barh(names, values, color=["#aaaaaa", "#888888", "#555555"]+[colors[p] for p in args.pooling], alpha=.85)
    for i, value in enumerate(values):
        axis.text(value+.002, i, f"{value:.4f}", va="center")
    for i, pooling in enumerate(args.pooling, start=3):
        runs = metrics.loc[metrics.pooling.eq(pooling), "validation_auc"]
        axis.scatter(runs, np.full(len(runs), i), color="black", marker="|", s=130, zorder=4)
    axis.set(xlim=(min(values)-.04, 1), xlabel="Validation AUROC", title="Same 2,000 validation passages")
    axis.invert_yaxis()
    fig.text(.5, .01, "Neural bars: mean AUC across seeds; black marks: individual seeds. LR/SVM: best prior grid settings.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)

    if "attention" in args.pooling:
        attention_name = f"bigru_attention_seed{args.seeds[0]}"
        checkpoint = torch.load(output / attention_name / "best_model.pt", map_location="cpu", weights_only=True)
        model = BiGRUClassifier(len(vocabulary), "attention", config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        examples = []
        chosen_ids = validation.groupby("label")["id"].nsmallest(2).to_numpy()
        for passage_id in chosen_ids:
            index = int(np.flatnonzero(validation.id.to_numpy() == passage_id)[0])
            tokens, lengths, _ = batch_at(sequences["validation"], y_val, [index], device)
            with torch.no_grad():
                _, weights = model(tokens, lengths, True)
            for position, (token, weight) in enumerate(zip(token_lists["validation"][index][:config["max_tokens"]], weights[0].cpu().tolist())):
                examples.append(dict(id=int(passage_id), label=int(y_val[index]), position=position,
                                     token=token, is_unknown=token not in vocabulary, attention_weight=weight,
                                     probability_AI=float(predictions[attention_name].iloc[index])))
        pd.DataFrame(examples).to_csv(output / "attention_examples.csv", index=False)
    return metrics, summary, comparisons, subgroup_records, diagnostic_name


def run_workers(args):
    """Separate processes give each independent fit its own RNG and log file.

    The parent alone writes common artifacts. Workers only write their own run
    directories. This is parallel numerical training, not distributed training.
    """
    waiting = [(pooling, seed) for seed in args.seeds for pooling in args.pooling]
    running, records = [], []
    last_progress = time.perf_counter()
    try:
        while waiting or running:
            while waiting and len(running) < args.workers:
                pooling, seed = waiting.pop(0)
                name = f"bigru_{pooling}_seed{seed}"
                directory = args.output_dir / name
                directory.mkdir(exist_ok=True)
                log = (directory / "training.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--zip", str(args.zip.resolve()), "--output-dir", str(args.output_dir),
                           "--device", args.device, "--epochs", str(args.epochs),
                           "--threads", str(args.threads), "--worker-run", f"{pooling}:{seed}"]
                if args.resume:
                    command.append("--resume")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                running.append((process, log, directory))
                print(f"Started {name}; detailed log: {directory / 'training.log'}", flush=True)
            for process, log, directory in list(running):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                running.remove((process, log, directory))
                if return_code != 0:
                    tail = (directory / "training.log").read_text(encoding="utf-8", errors="replace")[-5000:]
                    raise RuntimeError(f"Training failed in {directory.name}:\n{tail}")
                result = json.loads((directory / "summary.json").read_text())
                records.append(result)
                pd.DataFrame(records).to_csv(args.output_dir / "metrics.csv", index=False)
                print(f"FINISHED {directory.name}: epoch {result['best_epoch']}, "
                      f"train AUC={result['train_auc']:.6f}, val AUC={result['validation_auc']:.6f}", flush=True)
            if time.perf_counter() - last_progress >= 40:
                for _, _, directory in running:
                    lines = (directory / "training.log").read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines:
                        print(lines[-1], flush=True)
                last_progress = time.perf_counter()
            if running:
                time.sleep(2)
    finally:
        for process, log, _ in running:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    return sorted(records, key=lambda record: (args.seeds.index(record["seed"]), args.pooling.index(record["pooling"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=baseline.PROJECT_DIR.parent / "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")
    parser.add_argument("--output-dir", type=Path, default=baseline.PROJECT_DIR / "results_neural")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--pooling", choices=["mean", "attention"], nargs="+", default=["mean", "attention"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU fits at once; keep 1 for GPU.")
    parser.add_argument("--worker-run", help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.pooling)) != len(args.pooling):
        parser.error("Epochs/threads must be positive; seeds and pooling choices must be unique.")
    args.output_dir = args.output_dir.resolve()
    for folder in ["results", "results_punctuation", "results_svm"]:
        if args.output_dir.is_relative_to((baseline.PROJECT_DIR / folder).resolve()):
            parser.error("Choose an output directory outside previous experiments.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU unavailable in this Python environment.")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.workers != 1:
        parser.error("Use --workers 1 on the GPU to avoid competing training jobs.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"PyTorch {torch.__version__}; device={device}; CPU threads={args.threads}", flush=True)
    if not args.worker_run:
        check_model(device)
    if args.check_only:
        return
    config = dict(SETTINGS, max_epochs=args.epochs)
    train, validation, sequences, vocabulary, audit, reference_hashes, token_lists = prepare_data(args, config, write_outputs=not args.worker_run)
    fingerprint_data = dict(config=config, zip_sha256=baseline.sha256(args.zip),
                            vocabulary_sha256=baseline.sha256(args.output_dir / "vocabulary.json"),
                            script_sha256=baseline.sha256(Path(__file__)), torch_version=str(torch.__version__),
                            device=str(device), cpu_threads=args.threads)
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    if args.worker_run:
        pooling, seed = args.worker_run.split(":")
        fit_one(pooling, int(seed), config, vocabulary, sequences,
                train.label.to_numpy(), validation.label.to_numpy(), device,
                args.output_dir, fingerprint, args.resume)
        return
    (args.output_dir / "run_configuration.json").write_text(json.dumps(dict(fingerprint_data, token_audit=audit,
        seeds=args.seeds, pooling=args.pooling, split_seed=42, fingerprint=fingerprint), indent=2), encoding="utf-8")
    records = []
    if args.workers > 1:
        records = run_workers(args)
    else:
        for seed in args.seeds:
            for pooling in args.pooling:
                records.append(fit_one(pooling, seed, config, vocabulary, sequences,
                                       train.label.to_numpy(), validation.label.to_numpy(), device,
                                       args.output_dir, fingerprint, args.resume))
                pd.DataFrame(records).to_csv(args.output_dir / "metrics.csv", index=False)
    metrics, summary, comparisons, subgroups, diagnostic_name = analysis_outputs(
        pd.DataFrame(records), train, validation, sequences, vocabulary, token_lists, device, args, config)
    for folder, hashes in reference_hashes.items():
        assert all(baseline.sha256(baseline.PROJECT_DIR / folder / name) == digest for name, digest in hashes.items())
    metadata = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), **fingerprint_data,
                    split_seed=42, training_seeds=args.seeds, pooling=args.pooling, token_pattern=TOKEN_PATTERN,
                    parallel_workers=args.workers,
                    vocabulary_size=len(vocabulary), token_audit=audit,
                    device_name=torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                    python_version=platform.python_version(), versions={name:version(name) for name in
                        ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "torch"]},
                    bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=43,
                    reference_output_hashes=reference_hashes, diagnostic_model=diagnostic_name,
                    checks=dict(padding_and_attention_invariants=True, original_split_reused=True,
                                vocabulary_training_only=True, saved_models_reproduce_predictions=True,
                                prior_results_unchanged=True, finite_losses_and_gradients=True))
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [__doc__, "\nACTUAL CONFIGURATION\n", json.dumps(config, indent=2),
              f"\nDevice: {device}. PyTorch {torch.__version__}. CPU threads: {args.threads}.",
              "\nTOKEN AUDIT\n", json.dumps(audit, indent=2), "\nALL RUNS\n", metrics.to_string(index=False),
              "\nARCHITECTURE SUMMARY (SD across training seeds, NOT a confidence interval)\n", summary.to_string(index=False),
              "\nPAIRED COMPARISONS (A minus B; conditional 95% bootstrap intervals)\n", comparisons.to_string(index=False),
              "\nLENGTH SUBGROUPS (training-set tertile boundaries; predesignated first seed)\n", pd.DataFrame(subgroups).to_string(index=False),
              "\nINTERPRETATION GUIDE\n",
              "Higher AUROC is better ranking; smaller loss/Brier is better probability quality. They can disagree.\n"
              "An improving training loss with worsening validation loss signals overfitting/overconfidence.\n"
              "A mean-vs-attention difference does not prove a universal advantage: only this size, tokenizer,\n"
              "optimizer, split and three initializations were tested. The SVM comparison changes both\n"
              "classifier and representation. No pretrained knowledge is available to these embeddings.\n"
              "Online training loss averages changing parameters with dropout ON; validation loss uses\n"
              "the epoch-end model with dropout OFF. Final train_loss evaluates the selected checkpoint\n"
              "with dropout OFF, and is directly comparable with final validation_loss.\n"
              "Attention examples use the two smallest validation IDs per class, not selected explanations.\n"
              "Large weights are mixture weights on contextual vectors, not causal evidence of AI writing.\n"
              "The probability cutoff .5 is predeclared for errors/accuracy; AUROC has no fixed cutoff.\n"
              "All six checkpoints remain trained on only the 8,000-row training partition.\n"]
    (args.output_dir / "experiment_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"All checks passed. Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
