"""Domain-adaptive pretraining (DAPT): continue DistilBERT's masked-LM training on competition text.

No labels are used. Corpus:
    --corpus split   8,000 training-split passages + unlabeled.csv  (for validation experiments;
                     validation passages and test.csv are never included)
    --corpus full    all 10,000 train.csv passages + unlabeled.csv   (for the final full-data model)
The same whitespace markers as fine-tuning are applied, so the marker embeddings are learned here too.

    python -m experiments.distilbert_mlm --corpus split --newline-marker "[NL]"

Output: checkpoints/mlm/<name>/ (git-ignored). Fine-tune from it with
    python -m experiments.distilbert_train --model-name checkpoints/mlm/<name> --newline-marker "[NL]" ...
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from experiments.distilbert_train import CHECKPOINT_ROOT
from src.classifiers.distilbert import (
    DistilBertConfig,
    _EncodedDataset,
    add_marker_tokens,
    mark_whitespace,
    whitespace_markers,
)
from src.data import DATA_DIR, load_labeled, load_split, load_unlabeled
from src.utils import get_device, set_seed


def corpus_texts(corpus: str) -> list[str]:
    train = load_split().train_texts if corpus == "split" else load_labeled(DATA_DIR / "train.csv").texts
    return list(train) + load_unlabeled(DATA_DIR / "unlabeled.csv")[1]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=("split", "full"), required=True)
    p.add_argument("--model-name", default=DistilBertConfig.model_name)
    p.add_argument("--newline-marker", default=None)
    p.add_argument("--space-marker", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N corpus texts")
    a = p.parse_args(argv)

    cfg = DistilBertConfig(model_name=a.model_name, newline_marker=a.newline_marker, space_marker=a.space_marker)
    name = f"dapt-{a.corpus}" + ("-nl" if a.newline_marker else "") + ("-ds" if a.space_marker else "")
    out_dir = CHECKPOINT_ROOT / "mlm" / name
    set_seed(a.seed)
    device = get_device(a.device)

    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForMaskedLM.from_pretrained(a.model_name)
    add_marker_tokens(tokenizer, model, cfg)
    model.to(device)

    texts = corpus_texts(a.corpus)[: a.limit]
    if whitespace_markers(cfg):
        texts = [mark_whitespace(t, cfg) for t in texts]
    enc = tokenizer(texts, truncation=True, max_length=a.max_length, return_special_tokens_mask=True)
    keys = ("input_ids", "attention_mask", "special_tokens_mask")
    loader = DataLoader(
        _EncodedDataset({k: enc[k] for k in keys}),
        batch_size=a.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(a.seed),
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm_probability=a.mlm_probability),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total), total)
    print(f"[{name}] device={device} texts={len(texts)} steps={total}", flush=True)

    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, loss_sum, n = time.perf_counter(), 0.0, 0
        for step, batch in enumerate(loader, 1):
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum, n = loss_sum + loss.item(), n + 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} mlm_loss {loss_sum / n:.4f}", flush=True)
        history.append({"epoch": epoch, "mlm_loss": loss_sum / n, "seconds": round(time.perf_counter() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {**vars(a), "n_texts": len(texts), "history": history}
    (out_dir / "dapt_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{name}] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
