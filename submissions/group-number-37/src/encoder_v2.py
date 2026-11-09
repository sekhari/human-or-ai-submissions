"""Full-precision DeBERTa fine-tuning with head-and-tail coverage and training-only validation."""
import gc
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.normalize import clean


class EncodedPassages(Dataset):
    def __init__(self, texts, tokenizer, labels=None, max_length=256):
        self.items = []
        budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        for index, text in enumerate(texts):
            tokens = tokenizer.encode(clean(text), add_special_tokens=False, verbose=False)
            if len(tokens) > budget:
                tail = budget // 3
                tokens = tokens[:budget-tail] + tokens[-tail:]
            item = tokenizer.prepare_for_model(
                tokens, add_special_tokens=True, return_attention_mask=True,
                return_token_type_ids=False, verbose=False)
            if labels is not None:
                item["labels"] = int(labels[index])
            self.items.append(item)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def device_name():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def collator(tokenizer):
    def collate(rows):
        return tokenizer.pad(rows, padding=True, return_tensors="pt")
    return collate


def fit_encoder(texts, labels, directory, seed=3737, epochs=2, model_name="microsoft/deberta-v3-base",
                max_length=256, batch_size=8, learning_rate=1e-5, max_steps=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=False)
    config = AutoConfig.from_pretrained(model_name, token=False)
    config.num_labels = 2
    for attribute in ("hidden_dropout_prob", "attention_probs_dropout_prob", "pooler_dropout", "cls_dropout"):
        setattr(config, attribute, 0.)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, token=False, use_safetensors=False).to(device)
    # Keep the pretrained vocabulary geometry stable for words from unseen domains.
    for parameter in model.base_model.embeddings.parameters():
        parameter.requires_grad = False
    dataset = EncodedPassages(texts, tokenizer, labels, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator(tokenizer),
                        generator=torch.Generator().manual_seed(seed))
    accumulation = 4
    updates = epochs * math.ceil(len(loader) / accumulation)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(updates * .08), updates)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    import time
    started = time.perf_counter()
    seen_steps = 0
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {name:value.to(device) for name,value in batch.items()}
            # Correctly normalize the final partial accumulation group.
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"], label_smoothing=.02)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite encoder loss.")
            (loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
            seen_steps += 1
            if step % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f"encoder epoch {epoch+1}/{epochs} step {step+1}/{len(loader)} "
                      f"loss {loss.item():.4f} elapsed {elapsed/60:.1f} min", flush=True)
            if max_steps is not None and seen_steps >= max_steps:
                break
        if max_steps is not None and seen_steps >= max_steps:
            break
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    if max_steps is None:
        (directory/"training_complete.json").write_text(json.dumps(
            {"epochs":epochs,"rows":len(texts),"model":model_name,"seed":seed}
        )+"\n")
    del model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def predict_encoder(texts, directory, max_length=256, batch_size=16):
    device = device_name()
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False, token=False)
    model = AutoModelForSequenceClassification.from_pretrained(directory).to(device).eval()
    dataset = EncodedPassages(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator(tokenizer))
    result = []
    for batch in loader:
        batch = {name:value.to(device) for name,value in batch.items()}
        logits = model(**batch).logits.float().cpu().numpy()
        result.extend(logits[:, 1] - logits[:, 0])
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(result)
