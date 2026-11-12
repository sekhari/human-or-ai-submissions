"""Local transformer fine-tuning with fixed epochs and dynamic padding."""

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tokenizer, texts, max_length, batch_size, labels=None, seed=None):
    encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
    rows = [dict(zip(encoded.keys(), values)) for values in zip(*encoded.values())]
    if labels is not None:
        for row, label in zip(rows, labels):
            row["labels"] = int(label)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        rows, batch_size=batch_size, shuffle=seed is not None, generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def fit(texts, labels, args, folder):
    set_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, revision=args.revision, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, revision=args.revision, num_labels=2,
        id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1},
    ).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    loader = batches(tokenizer, texts, args.max_length, args.batch_size, labels, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates = math.ceil(len(loader) / args.accumulate) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * 0.1), updates)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        group_rows = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            n_rows = len(batch["labels"])
            context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
            with context:
                loss = model(**batch).loss
            (loss * n_rows).backward()
            loss_sum += loss.detach().item() * n_rows
            group_rows += n_rows
            if step % args.accumulate == 0 or step == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_rows)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_rows = 0
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {loss_sum / len(texts):.5f}", flush=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return model, tokenizer, device, revision


@torch.inference_mode()
def scores(model, tokenizer, texts, max_length, batch_size, device):
    model.eval()
    predictions = []
    for batch in batches(tokenizer, texts, max_length, batch_size):
        logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
        predictions.append(logits.float().softmax(dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(predictions)


def predict(folder, texts, max_length, batch_size, device_name):
    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).to(device)
    return scores(model, tokenizer, texts, max_length, batch_size, device)
