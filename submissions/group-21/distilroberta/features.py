"""Feature extraction shared by training and prediction."""
from collections import Counter
import re

import numpy as np

PUNCT = [",", ".", ";", ":", "!", "?", "(", '"', "'", "-", "—", "\n"]
STYLO_NAMES = (["chars", "words", "avg_word_len", "n_sentences", "sent_len_mean", "sent_len_std",
                "sent_len_burstiness", "type_token_ratio", "hapax_ratio", "upper_ratio", "digit_ratio"]
               + [f"punct_{i}" for i in range(len(PUNCT))])

def stylo(t):
    words = re.findall(r"[A-Za-z']+", t)
    sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    sl = np.array([len(s.split()) for s in sents]) if sents else np.array([0.0])
    nw, nc = max(len(words), 1), max(len(t), 1)
    counts = Counter(w.lower() for w in words)
    row = [nc, nw,
           np.mean([len(w) for w in words]) if words else 0.0,
           len(sents), sl.mean(), sl.std(), sl.std() / (sl.mean() + 1e-9),
           len(counts) / nw,
           sum(1 for c in counts.values() if c == 1) / nw,
           sum(ch.isupper() for ch in t) / nc,
           sum(ch.isdigit() for ch in t) / nc]
    row += [t.count(p) / nc for p in PUNCT]
    return row

LM_NAMES = ["mean_logprob", "std_logprob", "min_logprob", "mean_entropy",
            "mean_logprob_plus_entropy", "mean_log_rank", "top1_frac", "top10_frac"]

def lm_features(texts_, model_name="distilgpt2", max_len=256, bs=16, revision=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision); tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    if device == "cuda":
        model.half()
    order = np.argsort([len(t) for t in texts_])            # similar lengths together, less padding
    out = np.zeros((len(texts_), len(LM_NAMES)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts_), bs):
            ids = order[s:s + bs]
            enc = tok([texts_[i] or "." for i in ids], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 2:
                # A single-token passage contributes no next-token statistics.
                continue
            logp = torch.log_softmax(model(**enc).logits[:, :-1].float(), dim=-1)
            tgt = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].bool(); mf = mask.float(); n = mf.sum(1).clamp(min=1)
            tlp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)       # log-prob of the real next token
            ent = -(logp.exp() * logp).sum(-1)                          # model uncertainty at each position
            rank = (logp > tlp.unsqueeze(-1)).sum(-1) + 1               # rank of the real token
            mean_lp = (tlp * mf).sum(1) / n
            std_lp = ((((tlp - mean_lp[:, None]) ** 2) * mf).sum(1) / n).sqrt()
            min_lp = tlp.masked_fill(~mask, 0).min(1).values
            feats = torch.stack([
                mean_lp, std_lp, min_lp,
                (ent * mf).sum(1) / n,
                ((tlp + ent) * mf).sum(1) / n,
                (rank.float().log() * mf).sum(1) / n,
                ((rank == 1).float() * mf).sum(1) / n,
                ((rank <= 10).float() * mf).sum(1) / n], dim=1)
            out[ids] = feats.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return np.nan_to_num(out)

def clean(text):
    return ' '.join(text.split())

class Passages:
    def __init__(self, tokenizer, texts, labels=None):
        self.tokens = tokenizer([clean(t) for t in texts], truncation=True, max_length=320)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, i):
        row = {key: values[i] for key, values in self.tokens.items()}
        if self.labels is not None:
            row['labels'] = int(self.labels[i])
        return row

def ranking_margins(trainer, tokenizer, texts):
    logits = trainer.predict(Passages(tokenizer, texts)).predictions.astype(float)
    result = logits[:, 1] - logits[:, 0]
    assert np.isfinite(result).all()
    return result
