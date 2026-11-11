"""C3: a small Transformer encoder trained from scratch (random init, no pretrained weights).

Tokenizer: byte-level BPE trained here with the `tokenizers` library on the training rows plus
unlabeled.csv only (never validation or test text). Byte-level pre-tokenisation keeps case, spaces
and line breaks, which carry signal in this data. Model: learned token + position embeddings,
`n_layers` pre-norm encoder layers, mean pooling over tokens, linear head. Trained with AdamW and
a warmup + linear-decay schedule; early stopping on a stratified inner split of the training rows
(`inner_val_frac`), restoring the best epoch. The validation set of train.py is never seen.

Two options make it rely less on what does not carry over to test (unseen topics, far more verse):
`distort_k` > 0 feeds text-distorted input (features/distortion.py: words outside the k most frequent
training words masked), and `crop_len` > 0 trains on a random window of `crop_len` tokens per passage each
epoch and predicts the mean logit over consecutive windows, so no single position or length pattern decides.
`join_lines` turns single line breaks into spaces (blank lines stay): in train, human passages have long lines
and AI passages short ones, and without this the model scores almost any human verse as AI.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features.distortion import Distorter

from .base import BaseModel

PAD, UNK = "<pad>", "<unk>"
SINGLE_NL = re.compile(r"[ \t]*(?<!\n)\n(?!\n)[ \t]*")


def train_bpe(texts, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[PAD, UNK],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


class TransformerScratch(BaseModel):
    name = "transformer_scratch"

    def __init__(self, vocab_size: int = 12000, max_len: int = 512, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1, lr: float = 3e-4,
                 weight_decay: float = 0.01, batch_size: int = 32, max_epochs: int = 20, patience: int = 3,
                 warmup_frac: float = 0.06, inner_val_frac: float = 0.1, distort_k: int = 0, crop_len: int = 0,
                 join_lines: bool = False, device: str = "auto", seed: int = 0):
        self.hp = dict(vocab_size=vocab_size, max_len=max_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       d_ff=d_ff, dropout=dropout, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                       max_epochs=max_epochs, patience=patience, warmup_frac=warmup_frac,
                       inner_val_frac=inner_val_frac, distort_k=distort_k, crop_len=crop_len,
                       join_lines=join_lines, seed=seed)
        self.device_pref = device
        self.history: list[dict] = []
        self.distorter = None

    # ------------------------------------------------------------ torch pieces
    def _device(self):
        import torch
        if self.device_pref not in (None, "auto"):
            return torch.device(self.device_pref)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build(self):
        import torch
        from torch import nn
        hp = self.hp

        class Encoder(nn.Module):
            def __init__(self, vocab):
                super().__init__()
                self.tok = nn.Embedding(vocab, hp["d_model"], padding_idx=0)
                self.pos = nn.Embedding(hp["max_len"], hp["d_model"])
                layer = nn.TransformerEncoderLayer(hp["d_model"], hp["n_heads"], hp["d_ff"], hp["dropout"],
                                                   batch_first=True, norm_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(layer, hp["n_layers"], enable_nested_tensor=False)
                self.norm = nn.LayerNorm(hp["d_model"])
                self.head = nn.Linear(hp["d_model"], 1)

            def forward(self, ids):
                mask = ids == 0
                pos = torch.arange(ids.shape[1], device=ids.device)
                h = self.enc(self.tok(ids) + self.pos(pos)[None], src_key_padding_mask=mask)
                h = self.norm(h)
                keep = (~mask).unsqueeze(-1).float()
                return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(-1)

        return Encoder(self.tok.get_vocab_size())

    def _prep(self, texts):
        texts = [SINGLE_NL.sub(" ", t) for t in texts] if self.hp["join_lines"] else list(texts)
        return self.distorter.transform(texts) if self.distorter else texts

    def _encode(self, texts):
        texts = self._prep(texts)
        return [e.ids[: self.hp["max_len"]] or [1] for e in self.tok.encode_batch(texts)]

    def _windows(self, ids):
        """(text index, window) pairs: consecutive crop_len windows, the last one aligned to the end."""
        L = self.hp["crop_len"]
        out = []
        for i, x in enumerate(ids):
            starts = range(0, max(1, len(x) - L + 1), L) if L and len(x) > L else [0]
            starts = list(starts) + ([len(x) - L] if L and len(x) > L and (len(x) - L) % L else [])
            out += [(i, x[s:s + L] if L else x) for s in starts]
        return out

    def _batches(self, ids, order, bs):
        import torch
        for s in range(0, len(order), bs):
            b = [ids[i] for i in order[s:s + bs]]
            L = max(len(x) for x in b)
            yield order[s:s + bs], torch.tensor([x + [0] * (L - len(x)) for x in b], dtype=torch.long)

    def _scores(self, net, ids, dev):
        import torch
        net.eval()
        owner, wins = map(list, zip(*self._windows(ids)))
        order = np.argsort([len(x) for x in wins])  # length-sorted batches: less padding
        out = np.zeros(len(wins))
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for idx, x in self._batches(wins, order, 128):
                out[idx] = net(x.to(dev)).float().cpu().numpy()
        return np.bincount(owner, out, len(ids)) / np.bincount(owner, minlength=len(ids))  # mean logit per text

    # ------------------------------------------------------------ API
    def fit(self, texts, y, unlabeled_texts=None):
        import torch
        hp = self.hp
        torch.manual_seed(hp["seed"])
        rng = np.random.default_rng(hp["seed"])
        texts, y = list(texts), np.asarray(y)
        t0 = time.time()
        if hp["distort_k"]:
            self.distorter = Distorter(hp["distort_k"]).fit(texts)
        self.tok = train_bpe(self._prep(texts) + self._prep(unlabeled_texts or []), hp["vocab_size"])
        ids = self._encode(texts)
        tr, iv = train_test_split(np.arange(len(y)), test_size=hp["inner_val_frac"], stratify=y,
                                  random_state=hp["seed"])
        dev = self._device()
        self.device_used = str(dev)
        net = self._build().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        steps = math.ceil(len(tr) / hp["batch_size"]) * hp["max_epochs"]
        warm = max(1, int(hp["warmup_frac"] * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        yt = torch.tensor(y, dtype=torch.float32)
        best, best_state, bad = -1.0, None, 0
        self.history = []
        print(f"  BPE vocab {self.tok.get_vocab_size()} ({time.time() - t0:.0f}s), "
              f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params on {dev}", flush=True)
        for epoch in range(1, hp["max_epochs"] + 1):
            net.train()
            perm = rng.permutation(tr)
            L = hp["crop_len"]
            ep_ids = ids if not L else [x if len(x) <= L else x[(s := rng.integers(len(x) - L + 1)):s + L] for x in ids]
            # bucket by length within shuffled chunks: random batches, little padding
            chunks = [sorted(perm[i:i + hp["batch_size"] * 20], key=lambda j: len(ep_ids[j]))
                      for i in range(0, len(perm), hp["batch_size"] * 20)]
            order = np.concatenate(chunks)
            batches = list(self._batches(ep_ids, order, hp["batch_size"]))
            rng.shuffle(batches)
            total = 0.0
            for idx, x in batches:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    loss = loss_fn(net(x.to(dev)).float(), yt[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            auc = roc_auc_score(y[iv], self._scores(net, [ids[i] for i in iv], dev))
            self.history.append({"epoch": epoch, "train_loss": total / len(tr), "inner_val_auroc": auc,
                                 "elapsed_s": round(time.time() - t0, 1)})
            print(f"  epoch {epoch}: train loss {total / len(tr):.4f}, inner-val AUROC {auc:.4f}", flush=True)
            if auc > best:
                best, bad = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        net.load_state_dict(best_state)
        self.best_epoch = max(self.history, key=lambda h: h["inner_val_auroc"])["epoch"]
        self.net, self.dev = net, dev
        return self

    def predict_proba(self, texts):
        return 1 / (1 + np.exp(-self._scores(self.net, self._encode(texts), self.dev)))

    def save(self, path):
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path / "tokenizer.json"))
        torch.save(self.net.state_dict(), path / "model.pt")
        (path / "hp.json").write_text(json.dumps({"hp": self.hp, "best_epoch": self.best_epoch,
                                                  "distort_keep": sorted(self.distorter.keep) if self.distorter else None}),
                                      encoding="utf-8")

    @classmethod
    def load(cls, path, device: str = "auto"):
        import torch
        from tokenizers import Tokenizer
        path = Path(path)
        meta = json.loads((path / "hp.json").read_text(encoding="utf-8"))
        m = cls(**meta["hp"], device=device)
        m.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        if meta.get("distort_keep") is not None:
            m.distorter = Distorter(m.hp["distort_k"])
            m.distorter.keep = set(meta["distort_keep"])
        m.dev = m._device()
        m.net = m._build().to(m.dev)
        m.net.load_state_dict(torch.load(path / "model.pt", map_location=m.dev))
        m.best_epoch = meta["best_epoch"]
        return m
