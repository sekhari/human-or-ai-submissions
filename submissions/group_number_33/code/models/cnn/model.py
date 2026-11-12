"""CNN architecture and epoch/prediction functions used by the current workflow."""

# Submitted architecture (B = batch size, usually 64):
#
#                       Token IDs [B, 768]
#                               |
#                  Trainable embedding lookup
#                       [B, 768, 100]
#                               |
#                  Transpose -> [B, 100, 768]
#                               |
#              +----------------+----------------+
#              |                |                |
#          Conv1d k=3       Conv1d k=4       Conv1d k=5
#          64 filters      64 filters      64 filters
#              |                |                |
#             ReLU             ReLU             ReLU
#              |                |                |
#          Max pooling     Max pooling     Max pooling
#           [B, 64]         [B, 64]         [B, 64]
#              |                |                |
#              +----------------+----------------+
#                               |
#                      Concatenate [B, 192]
#                               |
#                       Dropout (p=0.5)
#                               |
#                       Linear (192 -> 1)
#                               |
#                         Logits [B]
#                               |
#                    Sigmoid during prediction
#                               |
#                            P(AI)
#
# Pooling excludes padding-only windows. Embeddings start randomly.
# Training uses logits directly with BCEWithLogitsLoss and AdamW.

import numpy as np
import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        filters=64,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, kernel_size=width)
            for width in (3, 4, 5)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters * 3, 1)

    def forward(self, tokens):
        # (batch, tokens) -> (batch, embedding_dim, tokens)
        embedded = self.embedding(tokens).transpose(1, 2)
        lengths = (tokens != 0).sum(dim=1)

        pooled = []

        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))

            # Exclude windows containing only padding.
            # Keep one window available for an empty passage.
            positions = torch.arange(
                features.size(2),
                device=tokens.device,
            )
            valid = positions[None, :] < lengths.clamp(min=1)[:, None]
            features = features.masked_fill(
                ~valid[:, None, :],
                float("-inf"),
            )

            pooled.append(features.amax(dim=2))

        combined = torch.cat(pooled, dim=1)

        # Return logits; apply sigmoid only when predicting probabilities.
        return self.output(self.dropout(combined)).squeeze(-1)


def run_epoch(
    model, loader, criterion, device, optimizer=None, return_probabilities=False
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    probabilities = []

    with torch.set_grad_enabled(training):
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(tokens)
            loss = criterion(logits, labels)

            if return_probabilities:
                probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(tokens)

    average_loss = total_loss / len(loader.dataset)
    if return_probabilities:
        return average_loss, np.concatenate(probabilities)
    return average_loss


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()

    probabilities = [
        torch.sigmoid(model(batch[0].to(device))).cpu().numpy()
        for batch in loader
    ]

    return np.concatenate(probabilities)
