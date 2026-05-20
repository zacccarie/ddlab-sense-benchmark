"""Modèle A — modèle séquentiel (analogue LLM).

GRU qui prédit l'état suivant d'une trajectoire. L'état de n cellules est
représenté en vecteur binaire: le modèle peut donc apprendre la règle locale
de l'automate et, potentiellement, généraliser à des états jamais vus."""

import numpy as np
import torch
import torch.nn as nn


def stateToBits(states, n):
    """Convertit des entiers-états en vecteurs binaires de forme (..., n)."""
    states = np.asarray(states, dtype=np.int64)
    bitPositions = np.arange(n, dtype=np.int64)
    return ((states[..., None] >> bitPositions) & 1).astype(np.float32)


def bitsToState(bits):
    """Convertit des vecteurs binaires (..., n) en entiers-états."""
    n = bits.shape[-1]
    weights = 1 << np.arange(n, dtype=np.int64)
    return (np.asarray(bits, dtype=np.int64) * weights).sum(axis=-1)


class SequenceModel(nn.Module):
    """GRU prédisant l'état suivant à partir de l'historique de trajectoire."""

    def __init__(self, n, hidden=128, layers=2):
        super().__init__()
        self.n = n
        self.gru = nn.GRU(n, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n)

    def forward(self, x):
        # x: (batch, time, n) -> logits par cellule (batch, time, n)
        out, _ = self.gru(x)
        return self.head(out)

    def predictNext(self, states, device):
        """Prédit l'état suivant (entiers) pour un batch d'états, en un pas."""
        self.eval()
        with torch.no_grad():
            bits = torch.tensor(stateToBits(states, self.n), device=device).unsqueeze(1)
            logits = self.forward(bits)[:, -1, :]
            predicted = (logits > 0).cpu().numpy()
        return bitsToState(predicted)

    def representation(self, states, device):
        """Représentation cachée du GRU (sortie avant la tête), un pas, pour un batch d'états."""
        self.eval()
        with torch.no_grad():
            bits = torch.tensor(stateToBits(states, self.n), device=device).unsqueeze(1)
            out, _ = self.gru(bits)
            return out[:, -1, :].cpu().numpy()
