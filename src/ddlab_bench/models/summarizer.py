"""Modèle de résumé de trajectoire (Information Bottleneck).

Encode une trajectoire entière en un code compressé de dimension `bottleneck`,
puis prédit le bassin — le « sens » de la trajectoire, vers quel attracteur elle
converge — depuis ce seul code. Le balayage de `bottleneck` trace la courbe IB:
combien de dimensions faut-il pour préserver le sens d'une trajectoire ?"""

import numpy as np
import torch
import torch.nn as nn


class TrajectorySummarizer(nn.Module):
    """GRU encodeur -> goulot de dimension `bottleneck` -> prédiction du bassin."""

    def __init__(self, n, numBasins, bottleneck, hidden=128):
        super().__init__()
        self.encoder = nn.GRU(n, hidden, batch_first=True)
        self.toCode = nn.Linear(hidden, bottleneck)
        self.basinHead = nn.Linear(bottleneck, numBasins)

    def encode(self, trajectoryBits):
        """Résume une trajectoire (batch, time, n) en un code (batch, bottleneck)."""
        out, _ = self.encoder(trajectoryBits)
        return torch.tanh(self.toCode(out[:, -1, :]))

    def forward(self, trajectoryBits):
        return self.basinHead(self.encode(trajectoryBits))


def trajectoryBasinLabels(field, trajectories):
    """Étiquette de bassin de chaque trajectoire.

    Le bassin est constant le long d'une trajectoire (un bassin est fermé sous la
    dynamique). Les bassins observés sont remappés sur 0..K-1 contigus.
    Retourne (labels, numBasins)."""
    rawLabels = field.attractorId[trajectories[:, 0]]
    uniqueBasins = np.unique(rawLabels)
    remap = {int(basin): i for i, basin in enumerate(uniqueBasins)}
    labels = np.array([remap[int(basin)] for basin in rawLabels], dtype=np.int64)
    return labels, len(uniqueBasins)
