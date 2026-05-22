"""Transformer de trajectoire — l'objectif récursif du §11 sur l'architecture
des world models.

Lit une fenêtre d'orbite (windowLength états successifs, n cellules chacun) et
produit un latent pour l'état initial. Petit : quelques têtes, quelques couches.
Entraîné par l'objectif de bisimulation récursif (cf.
`bisimulation.trainTrajectoryTransformer`) et audité, sur un environnement
énumérable, contre le quotient exact."""

import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    """phi : fenêtre d'orbite (windowLength, n) -> latent de l'état initial."""

    def __init__(self, n, windowLength, latentDim=16, dModel=32, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(n, dModel)
        self.posEmbed = nn.Parameter(torch.randn(windowLength, dModel) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dModel, heads, dim_feedforward=2 * dModel,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(dModel, latentDim)

    def forward(self, orbit):
        """orbit : (batch, windowLength, n) flottant -> (batch, latentDim)."""
        h = self.embed(orbit) + self.posEmbed
        h = self.encoder(h)
        return self.head(h[:, 0])  # token de position 0 = l'état initial
