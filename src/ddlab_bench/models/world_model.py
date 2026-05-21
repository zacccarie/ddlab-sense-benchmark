"""Phase 7 — world model tokenisé (Pont 1).

VQVAE : encodeur convolutif → quantification vectorielle (codebook discret) →
décodeur. Le tokeniseur est appris sur les seules images, sans connaître la
dynamique. LatentDynamics : prédit le code latent suivant depuis le code courant."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAE(nn.Module):
    """Auto-encodeur à quantification vectorielle : tokenise une image en un code
    discret parmi `codebookSize`."""

    def __init__(self, imageSide, codebookSize, latentDim=16):
        super().__init__()
        self.codebookSize = codebookSize
        self.latentDim = latentDim
        self.reduced = imageSide // 4  # deux convolutions stride 2
        self.encoderConv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.encoderHead = nn.Linear(32 * self.reduced * self.reduced, latentDim)
        self.codebook = nn.Embedding(codebookSize, latentDim)
        self.codebook.weight.data.uniform_(-1.0 / codebookSize, 1.0 / codebookSize)
        self.decoderHead = nn.Linear(latentDim, 32 * self.reduced * self.reduced)
        self.decoderConv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def encodeLatent(self, images):
        """Images (B, side, side) → vecteur latent continu (B, latentDim)."""
        features = self.encoderConv(images.unsqueeze(1))
        return self.encoderHead(features.flatten(1))

    def quantize(self, latent):
        """Latent continu → (indices de code, vecteurs quantifiés)."""
        distances = torch.cdist(latent, self.codebook.weight)
        codes = distances.argmin(dim=1)
        return codes, self.codebook(codes)

    def decode(self, quantized):
        features = self.decoderHead(quantized).view(-1, 32, self.reduced, self.reduced)
        return self.decoderConv(features).squeeze(1)

    def forward(self, images):
        latent = self.encodeLatent(images)
        codes, quantized = self.quantize(latent)
        # estimateur straight-through : le gradient de reconstruction traverse la
        # quantification non différentiable
        quantizedSt = latent + (quantized - latent).detach()
        decoded = self.decode(quantizedSt)
        codebookLoss = F.mse_loss(quantized, latent.detach())
        commitmentLoss = F.mse_loss(latent, quantized.detach())
        return decoded, codebookLoss + 0.25 * commitmentLoss, codes

    def encodeCodes(self, images, device, chunk=512):
        """Code discret de chaque image — le résultat de la tokenisation apprise."""
        self.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(images), chunk):
                block = torch.tensor(
                    images[start:start + chunk], dtype=torch.float32, device=device
                )
                codes, _ = self.quantize(self.encodeLatent(block))
                out.append(codes.cpu().numpy())
        return np.concatenate(out)


class LatentDynamics(nn.Module):
    """Prédit le code latent suivant depuis le code courant."""

    def __init__(self, codebookSize, embedDim=32, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(codebookSize, embedDim)
        self.net = nn.Sequential(
            nn.Linear(embedDim, hidden), nn.ReLU(),
            nn.Linear(hidden, codebookSize),
        )

    def forward(self, codes):
        return self.net(self.embed(codes))


class ContrastiveEncoder(nn.Module):
    """Encodeur convolutif pour l'apprentissage contrastif temporel.

    Image -> embedding L2-normalisé (la similarité cosinus est alors un simple
    produit scalaire)."""

    def __init__(self, imageSide, embedDim=32):
        super().__init__()
        reduced = imageSide // 4
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(32 * reduced * reduced, embedDim)

    def forward(self, images):
        features = self.conv(images.unsqueeze(1))
        return F.normalize(self.head(features.flatten(1)), dim=1)
