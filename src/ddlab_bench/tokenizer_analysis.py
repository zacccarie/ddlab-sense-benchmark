"""Phase 7-8 — machinerie d'analyse du tokeniseur VQ-VAE sur automate rendu.

`trainVqvae` accepte des images d'entrée et des images cible séparées :
  - cible == entrée            -> objectif de reconstruction (aveugle dynamique)
  - cible == entrée[succ]      -> objectif prédictif (le code doit retenir le
                                  rôle dynamique, pas l'apparence)."""

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.models.world_model import VQVAE


def reviveDeadCodes(model, data, device, rng):
    """Réinitialise les codes inutilisés vers des latents observés (+ bruit) —
    contre l'effondrement du codebook, pathologie classique du VQ-VAE."""
    model.eval()
    with torch.no_grad():
        latent = model.encodeLatent(data)
        codes, _ = model.quantize(latent)
        used = {int(c) for c in codes.cpu().numpy()}
        dead = [c for c in range(model.codebookSize) if c not in used]
        if dead:
            picks = rng.integers(0, len(latent), size=len(dead))
            noise = torch.randn(len(dead), model.latentDim, device=device) * 0.05
            model.codebook.weight.data[dead] = latent[picks] + noise
    model.train()


def trainVqvae(inputImages, targetImages, codebookSize, epochs, device):
    """Entraîne un VQ-VAE : encode inputImages, décode vers targetImages.

    targetImages == inputImages -> reconstruction ; targetImages == successeurs
    -> objectif prédictif."""
    torch.manual_seed(0)
    model = VQVAE(inputImages.shape[1], codebookSize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.BCEWithLogitsLoss()
    inp = torch.tensor(inputImages, dtype=torch.float32, device=device)
    tgt = torch.tensor(targetImages, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(inp))
        for start in range(0, len(order), 64):
            idx = torch.from_numpy(order[start:start + 64])
            decoded, vqLoss, _ = model(inp[idx])
            loss = lossFn(decoded, tgt[idx]) + vqLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch < int(epochs * 0.85):
            reviveDeadCodes(model, inp, device, rng)
    return model


def latentBasinField(code, succCode):
    """Champ de bassins du graphe de transition latent (code -> code-successeur
    majoritaire). Retourne (champ latent, remap code -> index contigu)."""
    usedCodes = sorted({int(c) for c in code})
    remap = {c: i for i, c in enumerate(usedCodes)}
    votes = defaultdict(Counter)
    for state in range(len(code)):
        votes[int(code[state])][int(succCode[state])] += 1
    latentSucc = np.empty(len(usedCodes), dtype=np.int64)
    for c in usedCodes:
        latentSucc[remap[c]] = remap[votes[c].most_common(1)[0][0]]
    return basinFieldFromSuccessor(latentSucc, 0, "latent"), remap


def metricsForCode(code, field):
    """Métriques de préservation de structure pour une tokenisation donnée."""
    size = len(code)
    succCode = code[field.succ]
    distinctCodes = len(np.unique(code))

    successorSets = defaultdict(set)
    for state in range(size):
        successorSets[int(code[state])].add(int(succCode[state]))
    consistent = sum(
        1 for state in range(size) if len(successorSets[int(code[state])]) == 1
    ) / size

    latentField, remap = latentBasinField(code, succCode)
    latentBasinOfState = np.array(
        [latentField.attractorId[remap[int(code[state])]] for state in range(size)]
    )
    ami = float(adjusted_mutual_info_score(field.attractorId, latentBasinOfState))
    return {
        "distinctCodes": distinctCodes,
        "collisionRate": 1.0 - distinctCodes / size,
        "consistent": consistent,
        "latentBasins": len(latentField.attractors),
        "ami": ami,
    }
