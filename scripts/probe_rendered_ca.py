"""Phase 7 — Pont 1 : balayage de la taille du codebook.

On rend les états d'un ECA en images, on entraîne un VQ-VAE dessus (tokeniseur
aveugle à la dynamique), puis on projette la vraie dynamique à travers la
tokenisation : graphe latent code[s] -> code[succ s].

Ici on balaye K (taille du codebook). À K plein chaque état peut garder son
code ; en réduisant K on FORCE la compression — des états distincts doivent
partager un code. Question : à quel K la structure causale (le champ de bassins)
casse ? C'est l'Information Bottleneck appliqué au tokeniseur lui-même."""

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import computeBasinField
from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.models.world_model import VQVAE
from ddlab_bench.rendering import renderAll

RULE = 110
N = 9
VQVAE_EPOCHS = 100
K_VALUES = [512, 256, 128, 64, 32, 16]


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


def trainVqvae(images, codebookSize, epochs, device):
    torch.manual_seed(0)
    model = VQVAE(images.shape[1], codebookSize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    reconLoss = nn.BCEWithLogitsLoss()
    data = torch.tensor(images, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(data))
        for start in range(0, len(order), 64):
            batch = data[torch.from_numpy(order[start:start + 64])]
            decoded, vqLoss, _ = model(batch)
            loss = reconLoss(decoded, batch) + vqLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch < int(epochs * 0.85):
            reviveDeadCodes(model, data, device, rng)
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
    """Métriques de préservation pour une tokenisation donnée."""
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


def main():
    device = "cpu"
    field = computeBasinField(RULE, N)
    images = renderAll(N)
    trueBasins = len(field.attractors)
    print(f"règle {RULE}, n={N} | {1 << N} états | balayage du codebook K")
    print(f"vrais bassins : {trueBasins}\n")

    header = (f"{'K':>5} {'codes':>7} {'collision':>10} {'cohérence':>10} "
              f"{'bassins lat.':>13} {'AMI':>8}")
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        vqvae = trainVqvae(images, k, VQVAE_EPOCHS, device)
        code = vqvae.encodeCodes(images, device)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['distinctCodes']:>7} {m['collisionRate']:>9.3f} "
              f"{m['consistent']:>10.3f} {m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — AMI vs K : où la compression du tokeniseur casse la structure"
        "\ncausale. AMI haut = champ de bassins latent fidèle ; chute = la"
        "\ntokenisation fusionne des états dynamiquement distincts."
    )


if __name__ == "__main__":
    main()
