"""Phase 7 — Pont 1 : world model tokenisé sur un automate rendu en image.

On dispose du champ de bassins EXACT d'un ECA. On rend chaque état en image, on
entraîne un VQ-VAE (tokeniseur appris, aveugle à la dynamique) sur ces images,
puis on projette la vraie dynamique à travers la tokenisation : le graphe de
transition latent est code[s] -> code[succ s].

Question : le tokeniseur préserve-t-il la structure causale, ou la détruit-il ?
On construit le champ de bassins du graphe latent induit et on le compare au
vrai champ de bassins (AMI). Le graphe latent se LIT (il est induit par la vraie
dynamique + l'encodeur) — il ne s'apprend pas : sur des indices de code
arbitraires, « apprendre » la transition serait mal posé."""

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
CODEBOOK_SIZE = 512
VQVAE_EPOCHS = 100


def reviveDeadCodes(model, data, device, rng):
    """Réinitialise les codes inutilisés vers des latents observés (+ bruit).

    Contre l'effondrement du codebook — pathologie classique du VQ-VAE où
    quelques codes gagnent tôt et les autres, jamais plus proches voisins, ne
    reçoivent plus de gradient et restent morts."""
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
        # réveil des codes morts pendant les 85 % premières epochs
        if epoch < int(epochs * 0.85):
            reviveDeadCodes(model, data, device, rng)
    return model


def latentBasinField(code, succCode):
    """Champ de bassins du graphe de transition latent.

    Le graphe latent : code -> code-successeur majoritaire (la cohérence
    dynamique mesure à quel point ce « majoritaire » est en fait unique).
    Retourne (champ latent, remap code -> index contigu)."""
    usedCodes = sorted({int(c) for c in code})
    remap = {c: i for i, c in enumerate(usedCodes)}
    votes = defaultdict(Counter)
    for state in range(len(code)):
        votes[int(code[state])][int(succCode[state])] += 1
    latentSucc = np.empty(len(usedCodes), dtype=np.int64)
    for c in usedCodes:
        latentSucc[remap[c]] = remap[votes[c].most_common(1)[0][0]]
    return basinFieldFromSuccessor(latentSucc, 0, "latent"), remap


def main():
    device = "cpu"
    field = computeBasinField(RULE, N)
    size = 1 << N
    print(f"règle {RULE}, n={N} | {size} états rendus en images | "
          f"VQ-VAE codebook {CODEBOOK_SIZE}\n")

    images = renderAll(N)
    vqvae = trainVqvae(images, CODEBOOK_SIZE, VQVAE_EPOCHS, device)
    code = vqvae.encodeCodes(images, device)
    succCode = code[field.succ]

    distinctCodes = len(np.unique(code))
    collisionRate = 1.0 - distinctCodes / size

    # cohérence dynamique : un code a-t-il un unique code-successeur ?
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
    basinAmi = float(adjusted_mutual_info_score(field.attractorId, latentBasinOfState))

    print(f"codes distincts utilisés      : {distinctCodes} / {size}")
    print(f"taux de collision             : {collisionRate:.4f}")
    print(f"cohérence dynamique           : {consistent:.4f}")
    print(f"bassins — vrai / latent       : {len(field.attractors)} / "
          f"{len(latentField.attractors)}")
    print(f"AMI bassins (vrai vs latent)  : {basinAmi:.4f}")
    print(
        "\nLecture — le tokeniseur, appris sur les seules images, préserve-t-il"
        "\nla structure causale ?"
        "\n  collision basse + cohérence haute => le tokeniseur préserve l'info"
        "\n     et la dynamique."
        "\n  AMI proche de 1 => le champ de bassins latent EST le vrai champ de"
        "\n     bassins, relabellisé : la tokenisation préserve la structure causale."
        "\n  AMI basse => on analyserait le tokeniseur, pas le monde."
    )


if __name__ == "__main__":
    main()
