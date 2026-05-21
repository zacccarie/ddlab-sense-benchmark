"""Phase 10 — tokeniseur appris par objectif CONTRASTIF temporel.

Phase 9 a montré que la tokenisation compressée-préservante existe (le quotient
dynamique), mais que les objectifs appris naïfs (reconstruction, prédiction
1-pas) ne la découvrent pas — ils compressent par apparence. Question : un
objectif CONTRASTIF temporel la fait-il émerger ?

Paire positive (s, succ(s)) : même trajectoire, donc même bassin (garanti).
Négatifs : autres états du batch. La chaîne s ~ succ(s) ~ succ^2(s) ... tire tout
un bassin en une grappe ; les négatifs empêchent l'effondrement trivial. On
entraîne l'encodeur (InfoNCE), puis on quantifie l'embedding par k-means
(balayage K) et on mesure la préservation des bassins.

Vrais bassins = 13. Si l'AMI tient jusqu'à K ~ 13 puis chute, le contrastif a
appris le quotient : il compresse jusqu'au nombre de bassins sans détruire le
sens — là où reconstruction et 1-pas s'effondrent dès K=256."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from ddlab_bench import computeBasinField
from ddlab_bench.models.world_model import ContrastiveEncoder
from ddlab_bench.rendering import renderAll
from ddlab_bench.tokenizer_analysis import metricsForCode

RULE = 110
N = 9
EPOCHS = 120
BATCH = 128
TEMPERATURE = 0.2
K_VALUES = [256, 64, 32, 16, 13, 8, 4]


def trainContrastive(images, succ, epochs, device):
    """Entraîne l'encodeur par InfoNCE temporel : positif = image du successeur."""
    torch.manual_seed(0)
    model = ContrastiveEncoder(images.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.CrossEntropyLoss()
    data = torch.tensor(images, dtype=torch.float32, device=device)
    succImages = data[succ]
    rng = np.random.default_rng(0)
    size = len(data)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(size)
        for start in range(0, size, BATCH):
            idx = torch.from_numpy(order[start:start + BATCH])
            z = model(data[idx])
            zPos = model(succImages[idx])
            logits = (z @ zPos.T) / TEMPERATURE
            loss = lossFn(logits, torch.arange(len(idx), device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def embedAll(model, images, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(images, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy()


def main():
    device = "cpu"
    field = computeBasinField(RULE, N)
    images = renderAll(N)
    print(f"règle {RULE}, n={N} | {1 << N} états | tokeniseur CONTRASTIF temporel")
    print(f"vrais bassins : {len(field.attractors)}\n")

    model = trainContrastive(images, field.succ, EPOCHS, device)
    embedding = embedAll(model, images, device)

    header = f"{'K':>5} {'collision':>10} {'cohérence':>10} {'bassins lat.':>13} {'AMI':>8}"
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        code = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embedding)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['collisionRate']:>9.3f} {m['consistent']:>10.3f} "
              f"{m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — vrais bassins = 13. AMI haut jusqu'à K ~ 13 puis chute =>"
        "\nle contrastif a appris le quotient dynamique (compression jusqu'au"
        "\nnombre de bassins, sens préservé). À comparer aux falaises K=256 de"
        "\nla reconstruction (Phase 7) et du 1-pas (Phase 8)."
    )


if __name__ == "__main__":
    main()
