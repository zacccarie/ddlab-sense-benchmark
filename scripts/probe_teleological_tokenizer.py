"""Phase 10b — tokeniseur téléologique : casser la similarité perceptuelle locale.

Phase 10 (contrastif 1-pas) a échoué : le positif (s, succ(s)) est
perceptuellement PROCHE, donc l'objectif était satisfiable en embeddant
l'apparence — aucune géométrie de bassin.

Ici le positif est le futur LOINTAIN : (s, succ^H(s)) avec H >= transitoire
maximal, donc succ^H(s) est sur le cycle attracteur de s. Image très
différente, même destin. « z(s) proche de z(succ^H(s)) » ne peut plus être
résolu par l'apparence — l'encodeur est forcé d'embedder le DESTIN. Géométrie
téléologique : la position encode où l'état VA, pas à quoi il ressemble.

Question : k-means sur cet embedding préserve-t-il les bassins jusqu'à K ~ 13 ?"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from ddlab_bench import computeBasinField
from ddlab_bench.basins import iterateSuccessor
from ddlab_bench.models.world_model import ContrastiveEncoder
from ddlab_bench.rendering import renderAll
from ddlab_bench.tokenizer_analysis import metricsForCode

RULE = 110
N = 9
EPOCHS = 150
BATCH = 128
TEMPERATURE = 0.2
K_VALUES = [256, 64, 32, 16, 13, 8, 4]


def trainTeleological(images, farTarget, epochs, device):
    """InfoNCE avec positif = futur lointain (état sur le cycle attracteur)."""
    torch.manual_seed(0)
    model = ContrastiveEncoder(images.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.CrossEntropyLoss()
    data = torch.tensor(images, dtype=torch.float32, device=device)
    farImages = data[farTarget]
    rng = np.random.default_rng(0)
    size = len(data)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(size)
        for start in range(0, size, BATCH):
            idx = torch.from_numpy(order[start:start + BATCH])
            z = model(data[idx])
            zPos = model(farImages[idx])
            logits = (z @ zPos.T) / TEMPERATURE
            loss = lossFn(logits, torch.arange(len(idx), device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def main():
    device = "cpu"
    field = computeBasinField(RULE, N)
    images = renderAll(N)
    horizon = int(field.transient.max()) + 4
    farTarget = iterateSuccessor(field.succ, horizon)

    print(f"règle {RULE}, n={N} | {1 << N} états | tokeniseur TÉLÉOLOGIQUE")
    print(f"positif = futur à H={horizon} (sur le cycle attracteur) | "
          f"vrais bassins : {len(field.attractors)}\n")

    model = trainTeleological(images, farTarget, EPOCHS, device)
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(images, dtype=torch.float32, device=device)
        embedding = model(tensor).cpu().numpy()

    header = f"{'K':>5} {'collision':>10} {'cohérence':>10} {'bassins lat.':>13} {'AMI':>8}"
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        code = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embedding)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['collisionRate']:>9.3f} {m['consistent']:>10.3f} "
              f"{m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — positif perceptuellement LOINTAIN => l'apparence ne peut plus"
        "\nsatisfaire l'objectif. AMI haut jusqu'à K ~ 13 => géométrie téléologique"
        "\napprise, le contrastif découvre le quotient dynamique. AMI ~ 0 => même"
        "\nla rupture perceptuelle ne suffit pas : apprendre le quotient depuis"
        "\nl'image reste un problème ouvert."
    )


if __name__ == "__main__":
    main()
