"""§11 — solidification du contre-test positif.

L'objectif récursif récupère-t-il le quotient dynamique de façon robuste
(multi-seeds, multi-régimes) ? Et embarque-t-il la métrique de bisimulation
EXACTE, pas seulement le partitionnement ?

Pour chaque règle et chaque seed, phi entraîné à gamma=0.5 :
  - AMI(k-means(phi), quotient exact)        -> récupération du partitionnement
  - corr(distances de phi, métrique exacte)  -> récupération de la métrique

Plafond géométrique : corrélation entre la métrique exacte et son meilleur
plongement euclidien en dimension latente (MDS classique). L'écart phi vs
plafond distingue la sous-optimisation de la non-euclidianité de la cible (P5)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.basins import computeBasinField
from ddlab_bench.bisimulation import embedAllStates, trainBisimulationEncoder
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.renormalization import (
    bisimulationMetric,
    bisimulationQuotient,
    densityLabel,
)

RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]
N = 9
GAMMA = 0.5
LATENT_DIM = 32
EPOCHS = 1500
SEEDS = [0, 1, 2, 3, 4]


def pairCorr(a, b):
    """Corrélation de Pearson sur le triangle supérieur strict de deux matrices."""
    upper = np.triu_indices_from(a, k=1)
    return float(np.corrcoef(a[upper], b[upper])[0, 1])


def pairwiseDistance(embedding):
    """Matrice des distances euclidiennes entre toutes les paires d'un plongement."""
    diff = embedding[:, None, :] - embedding[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def classicalMds(distance, dim):
    """Plongement MDS classique (PCoA) — meilleur plongement euclidien en `dim`
    dimensions ; les valeurs propres négatives (part non-euclidienne) sont
    écrêtées à zéro."""
    size = distance.shape[0]
    centering = np.eye(size) - np.ones((size, size)) / size
    gram = -0.5 * centering @ (distance ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:dim]
    return vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))


def main():
    print(f"ECA n={N}, γ={GAMMA}, objectif récursif §11 — {len(SEEDS)} seeds/règle\n")
    print(
        f"{'règle':>6} {'régime':>11} {'#blocs':>7} {'AMI bruts':>10} "
        f"{'AMI récursif':>16} {'corr(φ, d*)':>16} {'plafond MDS':>12}"
    )
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        size = 1 << N
        label = densityLabel(N)
        quotient = bisimulationQuotient(field.succ, label)
        k = int(quotient.max()) + 1
        exactMetric = bisimulationMetric(field.succ, label, gamma=GAMMA)
        ceiling = pairCorr(
            pairwiseDistance(classicalMds(exactMetric, LATENT_DIM)), exactMetric
        )

        stateFeatures = stateToBits(np.arange(size), N)
        rawClusters = KMeans(k, random_state=0, n_init=10).fit_predict(stateFeatures)
        rawAmi = float(adjusted_mutual_info_score(quotient, rawClusters))

        amis, corrs = [], []
        for seed in SEEDS:
            encoder, _ = trainBisimulationEncoder(
                stateFeatures, field.succ, label, latentDim=LATENT_DIM,
                gamma=GAMMA, epochs=EPOCHS, seed=seed,
            )
            embedding = embedAllStates(encoder, stateFeatures)
            clusters = KMeans(k, random_state=0, n_init=10).fit_predict(embedding)
            amis.append(float(adjusted_mutual_info_score(quotient, clusters)))
            corrs.append(pairCorr(pairwiseDistance(embedding), exactMetric))
        amis, corrs = np.array(amis), np.array(corrs)
        amiStr = f"{amis.mean():.3f}±{amis.std():.3f}"
        corrStr = f"{corrs.mean():.3f}±{corrs.std():.3f}"
        print(
            f"{rule:>6} {regime:>11} {k:>7} {rawAmi:>10.3f} "
            f"{amiStr:>16} {corrStr:>16} {ceiling:>12.3f}"
        )

    print(
        "\nLecture : AMI récursif >> AMI bruts, stable sur les seeds = contre-test"
        " solidifié.\ncorr(φ, d*) proche du plafond MDS = φ est optimal pour un"
        " plongement euclidien ;\nl'écart plafond -> 1.0 mesure la non-euclidianité"
        " de la métrique de bisimulation (P5)."
    )


if __name__ == "__main__":
    main()
