"""Objectif récursif sur un transformer de trajectoire (voie C) — vers les world models.

Le §11/§14 a récupéré le quotient de bisimulation avec un encodeur MLP qui ne
lit que l'état. Ici : le même objectif récursif, mais l'encodeur est un petit
TRANSFORMER qui lit une fenêtre d'orbite — une séquence temporelle, l'architecture
des world models. L'environnement reste énumérable (ECA) : le quotient exact est
connu, l'audit reste exact. On compare MLP vs transformer, à objectif identique."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import computeBasinField
from ddlab_bench.bisimulation import (
    embedAllStates,
    orbitWindows,
    trainBisimulationEncoder,
    trainTrajectoryTransformer,
)
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.renormalization import bisimulationQuotient, densityLabel

N = 9
WINDOW = 8
EPOCHS = 500
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]


def amiTo(features, reference, seed=0):
    """AMI entre un clustering k-means des features et une partition cible."""
    k = int(reference.max()) + 1
    clusters = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    return float(adjusted_mutual_info_score(reference, clusters))


def main():
    print(f"Objectif récursif §11 — MLP vs transformer de trajectoire | "
          f"n={N}, fenêtre={WINDOW}, {EPOCHS} epochs\n")
    size = 1 << N
    stateBits = stateToBits(np.arange(size), N)
    label = densityLabel(N)
    print(f"  {'règle':>7} {'régime':>10} {'#quot':>6} "
          f"{'AMI MLP':>9} {'AMI transformer':>17}")
    print("  " + "-" * 54)
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        quotient = bisimulationQuotient(field.succ, label)

        mlp, _ = trainBisimulationEncoder(
            stateBits, field.succ, label, gamma=0.7, epochs=EPOCHS, seed=0)
        amiMlp = amiTo(embedAllStates(mlp, stateBits), quotient)

        orbits = orbitWindows(field.succ, N, WINDOW)
        transformer, _ = trainTrajectoryTransformer(
            orbits, field.succ, label, gamma=0.7, epochs=EPOCHS, seed=0)
        amiTr = amiTo(embedAllStates(transformer, orbits), quotient)

        print(f"  {rule:>7} {regime:>10} {int(quotient.max()) + 1:>6} "
              f"{amiMlp:>9.3f} {amiTr:>17.3f}")
    print(
        "\nLecture : le transformer de trajectoire — l'architecture des world"
        "\nmodels, une séquence temporelle — atteint-il le quotient comme le MLP"
        "\ndu §11 ? Si oui, l'objectif récursif transfère à l'architecture"
        "\nséquentielle, audité exactement sur un environnement énumérable."
    )


if __name__ == "__main__":
    main()
