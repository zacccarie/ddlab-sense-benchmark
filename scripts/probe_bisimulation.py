"""§11 — les objectifs dynamiques contre l'objectif perceptuel (axe C).

Le §9 montre qu'aucun objectif perceptuel ne récupère la structure dynamique
(AMI ~ 0). Le §11 propose des objectifs RÉCURSIFS, fondés sur la dynamique :

  - bisimulation (récursif, gamma > 0) : équation de point fixe de la métrique
    de bisimulation -> récupère le QUOTIENT de bisimulation ;
  - invariance (Koopman-1) : phi(x) ~= phi(f(x)) -> récupère les BASSINS
    (l'espace propre de valeur propre 1 de Koopman = indicatrices de bassin).

On entraîne chacun sur ECA, on score son plongement (k-means + AMI) contre le
quotient de bisimulation exact ET la partition en bassins. Référence basse :
bits bruts — la cible perceptuelle du §9."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.basins import computeBasinField
from ddlab_bench.bisimulation import (
    embedAllStates,
    trainBisimulationEncoder,
    trainInvarianceEncoder,
)
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.renormalization import bisimulationQuotient, densityLabel

RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]
N = 10
EPOCHS = 2000
LATENT_DIM = 32


def amiTo(features, reference):
    """AMI entre un clustering k-means de `features` et une partition cible."""
    k = int(reference.max()) + 1
    if k < 2:
        return float("nan")
    clusters = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(features)
    return float(adjusted_mutual_info_score(reference, clusters))


def main():
    stateFeatures = stateToBits(np.arange(1 << N), N)
    print(f"§11 — objectifs dynamiques vs perceptuel | n={N}, {EPOCHS} epochs\n")
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        label = densityLabel(N)
        quotient = bisimulationQuotient(field.succ, label)
        basins = field.attractorId
        ceiling = adjusted_mutual_info_score(quotient, basins)

        encBase, _ = trainBisimulationEncoder(
            stateFeatures, field.succ, label, latentDim=LATENT_DIM,
            gamma=0.0, epochs=EPOCHS,
        )
        encRec, _ = trainBisimulationEncoder(
            stateFeatures, field.succ, label, latentDim=LATENT_DIM,
            gamma=0.5, epochs=EPOCHS,
        )
        encInv, _ = trainInvarianceEncoder(
            stateFeatures, field.succ, latentDim=LATENT_DIM, epochs=EPOCHS,
        )
        rows = [
            ("bits bruts (perceptuel)", stateFeatures),
            ("base seule  gamma=0", embedAllStates(encBase, stateFeatures)),
            ("bisimulation  gamma=0.5", embedAllStates(encRec, stateFeatures)),
            ("invariance (Koopman-1)", embedAllStates(encInv, stateFeatures)),
        ]

        print(
            f"règle {rule} ({regime}) — quotient {int(quotient.max()) + 1} blocs, "
            f"{len(field.attractors)} bassins  [AMI(quotient,bassins)={ceiling:.3f}]"
        )
        print(f"  {'objectif':>26} {'AMI(quotient)':>14} {'AMI(bassins)':>13}")
        for name, emb in rows:
            print(
                f"  {name:>26} {amiTo(emb, quotient):>14.3f} "
                f"{amiTo(emb, basins):>13.3f}"
            )
        print()
    print(
        "Attendu : bisimulation récursif -> quotient ; invariance -> bassins ;"
        "\nperceptuel (bits bruts) -> ni l'un ni l'autre — la barrière du §9."
    )


if __name__ == "__main__":
    main()
