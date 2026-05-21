"""§14 suites — stabilisation gamma->1, et évaluation pondérée des bassins.

Partie A : la cible normalisée (1-gamma)*base + gamma*d_succ borne le point
fixe dans [0,1] pour tout gamma. L'objectif de bisimulation reste-t-il stable
(perte finie, AMI tenue) quand gamma -> 1 ? Avant normalisation, gamma=0.9
divergeait.

Partie B : l'objectif d'invariance contre les bassins — AMI standard vs
balancedAmi. La pondération plafonne le bassin géant qui écrase l'AMI (§14)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.basins import computeBasinField
from ddlab_bench.bisimulation import (
    embedAllStates,
    trainBisimulationEncoder,
    trainInvarianceEncoder,
)
from ddlab_bench.metrics import balancedAmi
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.renormalization import bisimulationQuotient, densityLabel

N = 9
EPOCHS = 1500
LATENT_DIM = 32
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]
GAMMAS = [0.5, 0.7, 0.9, 0.95, 0.99]


def amiTo(features, reference, seed=0):
    k = int(reference.max()) + 1
    clusters = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    return float(adjusted_mutual_info_score(reference, clusters))


def main():
    stateFeatures = stateToBits(np.arange(1 << N), N)

    print(f"A — stabilisation γ→1 (cible normalisée) | bisimulation, n={N}")
    print("  cellule = AMI(quotient) / perte finale\n")
    print(f"{'règle':>8} {'régime':>10}  " + "  ".join(f"{'γ=' + str(g):>11}" for g in GAMMAS))
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        label = densityLabel(N)
        quotient = bisimulationQuotient(field.succ, label)
        cells = []
        for gamma in GAMMAS:
            encoder, history = trainBisimulationEncoder(
                stateFeatures, field.succ, label, latentDim=LATENT_DIM,
                gamma=gamma, epochs=EPOCHS, seed=0,
            )
            ami = amiTo(embedAllStates(encoder, stateFeatures), quotient)
            cells.append(f"{ami:.2f}/{history[-1]:.3f}")
        print(f"{rule:>8} {regime:>10}  " + "  ".join(f"{c:>11}" for c in cells))

    print(f"\nB — invariance vs bassins : AMI standard vs pondérée | n={N}\n")
    print(f"{'règle':>8} {'régime':>10} {'#bassins':>9} {'AMI std':>9} {'AMI pondérée':>13}")
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        k = len(field.attractors)
        encoder, _ = trainInvarianceEncoder(
            stateFeatures, field.succ, latentDim=LATENT_DIM, epochs=EPOCHS, seed=0,
        )
        clusters = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(
            embedAllStates(encoder, stateFeatures)
        )
        standard = float(adjusted_mutual_info_score(field.attractorId, clusters))
        balanced = balancedAmi(field.attractorId, clusters, perClass=64, seed=0)
        print(f"{rule:>8} {regime:>10} {k:>9} {standard:>9.3f} {balanced:>13.3f}")


if __name__ == "__main__":
    main()
