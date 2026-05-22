"""§11 sur le substrat de Newton — l'objectif récursif tient-il sur un substrat
fractal continu ?

On discrétise le flot de Newton en un graphe de transition (carte quantifiée),
on entraîne l'objectif de bisimulation récursif dessus, et on score son
plongement contre le bassin EXACT (les 3 racines) et le quotient de
bisimulation. Référence basse : k-means sur la position (Re, Im) — l'objectif
perceptuel.

La quantification n'affecte que le graphe entraîné ; le bassin (vérité-terrain)
reste exact. Question : §11 tient-il hors des automates, sur du continu
fractal ?"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.bisimulation import embedAllStates, trainBisimulationEncoder
from ddlab_bench.newton import quantizedSuccessor
from ddlab_bench.renormalization import bisimulationQuotient

LAM = 1.0
RESOLUTION = 40
COARSE = 6        # observable de base : bin spatial grossier COARSE x COARSE
EPOCHS = 1200
LATENT_DIM = 24


def amiTo(features, reference, seed=0):
    """AMI entre un clustering k-means de `features` et une partition cible."""
    k = int(reference.max()) + 1
    clusters = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    return float(adjusted_mutual_info_score(reference, clusters))


def main():
    succ, positions, basins = quantizedSuccessor(LAM, RESOLUTION)
    size = RESOLUTION * RESOLUTION

    # observable de base : bin spatial grossier (position perceptuelle grossière)
    binRe = np.clip(((positions[:, 0] + 2.0) / 4.0 * COARSE).astype(np.int64), 0, COARSE - 1)
    binIm = np.clip(((positions[:, 1] + 2.0) / 4.0 * COARSE).astype(np.int64), 0, COARSE - 1)
    coarseBin = binRe * COARSE + binIm
    quotient = bisimulationQuotient(succ, coarseBin)

    print(f"Newton (lambda={LAM}), grille {RESOLUTION}x{RESOLUTION} = {size} états")
    print(
        f"  quotient de bisimulation : {int(quotient.max()) + 1} blocs ; "
        f"bassins exacts : {int(basins.max()) + 1}\n"
    )
    print(f"{'objectif':>26} {'AMI(bassins)':>14} {'AMI(quotient)':>15}")
    print("-" * 57)
    print(
        f"{'position (perceptuel)':>26} {amiTo(positions, basins):>14.3f} "
        f"{amiTo(positions, quotient):>15.3f}"
    )
    for gamma in (0.0, 0.7):
        encoder, _ = trainBisimulationEncoder(
            positions, succ, coarseBin, latentDim=LATENT_DIM,
            gamma=gamma, epochs=EPOCHS, seed=0,
        )
        embedding = embedAllStates(encoder, positions)
        tag = f"bisimulation γ={gamma}" if gamma > 0 else f"base seule γ={gamma}"
        print(
            f"{tag:>26} {amiTo(embedding, basins):>14.3f} "
            f"{amiTo(embedding, quotient):>15.3f}"
        )
    print(
        "\nLecture : si bisimulation γ=0.7 dépasse la position perceptuelle,"
        "\nl'objectif récursif récupère la structure causale sur un substrat"
        "\nfractal continu — le §11 tient hors des automates cellulaires."
    )


if __name__ == "__main__":
    main()
