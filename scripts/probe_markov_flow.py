"""Le flot de renormalisation en temps de Markov — la hiérarchie comme un flot.

§16 : un spectre STATIQUE récupère une hiérarchie plantée. Ici la hiérarchie
est un FLOT : on balaie le temps de marche t, et la partition métastable se
grossit à mesure que t croît — Q(t=1) raffine Q(t=4) raffine Q(t=16)... Le
temps de Markov joue le rôle de paramètre de renormalisation du §11.

Substrat : marche aléatoire sur un graphe modulaire hiérarchique ÉCHANTILLONNÉ
(la topologie est plantée, la métastabilité émerge de la dynamique de marche).
Pour chaque t : IB agglomératif sur P^t, AMI de la partition contre chaque
niveau planté."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.agglomerative_ib import agglomerativeIB, partitionAt
from ddlab_bench.markov import hierarchicalGraphChain

BRANCHING = 2
DEPTH = 3
LEAF_SIZE = 8
EDGE_PROBS = [0.9, 0.3, 0.06, 0.01]  # proba d'arête par niveau d'ancêtre commun
MARKOV_TIMES = [1, 2, 4, 8, 16, 32, 64]


def main():
    chain, levels = hierarchicalGraphChain(
        BRANCHING, DEPTH, LEAF_SIZE, EDGE_PROBS, seed=0
    )
    nBlocks = [int(lv.max()) + 1 for lv in levels]  # ex. [8, 4, 2]
    print(f"Flot de renormalisation en temps de Markov — {chain.size} états, "
          f"niveaux plantés {nBlocks}\n")
    header = f"  {'t':>5}" + "".join(
        f"  AMI niv.{d} ({nBlocks[d]:>2} bl.)" for d in range(DEPTH))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in MARKOV_TIMES:
        _, assignments = agglomerativeIB(chain.multiStep(t))
        cells = [
            adjusted_mutual_info_score(levels[d], partitionAt(assignments, nBlocks[d]))
            for d in range(DEPTH)
        ]
        print(f"  {t:>5}" + "".join(f"  {c:>16.3f}" for c in cells))
    print(
        "\nLecture : à t petit, le niveau fin (feuilles) est récupéré ; à mesure"
        "\nque t croît, le mélange intra-communauté efface les niveaux fins l'un"
        "\naprès l'autre et seuls les niveaux grossiers survivent. La hiérarchie"
        "\nest un FLOT — chaque niveau a sa fenêtre de temps de Markov ; le temps"
        "\njoue le rôle de paramètre de renormalisation (§11)."
    )


if __name__ == "__main__":
    main()
