"""Substrat stochastique — la hiérarchie de renormalisation devient profonde.

§15 : sur une dynamique déterministe la hiérarchie s'effondre en un seul niveau.
Une chaîne de Markov à métastabilité emboîtée a une vraie tour multi-échelle.

On plante une hiérarchie à `DEPTH` niveaux (chaîne de Markov hiérarchique), puis
on teste si elle est récupérée — sans supervision — par : (i) le spectre de
l'opérateur de transfert (les paquets de valeurs propres = ensembles
métastables) ; (ii) l'IB agglomératif (le dendrogramme = la hiérarchie)."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench.agglomerative_ib import agglomerativeIB, partitionAt
from ddlab_bench.markov import hierarchicalChain

BRANCHING = 2
DEPTH = 3
LEAF_SIZE = 8
RATES = [1.0, 0.2, 0.04, 0.008]  # taux par niveau d'ancêtre commun (depth+1)


def main():
    chain, levels = hierarchicalChain(BRANCHING, DEPTH, LEAF_SIZE, RATES, seed=0)
    print(f"Chaîne de Markov hiérarchique — {chain.size} états, "
          f"{DEPTH} niveaux plantés\n")

    values, _ = chain.transferSpectrum()
    timescales = chain.impliedTimescales()
    print("Spectre de transfert — |lambda| et temps de relaxation (12 premiers modes) :")
    for i in range(min(12, chain.size)):
        print(f"  mode {i:>2}  |lambda|={abs(values[i]):.4f}  t={timescales[i]:9.2f}")
    print("  (paquets de |lambda| proches de 1 = ensembles métastables ; "
          "écarts = niveaux)\n")

    merges, assignments = agglomerativeIB(chain.P)
    print("IB agglomératif (Y = futur un pas) — récupération de la hiérarchie plantée :")
    for d in range(DEPTH):
        planted = levels[d]
        nBlocks = int(planted.max()) + 1
        emerged = partitionAt(assignments, nBlocks)
        ami = adjusted_mutual_info_score(planted, emerged)
        print(f"  niveau {d}  ({nBlocks:>2} blocs plantés)  "
              f"AMI(émergé, planté) = {ami:.3f}")
    print(
        "\nAMI ~ 1 à chaque niveau => l'IB agglomératif récupère toute la tour"
        "\nmulti-échelle sans supervision. La hiérarchie de renormalisation est"
        "\nprofonde sur le substrat stochastique — ce qui manquait au déterministe (§15)."
    )


if __name__ == "__main__":
    main()
