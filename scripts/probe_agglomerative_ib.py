"""Hiérarchie de renormalisation émergente — IB agglomératif (Directions 1 + 2).

Le §14 CHOISIT l'observable ℓ. Ici ℓ ÉMERGE. On donne une variable pertinente Y
(le futur de l'état) et l'IB agglomératif construit le dendrogramme Q_0 ⊃ Q_1 ⊃
... par fusions gloutonnes minimisant la perte d'information pertinente I(Z;Y).

Deux choix de Y, deux ℓ émergents :
  - Y = destin (l'attracteur atteint) : test propre — la partition en bassins
    doit émerger comme le palier de coût nul (compression IB sans perte) ;
  - Y = densité du futur lointain : un observable grossier et causal — la
    hiérarchie émergente n'est plus les bassins mais le quotient densité.

I(Z;Y) est exacte (espace énumérable, p(x,y) exacte ; aucun estimateur)."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import computeBasinField
from ddlab_bench.agglomerative_ib import agglomerativeIB, partitionAt
from ddlab_bench.basins import iterateSuccessor
from ddlab_bench.renormalization import densityLabel

N = 9
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]
HORIZON = 1 << N  # >= tout transitoire : succ^H projette sur le cycle attracteur


def _oneHot(labels):
    labels = np.asarray(labels)
    numClasses = int(labels.max()) + 1
    out = np.zeros((len(labels), numClasses))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def main():
    print(f"IB agglomératif — hiérarchie émergente | n={N}, MI exacte\n")
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        size = 1 << N
        nBasins = len(field.attractors)
        farFuture = iterateSuccessor(field.succ, HORIZON)
        print(f"règle {rule} ({regime}) — {nBasins} bassins")

        # Y = destin : la partition en bassins doit émerger sans perte
        mergesFate, assignFate = agglomerativeIB(_oneHot(field.attractorId))
        intraCost = max((c for c, _ in mergesFate[: size - nBasins]), default=0.0)
        emergedFate = partitionAt(assignFate, nBasins)
        amiFate = adjusted_mutual_info_score(field.attractorId, emergedFate)
        print(f"  Y = destin          : palier coût nul à {nBasins} blocs "
              f"(coût max intra {intraCost:.2e}) ; AMI(émergé, bassins) = {amiFate:.3f}")

        # Y = densité du futur lointain : observable grossier et causal
        densFuture = densityLabel(N)[farFuture]
        mergesDens, assignDens = agglomerativeIB(_oneHot(densFuture))
        costs = np.array([c for c, _ in mergesDens])
        infos = np.array([info for _, info in mergesDens])
        # niveau naturel = plus grand saut de coût ; partition juste avant
        jump = int(np.argmax(costs))
        natBlocks = size - jump
        amiNat = adjusted_mutual_info_score(field.attractorId, partitionAt(assignDens, natBlocks))
        amiAtBasins = adjusted_mutual_info_score(
            field.attractorId, partitionAt(assignDens, nBasins))
        infoAtBasins = infos[size - nBasins - 1] if size - nBasins - 1 < len(infos) else 0.0
        print(f"  Y = densité (succ^H): niveau naturel à {natBlocks} blocs ; "
              f"AMI(niveau, bassins) = {amiNat:.3f}")
        print(f"                        à {nBasins} blocs : AMI(bassins) = {amiAtBasins:.3f}, "
              f"I(Z;Y) retenue = {infoAtBasins:.3f} bits\n")

    print(
        "Lecture :"
        "\n  Y = destin   : AMI ~ 1 attendu — les bassins ÉMERGENT comme la"
        "\n                 compression IB sans perte (la macro-variable n'est"
        "\n                 pas imposée, elle se découvre)."
        "\n  Y = densité  : la hiérarchie émergente dépend de Y — un observable"
        "\n                 grossier fait émerger un quotient distinct des bassins."
    )


if __name__ == "__main__":
    main()
