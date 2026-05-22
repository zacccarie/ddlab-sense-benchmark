"""CA par blocs à deux échelles — la hiérarchie est dans la RÈGLE (§23, le « gem »).

Le papier 2 récupère des hiérarchies PLANTÉES : on y valide qu'une méthode
retrouve une tour *connue comme présente*. Le CA par blocs lève cette limite —
sa hiérarchie d'échelle n'est pas plantée, elle est une conséquence exacte du
produit semi-direct de la règle : couche grossière autonome, couche fine
pilotée. Sur l'espace d'états énuméré, on vérifie que :

  (i)  le quotient de bisimulation relatif à l'observable grossier est
       EXACTEMENT la partition par bits grossiers (2^nBlocks blocs) — le
       coarse-graining que la règle produit ;
  (ii) l'IB agglomératif, NON SUPERVISÉ, en récupère les étages : bassins de la
       couche grossière (sous-dynamique autonome) et bassins complets émergent
       comme paliers de coût nul.

Tout est exact — espace énumérable, I(Z;Y) sans estimateur."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import buildSuccessorArray
from ddlab_bench.agglomerative_ib import agglomerativeIB, partitionAt
from ddlab_bench.basins import findAttractors
from ddlab_bench.blockca import BlockCellularAutomaton, coarseProjection
from ddlab_bench.renormalization import bisimulationQuotient

# (nBlocks, blockSize, coarseRule, fineRule0, fineRule1, étiquette du régime grossier)
CONFIGS = [
    (3, 3, 45, 30, 110, "chaotique"),
    (3, 3, 90, 30, 110, "XOR"),
    (3, 3, 76, 30, 110, "fragmenté"),
]


def _oneHot(labels):
    labels = np.asarray(labels)
    out = np.zeros((len(labels), int(labels.max()) + 1))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def main():
    print("CA par blocs à deux échelles — la hiérarchie est dans la règle (§23)\n")
    for nBlocks, blockSize, coarseRule, fine0, fine1, regime in CONFIGS:
        ca = BlockCellularAutomaton(nBlocks, blockSize, coarseRule, fine0, fine1)
        succ = ca.successorArray()
        size = 1 << ca.n
        proj = coarseProjection(nBlocks, blockSize)

        # (i) le quotient de bisimulation EST la partition grossière
        q = bisimulationQuotient(succ, proj)
        nCoarse = int(q.max()) + 1
        _, projRelabel = np.unique(proj, return_inverse=True)
        exact = np.array_equal(q, projRelabel)

        # étages de la tour : bassins grossiers (couche autonome), bassins complets
        coarseAttId, coarseAtt = findAttractors(buildSuccessorArray(coarseRule, nBlocks))
        coarseFate = coarseAttId[proj]
        fullAttId, fullAtt = findAttractors(succ)

        # (ii) l'IB agglomératif récupère ces étages — paliers de coût nul
        mergesC, assignC = agglomerativeIB(_oneHot(coarseFate))
        costC = max((c for c, _ in mergesC[: size - len(coarseAtt)]), default=0.0)
        amiC = adjusted_mutual_info_score(coarseFate, partitionAt(assignC, len(coarseAtt)))
        _, assignF = agglomerativeIB(_oneHot(fullAttId))
        amiF = adjusted_mutual_info_score(fullAttId, partitionAt(assignF, len(fullAtt)))

        print(f"config {nBlocks}×{blockSize}  coarse={coarseRule} fine={fine0}/{fine1}"
              f"  (grossier {regime}) — n={ca.n}, {size} états")
        print(f"  (i)  quotient bisim / grossier : {nCoarse} blocs = 2^{nBlocks} ; "
              f"partition exacte : {exact}")
        print(f"  tour de renormalisation : {size} (état) > {nCoarse} (grossière) "
              f"> {len(coarseAtt)} (bassins grossiers)")
        print(f"  (ii) IB Y=destin grossier : palier coût nul à {len(coarseAtt)} blocs "
              f"(coût max intra {costC:.1e}) ; AMI = {amiC:.3f}")
        print(f"       IB Y=destin complet  : {len(fullAtt)} bassins ; "
              f"AMI = {amiF:.3f}\n")

    print(
        "Lecture : la hiérarchie n'est pas plantée — elle est une conséquence\n"
        "exacte du produit semi-direct de la règle (grossier autonome, fin\n"
        "piloté). Le quotient de bisimulation la récupère exactement (i) ; l'IB\n"
        "agglomératif, non supervisé, en retrouve les étages (ii). C'est le banc\n"
        "d'essai que le papier 2 appelait — une tour qui SURGIT de la règle, non\n"
        "posée à la main, et auditée exactement contre l'énumération."
    )


if __name__ == "__main__":
    main()
