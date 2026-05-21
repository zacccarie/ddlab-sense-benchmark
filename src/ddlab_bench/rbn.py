"""Réseau booléen aléatoire (RBN) — le modèle de Kauffman.

N nœuds binaires ; chaque nœud a K entrées tirées au hasard parmi les nœuds et
une fonction booléenne aléatoire (table de vérité de 2^K bits). La topologie est
IRRÉGULIÈRE — contrairement à l'anneau régulier d'un automate cellulaire. K est
le paramètre d'ordre : K=1 ordonné, K=2 critique (bord du chaos), K>=3 chaotique."""

import numpy as np

from ddlab_bench.eca import MAX_ENUMERABLE_N


class RandomBooleanNetwork:
    """Spec d'un RBN : câblage des entrées + fonction booléenne par nœud."""

    def __init__(self, n, k, seed):
        if n > MAX_ENUMERABLE_N:
            raise ValueError(f"n={n} dépasse l'énumération exhaustive (max {MAX_ENUMERABLE_N})")
        if not 1 <= k <= n:
            raise ValueError(f"k={k} hors borne [1, n]")
        rng = np.random.default_rng(seed)
        self.n = n
        self.k = k
        self.seed = seed
        # entrées de chaque nœud : k indices distincts parmi [0, n)
        self.inputs = np.array(
            [rng.choice(n, size=k, replace=False) for _ in range(n)], dtype=np.int64
        )
        # table de vérité de chaque nœud : 2^k sorties binaires
        self.tables = rng.integers(0, 2, size=(n, 1 << k), dtype=np.uint8)

    def stepState(self, state):
        """État suivant : chaque nœud applique sa fonction à ses k entrées."""
        nextState = 0
        for node in range(self.n):
            index = 0
            for j in range(self.k):
                index = (index << 1) | ((state >> int(self.inputs[node, j])) & 1)
            nextState |= int(self.tables[node, index]) << node
        return nextState

    def successorArray(self):
        """Graphe de transition complet : succ[s] pour tout s dans [0, 2^n)."""
        size = 1 << self.n
        succ = np.empty(size, dtype=np.int64)
        for state in range(size):
            succ[state] = self.stepState(state)
        return succ
