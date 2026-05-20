"""Automate cellulaire élémentaire (ECA): table de règle, dynamique, énumération.

ECA = automate 1D binaire, voisinage de rayon 1 (3 cellules), bord périodique.
Un état est encodé comme un entier dont le bit i porte la cellule i."""

import numpy as np

MAX_ENUMERABLE_N = 22


def ruleTable(ruleNumber):
    """Table de transition d'une règle ECA (0-255).

    Index = motif de voisinage (gauche << 2 | centre << 1 | droite), 0-7.
    Retourne un array de 8 sorties binaires."""
    if not 0 <= ruleNumber <= 255:
        raise ValueError(f"règle ECA hors borne [0,255]: {ruleNumber}")
    return np.array([(ruleNumber >> i) & 1 for i in range(8)], dtype=np.uint8)


def stepState(state, table, n):
    """Applique une règle ECA à un état de n cellules (bord périodique)."""
    nextState = 0
    for i in range(n):
        left = (state >> ((i - 1) % n)) & 1
        center = (state >> i) & 1
        right = (state >> ((i + 1) % n)) & 1
        neighborhood = (left << 2) | (center << 1) | right
        nextState |= int(table[neighborhood]) << i
    return nextState


def buildSuccessorArray(ruleNumber, n):
    """Graphe de transition complet: succ[s] = état suivant de s, pour tout s de [0, 2^n).

    Énumération exhaustive — exige un espace d'états énumérable (n <= MAX_ENUMERABLE_N)."""
    if n > MAX_ENUMERABLE_N:
        raise ValueError(f"n={n} dépasse l'énumération exhaustive (max {MAX_ENUMERABLE_N})")
    table = ruleTable(ruleNumber)
    size = 1 << n
    succ = np.empty(size, dtype=np.int64)
    for state in range(size):
        succ[state] = stepState(state, table, n)
    return succ
