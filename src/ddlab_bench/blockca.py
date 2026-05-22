"""Automate cellulaire par blocs à deux échelles — un système naturellement
hiérarchique, à vérité-terrain exacte (§23, le « gem »).

Les expériences de récupération du papier 2 reposent sur des substrats plantés
ou semi-plantés : on y valide qu'une méthode retrouve une hiérarchie *connue
comme présente*, non qu'une hiérarchie *surgisse sans qu'on l'ait posée*. Le CA
par blocs lève cette limite — sa hiérarchie d'échelle est inscrite dans la
RÈGLE, pas plantée dans la partition.

Construction. n = nBlocks × blockSize cellules, réparties en nBlocks blocs.
Dans chaque bloc, la cellule 0 est GROSSIÈRE, les cellules 1..blockSize-1 sont
FINES.
  - Couche grossière — les nBlocks cellules grossières forment un anneau et
    suivent une règle ECA : dynamique AUTONOME, qui ne lit qu'elle-même.
  - Couche fine — dans chaque bloc, les cellules fines forment un anneau et
    suivent une règle ECA SÉLECTIONNÉE par la cellule grossière du bloc
    (fineRule0 si elle vaut 0, fineRule1 sinon).

C'est un produit semi-direct : le grossier pilote le fin sans en dépendre.
Conséquence exacte — la projection sur la couche grossière est une
sous-dynamique FERMÉE, π(succ(x)) ne dépendant que de π(x). Le quotient de
bisimulation relatif à tout observable du grossier est donc EXACTEMENT la
partition par bits grossiers : 2^nBlocks blocs, compatibles avec la dynamique,
strictement plus grossiers que l'état complet — un coarse-graining que la règle
PRODUIT, là où le papier 2 le plantait. La tour de renormalisation

    état complet  >  partition grossière  >  quotient densité-grossière

est vérifiable étage par étage contre l'énumération exhaustive.

Un état est encodé comme un entier dont le bit i porte la cellule i — la
convention de `eca.py`. Index global de la cellule (bloc j, position p) :
j * blockSize + p."""

import numpy as np

from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.eca import MAX_ENUMERABLE_N, ruleTable


def _ringStep(cells, table):
    """Un pas d'ECA (rayon 1, anneau périodique) sur un array de cellules 0/1.

    L'anneau peut être de longueur quelconque ; pour 1 ou 2 cellules les
    voisins gauche et droit coïncident — la dynamique reste bien définie."""
    cells = np.asarray(cells, dtype=np.int64)
    if len(cells) == 0:
        return cells
    left = np.roll(cells, 1)
    right = np.roll(cells, -1)
    return table[(left << 2) | (cells << 1) | right].astype(np.int64)


class BlockCellularAutomaton:
    """CA par blocs à deux échelles : couche grossière autonome, couche fine pilotée.

    nBlocks blocs de blockSize cellules. Dans chaque bloc, la cellule 0 est
    grossière, les cellules 1..blockSize-1 sont fines. La cellule grossière du
    bloc sélectionne la règle ECA appliquée à ses cellules fines ; la couche
    grossière, elle, suit `coarseRule` seule — sans jamais lire le fin."""

    def __init__(self, nBlocks, blockSize, coarseRule, fineRule0, fineRule1):
        if blockSize < 2:
            raise ValueError(
                f"blockSize={blockSize} < 2 : il faut au moins une cellule fine")
        if nBlocks < 1:
            raise ValueError(f"nBlocks={nBlocks} < 1")
        n = nBlocks * blockSize
        if n > MAX_ENUMERABLE_N:
            raise ValueError(
                f"n={n} dépasse l'énumération exhaustive (max {MAX_ENUMERABLE_N})")
        self.nBlocks = nBlocks
        self.blockSize = blockSize
        self.n = n
        self.coarseRule = coarseRule
        self.fineRule0 = fineRule0
        self.fineRule1 = fineRule1
        self.coarseTable = ruleTable(coarseRule)
        self.fineTable0 = ruleTable(fineRule0)
        self.fineTable1 = ruleTable(fineRule1)

    def stepState(self, state):
        """État suivant : couche grossière (autonome) puis couche fine (pilotée)."""
        b, k = self.blockSize, self.nBlocks
        coarse = np.array([(state >> (j * b)) & 1 for j in range(k)], dtype=np.int64)
        coarseNext = _ringStep(coarse, self.coarseTable)
        nextState = 0
        for j in range(k):
            nextState |= int(coarseNext[j]) << (j * b)
            fine = np.array(
                [(state >> (j * b + 1 + q)) & 1 for q in range(b - 1)], dtype=np.int64)
            fineTable = self.fineTable1 if coarse[j] else self.fineTable0
            fineNext = _ringStep(fine, fineTable)
            for q in range(b - 1):
                nextState |= int(fineNext[q]) << (j * b + 1 + q)
        return nextState

    def successorArray(self):
        """Graphe de transition complet : succ[s] pour tout s dans [0, 2^n)."""
        size = 1 << self.n
        succ = np.empty(size, dtype=np.int64)
        for state in range(size):
            succ[state] = self.stepState(state)
        return succ


def computeBlockCaField(nBlocks, blockSize, coarseRule, fineRule0, fineRule1):
    """Champ de bassins d'un CA par blocs, par énumération exhaustive (2^n états)."""
    ca = BlockCellularAutomaton(nBlocks, blockSize, coarseRule, fineRule0, fineRule1)
    label = f"block-{nBlocks}x{blockSize}-{coarseRule}/{fineRule0}/{fineRule1}"
    return basinFieldFromSuccessor(ca.successorArray(), ca.n, label)


def coarseProjection(nBlocks, blockSize):
    """Observable lisant la couche grossière : coarseBits[s] = entier formé des
    nBlocks cellules grossières de s (bit j = cellule grossière du bloc j).

    La couche grossière étant une sous-dynamique autonome, le quotient de
    bisimulation relatif à cet observable est exactement cette partition —
    2^nBlocks blocs, le coarse-graining que la règle produit (§23)."""
    n = nBlocks * blockSize
    states = np.arange(1 << n, dtype=np.int64)
    proj = np.zeros(1 << n, dtype=np.int64)
    for j in range(nBlocks):
        proj |= ((states >> (j * blockSize)) & 1) << j
    return proj


def coarseDensityLabel(nBlocks, blockSize):
    """Observable de base grossier : densité de la couche grossière — nombre de
    cellules grossières à 1. Plus grossier que `coarseProjection`, il fournit le
    troisième étage candidat de la tour de renormalisation."""
    proj = coarseProjection(nBlocks, blockSize)
    return np.array([bin(int(c)).count("1") for c in proj], dtype=np.int64)
