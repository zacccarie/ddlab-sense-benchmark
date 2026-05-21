"""Coupled Map Lattice (CML) discrétisé — substrat continu rendu énumérable.

Chaque cellule porte une valeur continue dans [0,1], discrétisée en m niveaux
equirépartis. Couplage diffusif sur un anneau (bord périodique) :

  x_i(t+1) = (1-eps) * f(x_i) + (eps/2) * (f(x_{i-1}) + f(x_{i+1}))

avec f la carte logistique f(x) = r*x*(1-x). Le pipeline ddlab-sense-benchmark
exige un espace d'états ENUMERABLE (vérité-terrain exacte). Un CML est continu ;
en discrétisant les valeurs de cellule en m niveaux, l'espace d'états (m^n)
redevient énumérable — basinFieldFromSuccessor s'y branche tel quel, comme pour
les ECA/RBN. Les paramètres de contrôle (r, eps) remplacent le Z de Wuensche :
ils balayent ordre -> bord du chaos -> chaos spatio-temporel (diagramme de phase
de Kaneko)."""

import numpy as np

from ddlab_bench.eca import MAX_ENUMERABLE_N

# plafond d'énumération exhaustive, cohérent avec l'alphabet binaire de eca.py
MAX_ENUMERABLE_STATES = 1 << MAX_ENUMERABLE_N


def logisticMap(x, r):
    """Carte logistique f(x) = r*x*(1-x). Pour r dans [0,4] et x dans [0,1],
    le résultat reste dans [0,1]."""
    return r * x * (1.0 - x)


class CoupledMapLattice:
    """CML discrétisé sur un anneau de n cellules, m niveaux par cellule.

    État = entier en base m : le digit i (poids m^i) porte le niveau de la
    cellule i. Dynamique déterministe -> graphe de transition complet et
    énumérable, branché tel quel sur basinFieldFromSuccessor."""

    def __init__(self, n, m, r, epsilon):
        if n < 1:
            raise ValueError(f"n={n} : au moins 1 cellule requise")
        if m < 2:
            raise ValueError(f"m={m} : au moins 2 niveaux requis")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon={epsilon} hors borne [0,1]")
        if not 0.0 <= r <= 4.0:
            raise ValueError(f"r={r} hors borne [0,4]")
        size = m ** n
        if size > MAX_ENUMERABLE_STATES:
            raise ValueError(
                f"m^n={size} dépasse l'énumération exhaustive (max {MAX_ENUMERABLE_STATES})"
            )
        self.n = n
        self.m = m
        self.r = r
        self.epsilon = epsilon
        # niveau k <-> valeur de cellule k/(m-1), equirépartie de 0 à 1
        self.levelToValue = np.arange(m, dtype=np.float64) / (m - 1)

    def decode(self, state):
        """État entier -> vecteur de n niveaux (digits base m, cellule 0 = poids faible)."""
        digits = np.empty(self.n, dtype=np.int64)
        rem = int(state)
        for i in range(self.n):
            digits[i] = rem % self.m
            rem //= self.m
        return digits

    def encode(self, digits):
        """Vecteur de n niveaux -> état entier."""
        state = 0
        for i in range(self.n - 1, -1, -1):
            state = state * self.m + int(digits[i])
        return state

    def _decodeAll(self):
        """Matrice (m^n, n) de tous les états décodés : digits[s, i] = niveau
        de la cellule i de l'état s."""
        size = self.m ** self.n
        digits = np.empty((size, self.n), dtype=np.int64)
        rem = np.arange(size, dtype=np.int64)
        for i in range(self.n):
            digits[:, i] = rem % self.m
            rem //= self.m
        return digits

    def _quantize(self, values):
        """Valeurs continues dans [0,1] -> niveaux entiers les plus proches."""
        return np.clip(np.round(values * (self.m - 1)).astype(np.int64), 0, self.m - 1)

    def stepState(self, state):
        """État suivant : carte logistique par cellule, couplage diffusif aux
        deux voisins (anneau), re-discrétisation."""
        values = self.levelToValue[self.decode(state)]
        fv = logisticMap(values, self.r)
        coupled = (1.0 - self.epsilon) * fv + 0.5 * self.epsilon * (
            np.roll(fv, 1) + np.roll(fv, -1)
        )
        return self.encode(self._quantize(coupled))

    def successorArray(self):
        """Graphe de transition complet : succ[s] pour tout s dans [0, m^n)."""
        digits = self._decodeAll()
        fv = logisticMap(self.levelToValue[digits], self.r)
        coupled = (1.0 - self.epsilon) * fv + 0.5 * self.epsilon * (
            np.roll(fv, 1, axis=1) + np.roll(fv, -1, axis=1)
        )
        nextDigits = self._quantize(coupled)
        powers = self.m ** np.arange(self.n, dtype=np.int64)
        return (nextDigits * powers).sum(axis=1)

    def densityLabel(self):
        """Observable de base macroscopique : somme des niveaux de cellule
        (champ moyen discrétisé) de chaque état. Analogue m-aire de la densité
        binaire — l'observable grossier du §11, pour le quotient de bisimulation
        relatif au champ moyen."""
        return self._decodeAll().sum(axis=1)
