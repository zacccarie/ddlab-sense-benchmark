"""Substrat dynamique stochastique — chaînes de Markov énumérables.

Les automates déterministes (§1-§15) ont une hiérarchie de renormalisation
MINCE : pour une dynamique déterministe, p(y|x) est un Dirac, le dendrogramme
s'effondre en un seul niveau (les bassins). Une dynamique STOCHASTIQUE a une
vraie hiérarchie multi-échelle — ensembles métastables emboîtés, temps de
relaxation étagés.

Une chaîne de Markov sur un espace d'états énumérable garde la vérité-terrain
exacte — la matrice de transition EST le système ; spectre, mesure stationnaire,
puissances P^k sont exacts, sans échantillonnage — tout en offrant la profondeur
hiérarchique qui manquait."""

import numpy as np

from ddlab_bench.eca import buildSuccessorArray


class MarkovChain:
    """Chaîne de Markov : matrice de transition stochastique en ligne."""

    def __init__(self, transition):
        P = np.asarray(transition, dtype=np.float64)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("matrice de transition non carrée")
        if np.any(P < 0):
            raise ValueError("probabilités de transition négatives")
        self.size = P.shape[0]
        self.P = P / P.sum(axis=1, keepdims=True)  # normalisation en ligne

    def multiStep(self, k):
        """Matrice de transition à k pas P^k (exacte)."""
        return np.linalg.matrix_power(self.P, k)

    def stationary(self):
        """Distribution stationnaire — vecteur propre gauche de valeur propre 1."""
        values, vectors = np.linalg.eig(self.P.T)
        idx = int(np.argmin(np.abs(values - 1.0)))
        pi = np.abs(np.real(vectors[:, idx]))
        return pi / pi.sum()

    def transferSpectrum(self):
        """Spectre de l'opérateur de transfert, trié par module décroissant.

        Retourne (eigenvalues, eigenvectors). Les valeurs propres de module
        proche de 1 sont les modes LENTS — les coordonnées métastables ; les
        écarts spectraux marquent les frontières d'échelle de la hiérarchie."""
        values, vectors = np.linalg.eig(self.P)
        order = np.argsort(-np.abs(values))
        return values[order], vectors[:, order]

    def impliedTimescales(self):
        """Temps de relaxation t = -1 / ln|lambda| de chaque mode. Les sauts de
        cette suite sont les niveaux de la hiérarchie métastable."""
        values, _ = self.transferSpectrum()
        mags = np.clip(np.abs(values), 1e-12, 1.0 - 1e-12)
        return -1.0 / np.log(mags)


def hierarchicalChain(branching, depth, leafSize, ratesByLevel, noise=0.01, seed=0):
    """Chaîne de Markov à hiérarchie métastable PLANTÉE — vérité-terrain pour la
    hiérarchie elle-même.

    Arbre de blocs : `branching` enfants par nœud, `depth` niveaux, `leafSize`
    micro-états par feuille (size = branching^depth * leafSize). Deux états
    communiquent avec un taux qui décroît avec la profondeur de leur ancêtre
    commun — mélange rapide intra-feuille, lent entre branches profondes.
    `ratesByLevel` a depth+1 entrées : ratesByLevel[c] pour une paire dont
    l'ancêtre commun est à c niveaux des feuilles (0 = même feuille).

    Retourne (chain, levels) : levels[d] = partition plantée au niveau d,
    levels[0] = feuilles, levels[depth-1] = blocs sous la racine."""
    if len(ratesByLevel) != depth + 1:
        raise ValueError(f"ratesByLevel doit avoir depth+1={depth + 1} entrées")
    rng = np.random.default_rng(seed)
    nLeaves = branching ** depth
    size = nLeaves * leafSize
    leafOf = np.repeat(np.arange(nLeaves), leafSize)
    # chemin base-`branching` (depth chiffres) de chaque feuille
    paths = np.array([
        [(leaf // branching ** (depth - 1 - d)) % branching for d in range(depth)]
        for leaf in range(nLeaves)
    ])
    rates = np.asarray(ratesByLevel, dtype=np.float64)
    P = np.empty((size, size))
    for x in range(size):
        px = paths[leafOf[x]]
        for y in range(size):
            py = paths[leafOf[y]]
            shared = 0
            while shared < depth and px[shared] == py[shared]:
                shared += 1
            P[x, y] = rates[depth - shared]
    P *= 1.0 + noise * rng.random((size, size))  # casse les ex-aequo
    chain = MarkovChain(P)
    levels = []
    for d in range(depth):
        keys = [tuple(paths[leafOf[x]][: depth - d]) for x in range(size)]
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        levels.append(np.array([uniq[k] for k in keys], dtype=np.int64))
    return chain, levels


def noisyEcaChain(rule, n, epsilon):
    """Chaîne de Markov d'un ECA bruité : avec proba (1-epsilon) la règle
    déterministe, avec proba epsilon un voisin perceptuel (état à un bit).

    Les bassins déterministes deviennent des ensembles MÉTASTABLES de la chaîne ;
    epsilon -> 0 retrouve l'automate du §1-§15. Pont entre le banc d'essai
    déterministe et le substrat stochastique."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon={epsilon} hors borne [0,1]")
    succ = buildSuccessorArray(rule, n)
    size = 1 << n
    P = np.zeros((size, size))
    for s in range(size):
        P[s, int(succ[s])] += 1.0 - epsilon
        for bit in range(n):
            P[s, s ^ (1 << bit)] += epsilon / n
    return MarkovChain(P)
