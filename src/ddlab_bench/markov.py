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


def _hierarchyLayout(branching, depth, leafSize):
    """Disposition d'un arbre de communautés : niveau de l'ancêtre commun de
    chaque paire d'états, et partitions plantées par niveau."""
    nLeaves = branching ** depth
    size = nLeaves * leafSize
    leafOf = np.repeat(np.arange(nLeaves), leafSize)
    paths = np.array([
        [(leaf // branching ** (depth - 1 - d)) % branching for d in range(depth)]
        for leaf in range(nLeaves)
    ])
    lcaLevel = np.empty((size, size), dtype=np.int64)
    for x in range(size):
        px = paths[leafOf[x]]
        for y in range(size):
            py = paths[leafOf[y]]
            shared = 0
            while shared < depth and px[shared] == py[shared]:
                shared += 1
            lcaLevel[x, y] = depth - shared
    levels = []
    for d in range(depth):
        keys = [tuple(paths[leafOf[x]][: depth - d]) for x in range(size)]
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        levels.append(np.array([uniq[k] for k in keys], dtype=np.int64))
    return lcaLevel, levels


def hierarchicalGraphChain(branching, depth, leafSize, edgeProbsByLevel, seed=0):
    """Marche aléatoire sur un graphe modulaire hiérarchique ÉCHANTILLONNÉ.

    Graphe à communautés emboîtées : chaque paire de nœuds reçoit une arête avec
    une probabilité décroissant avec la profondeur de leur ancêtre commun (dense
    intra-feuille, rare entre branches profondes). La chaîne est la marche
    aléatoire sur ce graphe — P = adjacence normalisée en ligne. Contrairement à
    `hierarchicalChain` (taux POSÉS directement), ici la topologie est
    échantillonnée et la métastabilité ÉMERGE de la dynamique de marche.
    `edgeProbsByLevel` a depth+1 entrées (0 = même feuille). Des auto-boucles
    garantissent qu'aucun nœud n'est isolé.

    Retourne (chain, levels) — levels[d] = communauté plantée au niveau d."""
    if len(edgeProbsByLevel) != depth + 1:
        raise ValueError(f"edgeProbsByLevel doit avoir depth+1={depth + 1} entrées")
    rng = np.random.default_rng(seed)
    lcaLevel, levels = _hierarchyLayout(branching, depth, leafSize)
    size = lcaLevel.shape[0]
    probs = np.asarray(edgeProbsByLevel, dtype=np.float64)[lcaLevel]
    upper = np.triu(rng.random((size, size)) < probs, k=1)
    adjacency = (upper | upper.T).astype(np.float64)
    np.fill_diagonal(adjacency, 1.0)  # auto-boucles -> aucun nœud isolé
    return MarkovChain(adjacency), levels


def spinGlassChain(n, temperature, seed=0):
    """Chaîne de Markov d'un verre de spin — dynamique de Glauber sur un Ising
    à couplages aléatoires.

    n spins +-1 ; énergie E(s) = -sum_{i<j} J_ij s_i s_j, couplages J_ij
    gaussiens i.i.d. ; transition = retournement d'un spin, accepté selon la
    règle de Metropolis à `temperature`. À basse température la chaîne développe
    des états métastables — les minima d'énergie — dont la hiérarchie ÉMERGE des
    seuls couplages aléatoires : RIEN n'est planté (contraste §16/§18).
    Énumérable (2^n états) -> vérité-terrain exacte ; la mesure stationnaire est
    exactement la distribution de Boltzmann (bilan détaillé).

    Retourne (chain, energy) — energy[s] = énergie de l'état s."""
    if temperature <= 0:
        raise ValueError(f"temperature={temperature} doit être > 0")
    rng = np.random.default_rng(seed)
    size = 1 << n
    upper = np.triu(rng.standard_normal((n, n)), 1)
    J = upper + upper.T  # couplage symétrique, diagonale nulle
    spins = np.where(
        ((np.arange(size)[:, None] >> np.arange(n)) & 1) == 1, 1.0, -1.0
    )  # (size, n) spins +-1
    energy = -0.5 * np.einsum("si,ij,sj->s", spins, J, spins)
    P = np.zeros((size, size))
    for s in range(size):
        sig = spins[s]
        field = J @ sig  # champ local h_i
        for i in range(n):
            deltaE = 2.0 * sig[i] * field[i]
            accept = 1.0 if deltaE <= 0.0 else np.exp(-deltaE / temperature)
            P[s, s ^ (1 << i)] = accept / n
        P[s, s] = 1.0 - P[s].sum()
    return MarkovChain(P), energy
