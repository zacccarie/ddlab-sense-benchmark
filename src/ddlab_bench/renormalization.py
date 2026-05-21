"""Renormalisation dynamique — quotient de bisimulation et complexité statistique.

Le §9 de la note de recherche montre qu'aucun objectif perceptuel ne récupère
le quotient dynamique. Ce module construit ce quotient *exactement*, par
renormalisation du graphe de transition (et non de l'espace d'observation).

L'opérateur de raffinement R fusionne deux états ssi ils partagent le même
observable de base ET des successeurs déjà fusionnés. Itéré jusqu'au point
fixe, il donne le quotient de bisimulation relatif à l'observable : la
partition la plus grossière qui raffine l'observable et soit compatible avec
la dynamique. C'est la formulation coinductive du §11.

Sans observable de base plus grossier que l'identité, le quotient d'une
dynamique déterministe est trivial — chaque état est son propre futur. C'est
l'observable `label` qui définit le « comportement macroscopique »."""

import numpy as np


def refinePartition(succ, partition):
    """Un pas de l'opérateur de renormalisation R.

    Deux états restent fusionnés ssi ils sont déjà dans le même bloc ET leurs
    successeurs sont dans le même bloc — la partition est raffinée par la
    pré-image de la dynamique. Retourne une partition réétiquetée 0..k-1."""
    partition = np.asarray(partition, dtype=np.int64)
    succ = np.asarray(succ, dtype=np.int64)
    span = int(partition.max()) + 1
    # clé par état : (bloc courant, bloc du successeur)
    keys = partition * span + partition[succ]
    _, refined = np.unique(keys, return_inverse=True)
    return refined.astype(np.int64)


def bisimulationQuotient(succ, label):
    """Quotient de bisimulation relatif à l'observable de base `label`.

    Itère R depuis la partition induite par `label` jusqu'au point fixe : la
    partition la plus grossière qui (i) raffine `label` et (ii) soit compatible
    avec la dynamique — x ≡ x' implique succ(x) ≡ succ(x').

    Retourne `partition` : partition[s] = index de bloc de l'état s, 0..k-1.
    R ne fait que scinder des blocs, jamais en fusionner — la convergence est
    donc garantie en au plus |S| itérations."""
    succ = np.asarray(succ, dtype=np.int64)
    _, partition = np.unique(np.asarray(label), return_inverse=True)
    partition = partition.astype(np.int64)
    while True:
        refined = refinePartition(succ, partition)
        if int(refined.max()) == int(partition.max()):
            # aucun bloc scindé -> point fixe
            return partition
        partition = refined


def quotientGraph(succ, partition):
    """Graphe de transition quotient : successeur induit sur les blocs.

    Bien défini seulement si `partition` est compatible avec la dynamique — ce
    que garantit `bisimulationQuotient`. quotientSucc[b] = bloc du successeur
    de tout état du bloc b. Le quotient est lui-même un système dynamique
    discret, analysable par `basinFieldFromSuccessor`.

    Retourne (quotientSucc, representative), representative[b] = un état du
    bloc b."""
    partition = np.asarray(partition, dtype=np.int64)
    succ = np.asarray(succ, dtype=np.int64)
    nBlocks = int(partition.max()) + 1
    quotientSucc = np.full(nBlocks, -1, dtype=np.int64)
    representative = np.full(nBlocks, -1, dtype=np.int64)
    for state in range(len(partition)):
        block = int(partition[state])
        if representative[block] == -1:
            representative[block] = state
            quotientSucc[block] = partition[succ[state]]
    return quotientSucc, representative


def statisticalComplexity(partition, weights=None):
    """Complexité statistique C_mu : entropie de Shannon (bits) de la
    distribution de masse des blocs du quotient.

    weights=None -> mesure uniforme sur l'espace d'états (entropie des tailles
                    de blocs).
    weights      -> mesure quelconque par état (ex. masse de bassin, mesure
                    stationnaire) ; chaque bloc pèse alors la masse de ses états.

    C_mu n'est pas le taux d'entropie : elle mesure la richesse structurelle du
    quotient, pas le désordre du signal (§12). Ordre -> C_mu basse, chaos ->
    C_mu basse, bord du chaos -> C_mu maximale."""
    partition = np.asarray(partition, dtype=np.int64)
    nBlocks = int(partition.max()) + 1
    if weights is None:
        mass = np.bincount(partition, minlength=nBlocks).astype(np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        mass = np.bincount(partition, weights=weights, minlength=nBlocks)
    total = mass.sum()
    if total <= 0:
        return 0.0
    p = mass / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def densityLabel(n):
    """Observable de base perceptuellement grossier : densité (nombre de
    cellules à 1) de chaque état de [0, 2^n).

    Projection π non invariante par la dynamique — l'observable de référence du
    §11 : le quotient de bisimulation relatif à `densityLabel` regroupe les
    états indistinguables par toute observation de densité future."""
    return np.array([bin(state).count("1") for state in range(1 << n)], dtype=np.int64)


def windowLabel(n, width):
    """Observable de base : les `width` bits de poids faible de chaque état —
    une fenêtre d'observation partielle de l'automate."""
    if not 1 <= width <= n:
        raise ValueError(f"width={width} hors borne [1, {n}]")
    mask = (1 << width) - 1
    return np.array([state & mask for state in range(1 << n)], dtype=np.int64)


def bisimulationMetric(succ, label, gamma=0.9, baseCost=1.0, iters=200, tol=1e-12):
    """Métrique de bisimulation exacte — point fixe de

        d(x, x') = baseCost * [label(x) != label(x')] + gamma * d(f(x), f(x'))

    Calculée par itération sur la matrice |S| x |S| (gamma < 1 -> contraction,
    point fixe unique). C'est la version *quantitative* du quotient : son
    ensemble de niveau zéro est exactement le quotient de bisimulation. Sert de
    cible exacte — ce que l'objectif appris du §11 cherche à embarquer.

    Coût mémoire |S|^2 : réservé aux petits systèmes énumérables."""
    succ = np.asarray(succ, dtype=np.int64)
    label = np.asarray(label)
    base = baseCost * (label[:, None] != label[None, :]).astype(np.float64)
    d = base.copy()
    idx = np.ix_(succ, succ)  # d[idx][x, x'] = d[f(x), f(x')]
    for _ in range(iters):
        updated = base + gamma * d[idx]
        if np.max(np.abs(updated - d)) < tol:
            return updated
        d = updated
    return d
