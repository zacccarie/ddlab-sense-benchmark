"""Classification des règles ECA.

Deux mesures complémentaires:
  - zParameter: Z de Wuensche, a priori, calculé depuis la table de règle.
  - structuralSummary: mesures exactes a posteriori, extraites du champ de bassins."""

import numpy as np

from ddlab_bench.eca import ruleTable


def zParameter(ruleNumber):
    """Z-parameter de Wuensche, premier ordre (déterminisme du voisin immédiat).

    Lors de la reconstruction d'une pré-image, zRight (resp. zLeft) est la fraction
    de paires de cellules connues qui déterminent de façon unique la troisième.
    Z = max(zLeft, zRight). Référence de validation: règle 110 -> zRight 0.75."""
    table = ruleTable(ruleNumber)

    # reconstruction gauche -> droite: (gauche, centre) connus, droite à déterminer
    rightDeterministic = 0
    for left in (0, 1):
        for center in (0, 1):
            base = (left << 2) | (center << 1)
            if table[base | 0] != table[base | 1]:
                rightDeterministic += 1

    # reconstruction droite -> gauche: (centre, droite) connus, gauche à déterminer
    leftDeterministic = 0
    for center in (0, 1):
        for right in (0, 1):
            idx = (center << 1) | right
            if table[idx] != table[(1 << 2) | idx]:
                leftDeterministic += 1

    zRight = rightDeterministic / 4.0
    zLeft = leftDeterministic / 4.0
    return {"zLeft": zLeft, "zRight": zRight, "z": max(zLeft, zRight)}


def structuralSummary(field):
    """Mesures structurelles exactes extraites du champ de bassins (vérité-terrain)."""
    size = 1 << field.n
    periods = [len(cycle) for cycle in field.attractors]
    basinCounts = np.bincount(field.attractorId, minlength=len(field.attractors))
    inDegrees = np.fromiter(
        (len(pre) for pre in field.preimages), dtype=np.int64, count=size
    )
    return {
        "attractorCount": len(field.attractors),
        "maxAttractorPeriod": max(periods),
        "meanAttractorPeriod": float(np.mean(periods)),
        "maxTransient": int(field.transient.max()),
        "meanTransient": float(field.transient.mean()),
        "goeFraction": len(field.goe) / size,
        "maxBasinFraction": float(basinCounts.max()) / size,
        "inDegreeMax": int(inDegrees.max()),
    }


def regimeHint(zValue):
    """Indication grossière de régime à partir du Z-parameter (pour la sélection de règles)."""
    if zValue < 0.5:
        return "ordonné"
    if zValue < 0.85:
        return "complexe"
    return "chaotique"


def perceptualBasinMismatch(field):
    """Fraction des voisins perceptuels (états à 1 bit) qui changent de bassin.

    Élevé => les bassins sont perceptuellement éclatés : la proximité
    d'observation ne prédit pas le destin. À comparer à `randomBasinMismatch`."""
    size = 1 << field.n
    basin = field.attractorId
    different = 0
    for state in range(size):
        for bit in range(field.n):
            if basin[state ^ (1 << bit)] != basin[state]:
                different += 1
    return different / (size * field.n)


def randomBasinMismatch(field):
    """Mismatch attendu si les bassins étaient répartis au hasard, à tailles
    identiques : 1 - somme des p_b^2. Référence pour `perceptualBasinMismatch`."""
    sizes = np.bincount(field.attractorId)
    fractions = sizes / sizes.sum()
    return 1.0 - float((fractions ** 2).sum())
