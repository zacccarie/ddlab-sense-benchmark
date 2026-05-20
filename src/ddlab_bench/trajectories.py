"""Phase 1: génération de jeux de données et découpage out-of-distribution.

Le split OOD retient des bassins ENTIERS pour le test — il mesure ainsi la
généralisation de structure (le modèle voit-il une dynamique jamais observée ?),
pas la simple interpolation."""

import numpy as np


def generateTrajectories(field, count, length, rng, startStates=None):
    """Génère `count` trajectoires de longueur `length`.

    Les états initiaux sont tirés dans `startStates` si fourni, sinon dans tout
    l'espace d'états. Retourne un array (count, length) d'entiers-états."""
    if startStates is None:
        starts = rng.integers(0, 1 << field.n, size=count)
    else:
        startStates = np.asarray(startStates)
        starts = startStates[rng.integers(0, len(startStates), size=count)]
    traj = np.empty((count, length), dtype=np.int64)
    for i in range(count):
        node = int(starts[i])
        for t in range(length):
            traj[i, t] = node
            node = int(field.succ[node])
    return traj


def splitByBasins(field, holdoutFraction, rng):
    """Sépare des bassins entiers en train / test OOD.

    Retourne un dict avec oodPossible, les ensembles de bassins et les états de
    chaque partition. oodPossible est False si la règle n'a qu'un seul bassin
    (structure triviale: aucun bassin entier ne peut être retenu)."""
    size = 1 << field.n
    attractorCount = len(field.attractors)
    if attractorCount < 2:
        return {
            "oodPossible": False,
            "trainBasins": set(),
            "testBasins": set(),
            "trainStates": np.arange(size),
            "testStates": np.array([], dtype=np.int64),
        }

    order = rng.permutation(attractorCount)
    holdoutCount = max(1, min(attractorCount - 1, round(attractorCount * holdoutFraction)))
    testBasins = {int(x) for x in order[:holdoutCount]}
    trainBasins = {int(x) for x in order[holdoutCount:]}
    testMask = np.isin(field.attractorId, list(testBasins))
    return {
        "oodPossible": True,
        "trainBasins": trainBasins,
        "testBasins": testBasins,
        "trainStates": np.where(~testMask)[0],
        "testStates": np.where(testMask)[0],
    }
