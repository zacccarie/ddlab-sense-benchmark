"""Analyse du graphe de transition (graphe fonctionnel) d'un automate cellulaire.

Le graphe de transition d'un CA déterministe est fonctionnel: chaque état a
exactement un successeur. On en extrait la vérité-terrain de l'expérience:
attracteurs (cycles), bassins, pré-images, garden-of-Eden, longueurs de transitoire."""

from dataclasses import dataclass

import numpy as np

from ddlab_bench.eca import buildSuccessorArray


def buildPreimageArray(succ):
    """preimages[s] = liste des prédécesseurs de s (inversion du graphe de transition)."""
    preimages = [[] for _ in range(len(succ))]
    for state, nextState in enumerate(succ):
        preimages[int(nextState)].append(state)
    return preimages


def gardenOfEden(preimages):
    """États sans prédécesseur — inatteignables par la dynamique forward."""
    return [state for state, pre in enumerate(preimages) if not pre]


def findAttractors(succ):
    """Détecte les cycles attracteurs et l'appartenance aux bassins.

    Retourne (attractorId, attractors):
      - attractorId[s] = index du bassin (attracteur) auquel s draine
      - attractors = liste de cycles, chaque cycle étant une liste d'états."""
    size = len(succ)
    UNVISITED, INPROGRESS, DONE = 0, 1, 2
    color = np.zeros(size, dtype=np.uint8)
    attractorId = np.full(size, -1, dtype=np.int64)
    attractors = []

    for start in range(size):
        if color[start] != UNVISITED:
            continue
        # marche avant jusqu'à un état déjà coloré
        path = []
        node = start
        while color[node] == UNVISITED:
            color[node] = INPROGRESS
            path.append(node)
            node = int(succ[node])

        if color[node] == INPROGRESS:
            # nouveau cycle: de node jusqu'à son retour dans le chemin courant
            cycleStart = path.index(node)
            cycle = path[cycleStart:]
            attractorIndex = len(attractors)
            attractors.append(cycle)
            for state in cycle:
                attractorId[state] = attractorIndex
        else:
            # le chemin rejoint un bassin déjà résolu
            attractorIndex = int(attractorId[node])

        for state in path:
            if attractorId[state] == -1:
                attractorId[state] = attractorIndex
            color[state] = DONE

    return attractorId, attractors


def transientLengths(succ, attractors):
    """Longueur du transitoire de chaque état: distance à son cycle attracteur (0 sur le cycle)."""
    size = len(succ)
    transient = np.full(size, -1, dtype=np.int64)
    for cycle in attractors:
        for state in cycle:
            transient[state] = 0

    for start in range(size):
        if transient[start] != -1:
            continue
        # remonte la chaîne forward jusqu'à une valeur connue, puis redistribue
        path = []
        node = start
        while transient[node] == -1:
            path.append(node)
            node = int(succ[node])
        base = int(transient[node])
        for offset, state in enumerate(reversed(path)):
            transient[state] = base + offset + 1
    return transient


@dataclass
class BasinField:
    """Champ de bassins complet d'une règle ECA — la vérité-terrain de l'expérience."""

    ruleNumber: int
    n: int
    succ: np.ndarray
    preimages: list
    attractorId: np.ndarray
    attractors: list
    transient: np.ndarray
    goe: list


def computeBasinField(ruleNumber, n):
    """Champ de bassins complet d'une règle ECA, par énumération exhaustive (2^n états)."""
    succ = buildSuccessorArray(ruleNumber, n)
    preimages = buildPreimageArray(succ)
    attractorId, attractors = findAttractors(succ)
    transient = transientLengths(succ, attractors)
    goe = gardenOfEden(preimages)
    return BasinField(ruleNumber, n, succ, preimages, attractorId, attractors, transient, goe)
