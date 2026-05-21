"""Tests Phase 3 — métriques sur le graphe induit et homomorphisme de bassins."""

import numpy as np

from ddlab_bench import computeBasinField
from ddlab_bench.metrics import (
    balancedAmi,
    basinHomomorphism,
    forwardEdgeAccuracy,
    goeDetectionMetrics,
    preimageSetMetrics,
)


def testPerfectInducedGraphScoresPerfect():
    # graphe appris == vérité-terrain -> toutes les métriques parfaites
    field = computeBasinField(110, 10)
    induced = field.succ.copy()
    assert forwardEdgeAccuracy(induced, field.succ) == 1.0
    pre = preimageSetMetrics(induced, field.succ)
    assert pre["preimageExactMatch"] == 1.0
    assert pre["preimageJaccardMean"] == 1.0
    assert goeDetectionMetrics(induced, field.succ)["goeF1"] == 1.0


def testGoeCountMatchesGroundTruth():
    field = computeBasinField(30, 10)
    goe = goeDetectionMetrics(field.succ.copy(), field.succ)
    assert goe["trueGoeCount"] == len(field.goe)
    assert goe["inducedGoeCount"] == len(field.goe)


def testSingleWrongEdgeDegradesStructure():
    field = computeBasinField(110, 10)
    induced = field.succ.copy()
    # détourne une arête vers une cible différente
    induced[0] = (int(induced[0]) + 1) % len(induced)
    size = len(induced)
    assert forwardEdgeAccuracy(induced, field.succ) == (size - 1) / size
    # une arête détournée corrompt exactement deux ensembles de pré-images
    pre = preimageSetMetrics(induced, field.succ)
    assert pre["preimageExactMatch"] == (size - 2) / size


def testBasinHomomorphismPerfectOnOneHotLabels():
    # features = one-hot du vrai bassin -> k-means retrouve la partition (AMI ~ 1)
    field = computeBasinField(110, 9)
    k = len(field.attractors)
    oneHot = np.eye(k, dtype=np.float32)[field.attractorId]
    assert basinHomomorphism(oneHot, field.attractorId, seed=0)["ami"] > 0.99


def testBasinHomomorphismLowOnRandomFeatures():
    # features aléatoires -> aucun lien avec les bassins -> AMI proche de 0
    field = computeBasinField(110, 9)
    randomFeats = np.random.default_rng(0).normal(size=(1 << 9, 16)).astype(np.float32)
    assert basinHomomorphism(randomFeats, field.attractorId, seed=0)["ami"] < 0.3


def testBalancedAmiPerfectClustering():
    # clustering parfait -> AMI rééquilibrée ~ 1 malgré le fort déséquilibre
    reference = np.array([0] * 900 + [1] * 50 + [2] * 50)
    assert balancedAmi(reference, reference.copy(), perClass=50, seed=0) > 0.99


def testBalancedAmiPenalizesIgnoredSmallClasses():
    # clustering qui ne récupère que la classe géante (petites classes fusionnées) :
    # l'AMI standard reste élevée — le géant domine —, la rééquilibrée chute.
    from sklearn.metrics import adjusted_mutual_info_score

    reference = np.array([0] * 900 + [1] * 50 + [2] * 50)
    clusters = np.array([0] * 900 + [1] * 100)
    standard = adjusted_mutual_info_score(reference, clusters)
    balanced = balancedAmi(reference, clusters, perClass=50, seed=0)
    assert balanced < standard
