"""Tests Phase 3 — sondes linéaire et MLP, étiquettes causales."""

import numpy as np

from ddlab_bench import computeBasinField
from ddlab_bench.probes import MLPProbe, goeLabels, inDegreeLabels, trainProbe


def testGoeLabelsMatchField():
    field = computeBasinField(30, 10)
    labels = goeLabels(field)
    assert int(labels.sum()) == len(field.goe)
    for state in field.goe:
        assert labels[state] == 1


def testInDegreeLabelsClippedToThreeClasses():
    field = computeBasinField(110, 10)
    labels = inDegreeLabels(field)
    assert labels.min() >= 0
    assert labels.max() <= 2
    # un état garden-of-Eden a pour étiquette d'in-degré 0
    for state in field.goe[:20]:
        assert labels[state] == 0


def _separableTask(rng):
    """Tâche 3 classes linéairement séparable (features one-hot bruitées)."""
    count = 2000
    labels = rng.integers(0, 3, size=count)
    features = np.zeros((count, 3), dtype=np.float32)
    features[np.arange(count), labels] = 1.0
    features += rng.normal(0, 0.05, size=features.shape).astype(np.float32)
    perm = rng.permutation(count)
    return features, labels, perm[:1400], perm[1400:]


def testLinearProbeRecoversSeparableSignal():
    rng = np.random.default_rng(0)
    features, labels, trainIdx, testIdx = _separableTask(rng)
    result = trainProbe(features, labels, trainIdx, testIdx, 3, rng)
    assert result["testAccuracy"] > 0.95


def testMlpProbeRecoversSeparableSignal():
    rng = np.random.default_rng(0)
    features, labels, trainIdx, testIdx = _separableTask(rng)
    result = trainProbe(features, labels, trainIdx, testIdx, 3, rng, probeFactory=MLPProbe)
    assert result["testAccuracy"] > 0.95
