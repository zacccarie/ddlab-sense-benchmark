"""Tests Phase 2 — roundtrip des artefacts et conversions d'état."""

import numpy as np

from ddlab_bench import computeBasinField
from ddlab_bench.artifacts import buildBundle, loadBundle
from ddlab_bench.models.sequence_model import bitsToState, stateToBits


def testBitConversionRoundtrip():
    states = np.array([0, 1, 42, 8191, 16383])
    assert np.array_equal(bitsToState(stateToBits(states, 14)), states)


def testBundleRoundtrip(tmp_path):
    field = computeBasinField(110, 12)
    rng = np.random.default_rng(1)
    config = {"count": 100, "length": 16, "holdoutFraction": 0.3}
    path = buildBundle(field, config, rng, tmp_path / "rule110_n12.npz")
    reloaded, data = loadBundle(path)

    assert reloaded.ruleNumber == 110
    assert reloaded.n == 12
    assert np.array_equal(reloaded.succ, field.succ)
    assert np.array_equal(reloaded.attractorId, field.attractorId)
    assert len(reloaded.attractors) == len(field.attractors)
    assert sum(len(pre) for pre in reloaded.preimages) == (1 << 12)
    assert data["trainTraj"].shape == (100, 16)
    assert data["testTraj"].shape == (100, 16)


def testOodTrajectoriesStayInHeldoutBasins(tmp_path):
    # les trajectoires OOD doivent partir de bassins retenus
    field = computeBasinField(110, 12)
    rng = np.random.default_rng(2)
    config = {"count": 200, "length": 8, "holdoutFraction": 0.3}
    path = buildBundle(field, config, rng, tmp_path / "bundle.npz")
    reloaded, data = loadBundle(path)

    testBasinSet = set(data["testBasins"].tolist())
    for startState in data["testTraj"][:, 0]:
        assert int(reloaded.attractorId[startState]) in testBasinSet


def testTrainAndOodStatesDisjoint(tmp_path):
    field = computeBasinField(54, 12)
    rng = np.random.default_rng(3)
    config = {"count": 50, "length": 8, "holdoutFraction": 0.3}
    path = buildBundle(field, config, rng, tmp_path / "bundle.npz")
    _, data = loadBundle(path)
    assert set(data["trainStates"].tolist()).isdisjoint(data["testStates"].tolist())
