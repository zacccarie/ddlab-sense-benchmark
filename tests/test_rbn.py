"""Tests Phase 6 — réseaux booléens aléatoires."""

import numpy as np

from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.rbn import RandomBooleanNetwork


def testRbnReproducibleWithSeed():
    a = RandomBooleanNetwork(10, 2, seed=0).successorArray()
    b = RandomBooleanNetwork(10, 2, seed=0).successorArray()
    assert np.array_equal(a, b)


def testRbnSeedsDiffer():
    a = RandomBooleanNetwork(10, 2, seed=0).successorArray()
    b = RandomBooleanNetwork(10, 2, seed=1).successorArray()
    assert not np.array_equal(a, b)


def testRbnSuccessorArrayInRange():
    succ = RandomBooleanNetwork(10, 3, seed=0).successorArray()
    assert len(succ) == (1 << 10)
    assert succ.min() >= 0
    assert succ.max() < (1 << 10)


def testRbnStepMatchesSuccessorArray():
    rbn = RandomBooleanNetwork(9, 2, seed=2)
    succ = rbn.successorArray()
    for state in (0, 1, 100, 383, 511):
        assert rbn.stepState(state) == succ[state]


def testRbnBasinFieldIsFunctionalGraph():
    rbn = RandomBooleanNetwork(10, 2, seed=0)
    field = basinFieldFromSuccessor(rbn.successorArray(), 10, rbn.seed)
    assert np.all(field.attractorId >= 0)
    assert sum(len(pre) for pre in field.preimages) == (1 << 10)
    for cycle in field.attractors:
        cycleSet = set(cycle)
        for state in cycle:
            assert int(field.succ[state]) in cycleSet
