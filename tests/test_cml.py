"""Tests — Coupled Map Lattice discrétisé (substrat continu rendu énumérable)."""

import numpy as np
import pytest

from ddlab_bench import basinFieldFromSuccessor, computeBasinField
from ddlab_bench.classify import maryBasinMismatch, perceptualBasinMismatch
from ddlab_bench.cml import CoupledMapLattice, logisticMap


def testLogisticStaysInUnitInterval():
    x = np.linspace(0.0, 1.0, 50)
    for r in (2.0, 3.5, 4.0):
        y = logisticMap(x, r)
        assert y.min() >= 0.0
        assert y.max() <= 1.0 + 1e-12


def testLogisticPeakAtHalf():
    assert logisticMap(0.5, 4.0) == pytest.approx(1.0)


def testDecodeEncodeRoundTrip():
    cml = CoupledMapLattice(n=6, m=4, r=3.8, epsilon=0.2)
    for state in (0, 1, 137, 4095):
        assert cml.encode(cml.decode(state)) == state


def testSuccessorArrayShapeAndRange():
    cml = CoupledMapLattice(n=6, m=4, r=3.8, epsilon=0.2)
    succ = cml.successorArray()
    size = 4 ** 6
    assert len(succ) == size
    assert succ.min() >= 0
    assert succ.max() < size


def testStepMatchesSuccessorArray():
    cml = CoupledMapLattice(n=5, m=5, r=3.9, epsilon=0.3)
    succ = cml.successorArray()
    for state in (0, 1, 42, 1000, 5 ** 5 - 1):
        assert cml.stepState(state) == succ[state]


def testQuiescentAndSaturatedMapToZero():
    # tout-à-zéro : f(0)=0 -> point fixe ; tout-au-max : f(1)=0 -> draine vers 0
    cml = CoupledMapLattice(n=5, m=4, r=4.0, epsilon=0.4)
    succ = cml.successorArray()
    assert succ[0] == 0
    assert succ[4 ** 5 - 1] == 0


def testUncoupledLatticeIsCellIndependent():
    # eps=0 -> chaque cellule évolue indépendamment ; modifier une cellule ne
    # change que cette cellule dans le successeur
    cml = CoupledMapLattice(n=5, m=4, r=3.8, epsilon=0.0)
    succ = cml.successorArray()
    base = 137
    for cell in range(cml.n):
        level = (base // cml.m ** cell) % cml.m
        if level + 1 >= cml.m:
            continue
        neighbor = base + cml.m ** cell
        dBase = cml.decode(int(succ[base]))
        dNeighbor = cml.decode(int(succ[neighbor]))
        for other in range(cml.n):
            if other != cell:
                assert dBase[other] == dNeighbor[other]


def testCouplingChangesDynamics():
    a = CoupledMapLattice(n=5, m=4, r=3.8, epsilon=0.0).successorArray()
    b = CoupledMapLattice(n=5, m=4, r=3.8, epsilon=0.3).successorArray()
    assert not np.array_equal(a, b)


def testBasinFieldIsFunctionalGraph():
    cml = CoupledMapLattice(n=6, m=4, r=3.9, epsilon=0.25)
    field = basinFieldFromSuccessor(cml.successorArray(), 6, "cml-test")
    size = 4 ** 6
    assert np.all(field.attractorId >= 0)
    assert sum(len(pre) for pre in field.preimages) == size
    for cycle in field.attractors:
        cycleSet = set(cycle)
        for state in cycle:
            assert int(field.succ[state]) in cycleSet


def testMaryBasinMismatchGeneralizesBinary():
    # sur un automate binaire, le mismatch m-aire (m=2) doit coïncider avec le
    # mismatch perceptuel binaire — un décalage de ±1 niveau = une inversion de bit
    field = computeBasinField(110, 9)
    assert maryBasinMismatch(field, 2) == pytest.approx(perceptualBasinMismatch(field))


def testEnumerationGuardRejectsOversizedLattice():
    with pytest.raises(ValueError):
        CoupledMapLattice(n=30, m=8, r=3.8, epsilon=0.2)
