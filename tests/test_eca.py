"""Tests de sanité Phase 0 — propriétés exactes du graphe de transition."""

import numpy as np
import pytest

from ddlab_bench import buildSuccessorArray, computeBasinField, ruleTable, stepState, zParameter


def testRuleTableDecoding():
    # règle 110 = 0b01101110
    assert list(ruleTable(110)) == [0, 1, 1, 1, 0, 1, 1, 0]


def testRuleTableBounds():
    with pytest.raises(ValueError):
        ruleTable(256)


def testStepStateRule110():
    # état 0b010 sous la règle 110, bord périodique n=3, donne 0b011
    assert stepState(0b010, ruleTable(110), 3) == 0b011


def testIdentityRule204():
    # règle 204: la sortie vaut la cellule centrale -> succ[s] == s
    succ = buildSuccessorArray(204, 8)
    assert np.array_equal(succ, np.arange(len(succ)))


def testNullRule0():
    # règle 0: tout converge vers l'état 0
    succ = buildSuccessorArray(0, 8)
    assert np.all(succ == 0)


def testFunctionalGraphPreimageCount():
    # graphe fonctionnel: la somme des in-degrés vaut le nombre d'états
    field = computeBasinField(110, 10)
    assert sum(len(pre) for pre in field.preimages) == (1 << 10)


def testEveryStateBelongsToBasin():
    field = computeBasinField(30, 10)
    assert np.all(field.attractorId >= 0)


def testAttractorStatesHaveZeroTransient():
    field = computeBasinField(110, 10)
    for cycle in field.attractors:
        for state in cycle:
            assert field.transient[state] == 0


def testAttractorIsClosedCycle():
    field = computeBasinField(110, 10)
    for cycle in field.attractors:
        cycleSet = set(cycle)
        for state in cycle:
            assert int(field.succ[state]) in cycleSet


def testGardenOfEdenHasNoPreimage():
    field = computeBasinField(30, 10)
    for state in field.goe:
        assert len(field.preimages[state]) == 0


def testTransientReachesAttractor():
    # appliquer succ `transient[s]` fois mène à un état de cycle (transitoire 0)
    field = computeBasinField(110, 10)
    onCycle = {state for cycle in field.attractors for state in cycle}
    for state in (0, 17, 511, 1023):
        node = state
        for _ in range(int(field.transient[state])):
            node = int(field.succ[node])
        assert node in onCycle


def testZParameterRule110():
    # valeur de référence de la littérature: règle 110 -> zLeft 0.25, zRight 0.75
    z = zParameter(110)
    assert z["zLeft"] == 0.25
    assert z["zRight"] == 0.75
    assert z["z"] == 0.75


def testZParameterRule0():
    assert zParameter(0)["z"] == 0.0
