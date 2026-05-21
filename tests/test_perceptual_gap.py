"""Tests Phase 11 (axe C) — désaccord apparence / destin."""

from ddlab_bench import computeBasinField
from ddlab_bench.classify import perceptualBasinMismatch, randomBasinMismatch


def testMismatchZeroWhenSingleBasin():
    # règle 0 : tout converge vers l'état 0 -> un seul bassin -> aucun voisin
    # ne change de bassin
    field = computeBasinField(0, 8)
    assert perceptualBasinMismatch(field) == 0.0


def testMismatchOneWhenEveryStateOwnBasin():
    # règle 204 (identité) : chaque état est son propre point fixe -> tout
    # voisin perceptuel est dans un autre bassin
    field = computeBasinField(204, 8)
    assert perceptualBasinMismatch(field) == 1.0


def testRandomMismatchInBounds():
    field = computeBasinField(110, 9)
    baseline = randomBasinMismatch(field)
    assert 0.0 <= baseline < 1.0
