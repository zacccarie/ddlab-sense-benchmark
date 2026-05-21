"""Tests de la renormalisation dynamique — opérateur R, quotient de bisimulation,
complexité statistique. Exemples exacts vérifiables à la main + invariants
structurels sur une vraie règle ECA."""

import numpy as np

from ddlab_bench.basins import computeBasinField
from ddlab_bench.renormalization import (
    bisimulationMetric,
    bisimulationQuotient,
    densityLabel,
    quotientGraph,
    refinePartition,
    statisticalComplexity,
    windowLabel,
)


def test_refine_split_example():
    # chaîne 0->1->2->3->3 (état 3 fixe), observable de base [0,0,1,1]
    succ = np.array([1, 2, 3, 3])
    refined = refinePartition(succ, np.array([0, 0, 1, 1]))
    # 2 et 3 : même bloc, successeurs même bloc -> restent fusionnés
    assert refined[2] == refined[3]
    # 0 et 1 : même bloc mais successeurs (1 et 2) de blocs différents -> scindés
    assert refined[0] != refined[1]
    assert int(refined.max()) + 1 == 3


def test_bisimulation_fixed_point_example():
    succ = np.array([1, 2, 3, 3])
    q = bisimulationQuotient(succ, np.array([0, 0, 1, 1]))
    assert int(q.max()) + 1 == 3
    assert q[2] == q[3]
    # un raffinement de plus ne scinde rien : point fixe
    assert int(refinePartition(succ, q).max()) == int(q.max())


def test_quotient_by_identity_is_discrete():
    field = computeBasinField(110, 8)
    size = 1 << 8
    q = bisimulationQuotient(field.succ, np.arange(size))
    # observable = identité -> aucun état n'est fusionnable
    assert int(q.max()) + 1 == size


def test_quotient_by_constant_is_trivial():
    field = computeBasinField(250, 8)
    q = bisimulationQuotient(field.succ, np.zeros(1 << 8, dtype=np.int64))
    # observable constant -> la dynamique ne distingue rien -> 1 bloc
    assert int(q.max()) + 1 == 1


def test_quotient_by_basin_is_basin():
    field = computeBasinField(110, 9)
    # attractorId est déjà invariant par succ : la partition en bassins est
    # un point fixe de R, le quotient doit la rendre telle quelle.
    q = bisimulationQuotient(field.succ, field.attractorId)
    assert int(q.max()) + 1 == len(field.attractors)
    for block in range(int(q.max()) + 1):
        members = np.where(q == block)[0]
        assert len(np.unique(field.attractorId[members])) == 1


def test_quotient_refines_label_and_is_dynamics_compatible():
    field = computeBasinField(110, 9)
    label = densityLabel(9)
    q = bisimulationQuotient(field.succ, label)
    # (i) le quotient raffine l'observable de base
    for block in range(int(q.max()) + 1):
        members = np.where(q == block)[0]
        assert len(np.unique(label[members])) == 1
    # (ii) compatible avec la dynamique : succ préserve les blocs
    quotientSucc, _ = quotientGraph(field.succ, q)
    assert np.array_equal(q[field.succ], quotientSucc[q])
    # (iii) point fixe de R
    assert int(refinePartition(field.succ, q).max()) == int(q.max())


def test_statistical_complexity_uniform():
    # 4 blocs égaux -> log2(4) = 2 bits
    assert abs(statisticalComplexity(np.array([0, 1, 2, 3])) - 2.0) < 1e-9
    # 1 bloc -> 0 bit
    assert statisticalComplexity(np.zeros(8, dtype=np.int64)) == 0.0
    # blocs de tailles [1, 1, 2] -> C_mu = 1.5 bit
    assert abs(statisticalComplexity(np.array([0, 1, 2, 2])) - 1.5) < 1e-9


def test_statistical_complexity_weighted():
    # masses : bloc 0 -> 1+2 = 3, bloc 1 -> 1  =>  p = [0.75, 0.25]
    cmu = statisticalComplexity(np.array([0, 0, 1]), weights=np.array([1.0, 2.0, 1.0]))
    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert abs(cmu - expected) < 1e-9


def test_density_and_window_labels():
    density = densityLabel(4)
    assert density[0b0000] == 0
    assert density[0b1011] == 3
    assert density[0b1111] == 4

    window = windowLabel(6, 3)
    assert window[0b101010] == 0b010
    assert window[0b000111] == 0b111
    assert len(np.unique(window)) == 8


def test_metric_symmetric_zero_diagonal_nonnegative():
    field = computeBasinField(110, 7)
    d = bisimulationMetric(field.succ, densityLabel(7), gamma=0.9)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert d.min() >= 0.0


def test_metric_zero_set_is_bisimulation_quotient():
    # d(x, x') = 0  <=>  x, x' dans le même bloc du quotient de bisimulation
    field = computeBasinField(110, 7)
    label = densityLabel(7)
    d = bisimulationMetric(field.succ, label, gamma=0.9)
    partition = bisimulationQuotient(field.succ, label)
    sameBlock = partition[:, None] == partition[None, :]
    assert np.array_equal(sameBlock, d < 1e-9)


def test_metric_fixed_point():
    field = computeBasinField(30, 7)
    label = densityLabel(7)
    a = bisimulationMetric(field.succ, label, gamma=0.9, iters=200)
    b = bisimulationMetric(field.succ, label, gamma=0.9, iters=400)
    assert np.allclose(a, b)
