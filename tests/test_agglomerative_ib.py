"""Tests — Information Bottleneck agglomératif (hiérarchie de quotients émergente)."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import computeBasinField
from ddlab_bench.agglomerative_ib import agglomerativeIB, mergeCost, partitionAt


def _oneHot(labels, numClasses):
    out = np.zeros((len(labels), numClasses))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def test_merge_cost_zero_for_identical_distributions():
    pY = np.array([0.3, 0.7])
    assert mergeCost(0.5, pY, 0.5, pY) < 1e-12


def test_merge_cost_positive_for_different_distributions():
    assert mergeCost(0.5, np.array([1.0, 0.0]), 0.5, np.array([0.0, 1.0])) > 0.0


def test_shapes_and_endpoints():
    pYgivenX = _oneHot(np.array([0, 0, 1, 1, 2]), 3)
    merges, assignments = agglomerativeIB(pYgivenX)
    assert len(merges) == 4          # sizeX - 1 fusions
    assert len(assignments) == 5     # sizeX partitions
    assert int(assignments[0].max()) + 1 == 5    # identité
    assert int(assignments[-1].max()) + 1 == 1   # bloc unique


def test_info_non_increasing_and_costs_nonnegative():
    rng = np.random.default_rng(0)
    pYgivenX = rng.dirichlet(np.ones(4), size=16)
    merges, _ = agglomerativeIB(pYgivenX)
    costs = [c for c, _ in merges]
    infos = [info for _, info in merges]
    assert all(c >= -1e-12 for c in costs)
    assert all(infos[k] <= infos[k - 1] + 1e-12 for k in range(1, len(infos)))
    assert abs(infos[-1]) < 1e-9     # un seul bloc -> I(Z;Y) = 0


def test_partition_at():
    pYgivenX = _oneHot(np.array([0, 0, 1, 1, 2]), 3)
    _, assignments = agglomerativeIB(pYgivenX)
    assert int(partitionAt(assignments, 5).max()) + 1 == 5
    assert int(partitionAt(assignments, 1).max()) + 1 == 1


def test_ib_recovers_basins_when_relevant_variable_is_fate():
    # variable pertinente Y = le bassin atteint. Les fusions intra-bassin ont un
    # coût nul (même Y) ; l'IB agglomératif les fait toutes EN PREMIER -> la
    # partition en bassins émerge comme la compression IB sans perte.
    field = computeBasinField(110, 8)
    nBasins = len(field.attractors)
    pYgivenX = _oneHot(field.attractorId, nBasins)
    merges, assignments = agglomerativeIB(pYgivenX)
    size = len(field.attractorId)

    # toutes les fusions menant à nBasins blocs sont à coût ~ nul
    intraBasin = [c for c, _ in merges[: size - nBasins]]
    assert max(intraBasin) < 1e-9
    # et cette partition est exactement la partition en bassins
    emerged = partitionAt(assignments, nBasins)
    assert adjusted_mutual_info_score(field.attractorId, emerged) > 0.999
