"""Tests du substrat de Newton — bassins continus à frontière fractale."""

import numpy as np

from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.newton import (
    ROOTS,
    basinGrid,
    basinLabel,
    boundaryBasinCount,
    boxCountingDimension,
    newtonStep,
    perceptualMismatch,
    quantizedSuccessor,
)


def test_roots_are_fixed_points():
    # z^3 = 1 aux racines -> le pas de Newton ne les déplace pas
    for lam in (0.5, 1.0, 1.7):
        assert np.allclose(newtonStep(ROOTS, lam), ROOTS)


def test_basin_label_identifies_each_root():
    # un point très proche d'une racine converge vers son propre bassin
    labels = basinLabel(ROOTS + 1e-3, lam=1.0)
    assert labels.tolist() == [0, 1, 2]


def test_basin_grid_shape_and_labels():
    labels = basinGrid(lam=1.0, resolution=64)
    assert labels.shape == (64, 64)
    assert set(np.unique(labels)).issubset({-1, 0, 1, 2})
    # le plan contient bien les trois bassins
    assert {0, 1, 2}.issubset(set(np.unique(labels)))


def test_perceptual_mismatch_bounds():
    uniform = np.zeros((32, 32), dtype=np.int64)
    assert perceptualMismatch(uniform) == 0.0
    # damier : tout voisin diffère -> mismatch 1.0
    checker = np.indices((32, 32)).sum(axis=0) % 2
    assert perceptualMismatch(checker) == 1.0


def test_box_dimension_straight_boundary_is_one():
    # frontière droite (1D) -> dimension fractale ~ 1
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[:, 64:] = 1
    assert 0.8 < boxCountingDimension(labels) < 1.25


def test_box_dimension_newton_boundary_is_fractal():
    # la frontière de Julia est fractale -> dimension strictement entre 1 et 2
    dim = boxCountingDimension(basinGrid(lam=1.0, resolution=256))
    assert 1.0 < dim < 2.0


def test_boundary_basin_count_two_basin_boundary():
    # frontière droite, 2 bassins seulement -> au plus 2 bassins par fenêtre
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[:, 64:] = 1
    assert boundaryBasinCount(labels) == 2.0


def test_boundary_basin_count_newton_is_multi_basin():
    # frontière de Julia : les trois bassins se touchent -> compte > 2
    labels = basinGrid(lam=1.0, resolution=256)
    assert boundaryBasinCount(labels) > 2.0


def test_quantized_successor():
    succ, positions, basins = quantizedSuccessor(lam=1.0, resolution=32)
    size = 32 * 32
    assert succ.shape == (size,)
    assert positions.shape == (size, 2)
    assert succ.min() >= 0 and succ.max() < size
    assert set(np.unique(basins)).issubset({-1, 0, 1, 2})
    # graphe fonctionnel quantifié -> champ de bassins calculable
    field = basinFieldFromSuccessor(succ, 0, "newton-quant")
    assert len(field.attractors) >= 1
