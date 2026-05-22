"""Tests — transformer de trajectoire (objectif récursif §11, architecture world model)."""

import numpy as np
import torch

from ddlab_bench import computeBasinField
from ddlab_bench.bisimulation import (
    embedAllStates,
    orbitWindows,
    trainTrajectoryTransformer,
)
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.models.trajectory_transformer import TrajectoryTransformer
from ddlab_bench.renormalization import densityLabel


def test_transformer_output_shape():
    model = TrajectoryTransformer(n=9, windowLength=8, latentDim=16)
    out = model(torch.zeros(5, 8, 9))
    assert out.shape == (5, 16)


def test_orbit_windows():
    field = computeBasinField(110, 7)
    size = 1 << 7
    orbits = orbitWindows(field.succ, 7, 5)
    assert orbits.shape == (size, 5, 7)
    # tranche 0 = l'état lui-même ; tranche 1 = son successeur
    assert np.array_equal(orbits[:, 0, :], stateToBits(np.arange(size), 7))
    assert np.array_equal(orbits[:, 1, :], stateToBits(field.succ, 7))


def test_trajectory_transformer_trains_without_collapse():
    field = computeBasinField(110, 7)
    orbits = orbitWindows(field.succ, 7, 5)
    encoder, history = trainTrajectoryTransformer(
        orbits, field.succ, densityLabel(7), epochs=40, seed=0)
    embedding = embedAllStates(encoder, orbits)
    assert embedding.shape == (1 << 7, 16)
    assert np.all(np.isfinite(embedding))
    assert np.isfinite(history[-1])
    # le terme d'ancre interdit l'effondrement sur une constante
    assert embedding.std(axis=0).mean() > 1e-3
