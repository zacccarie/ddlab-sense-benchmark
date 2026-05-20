"""Tests — reproductibilité de l'entraînement (seed)."""

import numpy as np

from ddlab_bench.training import trainSequenceModel


def testTrainingReproducibleWithSeed():
    # même seed + mêmes données + même ordre de batch => poids identiques
    traj = np.random.default_rng(5).integers(0, 1 << 8, size=(200, 12))
    modelA = trainSequenceModel(
        traj, 8, hidden=16, epochs=2, rng=np.random.default_rng(0), seed=0,
    )
    modelB = trainSequenceModel(
        traj, 8, hidden=16, epochs=2, rng=np.random.default_rng(0), seed=0,
    )
    for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
        assert bool((paramA == paramB).all())
