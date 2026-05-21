"""Tests Phase 4 — modèle de résumé de trajectoire."""

import numpy as np
import torch

from ddlab_bench import computeBasinField, generateTrajectories
from ddlab_bench.models.summarizer import TrajectorySummarizer, trajectoryBasinLabels


def testSummarizerOutputShapes():
    model = TrajectorySummarizer(n=14, numBasins=15, bottleneck=4)
    trajBits = torch.zeros((8, 24, 14))
    code = model.encode(trajBits)
    assert code.shape == (8, 4)
    logits = model(trajBits)
    assert logits.shape == (8, 15)


def testSummarizerBottleneckLimitsCodeWidth():
    # le code a exactement la largeur du goulot demandé
    for bottleneck in (1, 3, 16):
        model = TrajectorySummarizer(n=8, numBasins=5, bottleneck=bottleneck)
        code = model.encode(torch.zeros((4, 10, 8)))
        assert code.shape[1] == bottleneck


def testTrajectoryBasinLabelConstantAlongTrajectory():
    # vérité-terrain: tous les états d'une trajectoire sont dans le même bassin
    field = computeBasinField(110, 10)
    traj = generateTrajectories(field, 200, 12, np.random.default_rng(0))
    labels, numBasins = trajectoryBasinLabels(field, traj)
    assert len(labels) == 200
    assert numBasins >= 1
    for i in range(200):
        basinsAlong = field.attractorId[traj[i]]
        assert len(set(basinsAlong.tolist())) == 1
