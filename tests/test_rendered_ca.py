"""Tests Phase 7 — rendu d'états et world model tokenisé."""

import numpy as np
import pytest
import torch

from ddlab_bench.models.world_model import ContrastiveEncoder, LatentDynamics, VQVAE
from ddlab_bench.rendering import renderAll, renderState


def testRenderStateShape():
    image = renderState(0b101, 9, gridSide=3, cellSize=8)
    assert image.shape == (24, 24)
    assert set(np.unique(image)).issubset({0.0, 1.0})


def testRenderAllInjective():
    # états distincts -> images distinctes : le rendu ne perd pas d'information
    images = renderAll(9, cellSize=4)
    flat = images.reshape(len(images), -1)
    assert len(np.unique(flat, axis=0)) == len(images)


def testRenderAllRejectsNonSquare():
    with pytest.raises(ValueError):
        renderAll(10)


def testVqvaeForwardShapes():
    model = VQVAE(imageSide=24, codebookSize=64)
    images = torch.zeros((8, 24, 24))
    decoded, vqLoss, codes = model(images)
    assert decoded.shape == (8, 24, 24)
    assert codes.shape == (8,)
    assert vqLoss.ndim == 0


def testLatentDynamicsForwardShape():
    model = LatentDynamics(codebookSize=64)
    codes = torch.zeros(5, dtype=torch.int64)
    assert model(codes).shape == (5, 64)


def testContrastiveEncoderShapeAndNorm():
    model = ContrastiveEncoder(imageSide=24, embedDim=32)
    z = model(torch.randn(8, 24, 24))
    assert z.shape == (8, 32)
    # embedding L2-normalisé : norme 1
    assert torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-5)
