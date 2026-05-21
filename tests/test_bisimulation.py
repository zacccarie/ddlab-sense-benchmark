"""Tests de l'objectif de renormalisation appris (§11) — formes, décroissance
de la perte, et invariant : l'objectif de base sépare bien l'observable."""

import numpy as np
import torch

from ddlab_bench.basins import computeBasinField
from ddlab_bench.bisimulation import (
    BisimulationEncoder,
    embedAllStates,
    trainBisimulationEncoder,
    trainInvarianceEncoder,
)
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.renormalization import densityLabel


def test_encoder_output_shape():
    encoder = BisimulationEncoder(featureDim=10, latentDim=8)
    out = encoder(torch.zeros(5, 10))
    assert out.shape == (5, 8)


def test_training_reduces_loss():
    # gamma=0.9 : cas récursif — sans réseau cible il diverge ; le test garde
    # que la perte reste finie et décroît (moyenne de queue sous le départ).
    field = computeBasinField(110, 8)
    features = stateToBits(np.arange(1 << 8), 8)
    _, history = trainBisimulationEncoder(
        features, field.succ, densityLabel(8), gamma=0.9, epochs=300, seed=0
    )
    assert np.isfinite(history[-1])
    assert np.mean(history[-30:]) < history[0]


def test_embed_all_states_shape():
    field = computeBasinField(110, 8)
    features = stateToBits(np.arange(1 << 8), 8)
    encoder, _ = trainBisimulationEncoder(
        features, field.succ, densityLabel(8), gamma=0.0, epochs=20,
        latentDim=12, seed=0,
    )
    embedding = embedAllStates(encoder, features)
    assert embedding.shape == (1 << 8, 12)


def test_base_objective_separates_label():
    # gamma=0 : phi doit satisfaire d(phi) ~= 1[l != l'] -> les états de même
    # densité doivent être en moyenne plus proches que ceux de densité différente.
    field = computeBasinField(110, 8)
    features = stateToBits(np.arange(1 << 8), 8)
    label = densityLabel(8)
    encoder, _ = trainBisimulationEncoder(
        features, field.succ, label, gamma=0.0, epochs=300, seed=0
    )
    embedding = embedAllStates(encoder, features)
    diff = embedding[:, None, :] - embedding[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=-1))
    sameLabel = label[:, None] == label[None, :]
    assert distance[sameLabel].mean() < distance[~sameLabel].mean()


def test_invariance_encoder_shape_and_no_collapse():
    field = computeBasinField(110, 8)
    features = stateToBits(np.arange(1 << 8), 8)
    encoder, history = trainInvarianceEncoder(features, field.succ, epochs=200, seed=0)
    embedding = embedAllStates(encoder, features)
    assert embedding.shape == (1 << 8, 16)
    assert np.isfinite(history[-1])
    # les termes de variance / covariance interdisent l'effondrement sur une constante
    assert embedding.std(axis=0).mean() > 0.1


def test_invariance_encoder_recovers_basins_better_than_raw():
    # phi succ-invariant est constant sur les bassins -> mieux aligné sur la
    # partition en bassins que les bits bruts (la cible perceptuelle du §9).
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score

    field = computeBasinField(110, 8)
    features = stateToBits(np.arange(1 << 8), 8)
    basins = field.attractorId
    k = len(field.attractors)
    encoder, _ = trainInvarianceEncoder(features, field.succ, epochs=600, seed=0)
    embedding = embedAllStates(encoder, features)

    def ami(x):
        clusters = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(x)
        return adjusted_mutual_info_score(basins, clusters)

    assert ami(embedding) > ami(features)
