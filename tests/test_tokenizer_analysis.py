"""Tests Phase 7-8 — métriques d'analyse du tokeniseur."""

import numpy as np

from ddlab_bench import computeBasinField, iterateSuccessor
from ddlab_bench.tokenizer_analysis import metricsForCode


def testBijectiveCodePreservesStructure():
    # tokenisation bijective (code = identité) => structure parfaitement préservée
    field = computeBasinField(110, 9)
    code = np.arange(1 << 9, dtype=np.int64)
    m = metricsForCode(code, field)
    assert m["collisionRate"] == 0.0
    assert m["consistent"] == 1.0
    assert m["ami"] > 0.999
    assert m["latentBasins"] == len(field.attractors)


def testConstantCodeDestroysStructure():
    # tous les états sur un seul code => information et structure détruites
    field = computeBasinField(110, 9)
    code = np.zeros(1 << 9, dtype=np.int64)
    m = metricsForCode(code, field)
    assert m["collisionRate"] > 0.99
    assert m["latentBasins"] == 1
    assert m["ami"] < 0.01


def testHorizonQuotientCompressesAndPreserves():
    # code_H(s) = succ^H(s) ne fusionne que des états convergents (même bassin)
    # => il compresse fortement tout en préservant exactement la structure
    field = computeBasinField(110, 9)
    horizon = int(field.transient.max()) + 4
    m = metricsForCode(iterateSuccessor(field.succ, horizon), field)
    assert m["collisionRate"] > 0.5
    assert m["ami"] > 0.999
    assert m["latentBasins"] == len(field.attractors)
