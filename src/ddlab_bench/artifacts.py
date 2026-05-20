"""Persistance des artefacts de benchmark.

Un bundle = champ de bassins (vérité-terrain) + trajectoires train/OOD d'une
règle, sauvé en .npz compressé. Le champ de bassins étant une fonction pure de
`succ`, les pré-images sont recalculées au chargement plutôt que stockées."""

from pathlib import Path

import numpy as np

from ddlab_bench.basins import BasinField, buildPreimageArray
from ddlab_bench.trajectories import generateTrajectories, splitByBasins


def _flattenAttractors(attractors):
    """Concatène les cycles en un array plat + offsets (format CSR)."""
    offsets = np.zeros(len(attractors) + 1, dtype=np.int64)
    for i, cycle in enumerate(attractors):
        offsets[i + 1] = offsets[i] + len(cycle)
    flat = np.empty(int(offsets[-1]), dtype=np.int64)
    for i, cycle in enumerate(attractors):
        flat[offsets[i]:offsets[i + 1]] = cycle
    return flat, offsets


def _rebuildAttractors(flat, offsets):
    return [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def buildBundle(field, trajConfig, rng, path):
    """Construit et sauve un bundle: champ de bassins + trajectoires train/OOD.

    trajConfig = dict(count, length, holdoutFraction). Le split OOD retient des
    bassins entiers; les trajectoires test partent exclusivement de ces bassins."""
    split = splitByBasins(field, trajConfig["holdoutFraction"], rng)
    if not split["oodPossible"]:
        raise ValueError(f"règle {field.ruleNumber}: bassin unique, split OOD impossible")

    count = trajConfig["count"]
    length = trajConfig["length"]
    trainTraj = generateTrajectories(field, count, length, rng, split["trainStates"])
    testTraj = generateTrajectories(field, count, length, rng, split["testStates"])
    flat, offsets = _flattenAttractors(field.attractors)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ruleNumber=field.ruleNumber,
        n=field.n,
        succ=field.succ,
        attractorId=field.attractorId,
        transient=field.transient,
        goe=np.asarray(field.goe, dtype=np.int64),
        attractorsFlat=flat,
        attractorOffsets=offsets,
        trainStates=split["trainStates"],
        testStates=split["testStates"],
        trainBasins=np.asarray(sorted(split["trainBasins"]), dtype=np.int64),
        testBasins=np.asarray(sorted(split["testBasins"]), dtype=np.int64),
        trainTraj=trainTraj,
        testTraj=testTraj,
    )
    return path


def loadBundle(path):
    """Recharge un bundle. Retourne (field, data), data portant trajectoires et splits."""
    raw = np.load(path)
    succ = raw["succ"]
    field = BasinField(
        ruleNumber=int(raw["ruleNumber"]),
        n=int(raw["n"]),
        succ=succ,
        preimages=buildPreimageArray(succ),
        attractorId=raw["attractorId"],
        attractors=_rebuildAttractors(raw["attractorsFlat"], raw["attractorOffsets"]),
        transient=raw["transient"],
        goe=raw["goe"].tolist(),
    )
    data = {
        "trainStates": raw["trainStates"],
        "testStates": raw["testStates"],
        "trainBasins": raw["trainBasins"],
        "testBasins": raw["testBasins"],
        "trainTraj": raw["trainTraj"],
        "testTraj": raw["testTraj"],
    }
    return field, data
