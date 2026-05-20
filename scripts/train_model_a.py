"""Entraîne le modèle A sur un bundle et mesure la précision forward train vs OOD."""

import sys
from pathlib import Path

import numpy as np

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.training import forwardAccuracy, trainSequenceModel

EPOCHS = 30


def main(bundlePath):
    device = "cpu"
    field, data = loadBundle(bundlePath)
    rng = np.random.default_rng(0)
    print(
        f"règle {field.ruleNumber}, n={field.n} | train traj {data['trainTraj'].shape} "
        f"| OOD traj {data['testTraj'].shape}"
    )
    model = trainSequenceModel(
        data["trainTraj"], field.n, hidden=128, epochs=EPOCHS,
        rng=rng, device=device, verbose=True,
    )
    trainAcc = forwardAccuracy(model, data["trainStates"], field.succ, device)
    oodAcc = forwardAccuracy(model, data["testStates"], field.succ, device)
    print("\nprécision forward un-pas (état exact):")
    print(f"  états train : {trainAcc:.4f}  ({len(data['trainStates'])} états)")
    print(f"  états OOD   : {oodAcc:.4f}  ({len(data['testStates'])} états)")
    print(f"  écart train - OOD : {trainAcc - oodAcc:+.4f}")


if __name__ == "__main__":
    bundlePath = sys.argv[1] if len(sys.argv) > 1 else "data/rule110_n14.npz"
    if not Path(bundlePath).exists():
        sys.exit(f"bundle introuvable: {bundlePath} — lancer scripts/build_data.py")
    main(bundlePath)
