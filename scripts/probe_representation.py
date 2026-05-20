"""Phase 3 — sonde de représentation, test de H1.

Entraîne le modèle A forward jusqu'à convergence, gèle ses poids, puis entraîne
des sondes (linéaire et MLP) à décoder des faits causaux inverses (garden-of-Eden,
in-degré) depuis trois jeux de features:
  - forwardRep : représentation cachée du GRU forward convergé
  - randomRep  : représentation du même GRU à son initialisation (avant entraînement)
  - rawBits    : les bits bruts de l'état

Comparaison décisive, à sonde fixée: forwardRep vs rawBits. randomRep est le même
réseau avant entraînement — il isole l'apport de l'apprentissage."""

import sys
from pathlib import Path

import numpy as np
import torch

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.models.sequence_model import SequenceModel, stateToBits
from ddlab_bench.probes import (
    LinearProbe,
    MLPProbe,
    extractRepresentations,
    goeLabels,
    inDegreeLabels,
    majorityBaseline,
    trainProbe,
)
from ddlab_bench.training import forwardAccuracy, trainSequenceModel

FORWARD_EPOCHS = 45
HIDDEN = 128


def main(bundlePath):
    device = "cpu"
    field, data = loadBundle(bundlePath)
    n = field.n
    size = 1 << n
    rng = np.random.default_rng(0)

    print(f"règle {field.ruleNumber}, n={n} | entraînement du modèle forward...")
    forwardModel = trainSequenceModel(
        data["trainTraj"], n, hidden=HIDDEN, epochs=FORWARD_EPOCHS,
        rng=rng, device=device,
    )
    fwdTrain = forwardAccuracy(forwardModel, data["trainStates"], field.succ, device)
    fwdOod = forwardAccuracy(forwardModel, data["testStates"], field.succ, device)
    print(f"  forward un-pas: train {fwdTrain:.4f} | OOD {fwdOod:.4f}")

    # randomModel = le modèle forward à son initialisation (seed 0, avant entraînement)
    torch.manual_seed(0)
    randomModel = SequenceModel(n, hidden=HIDDEN).to(device)

    featureSets = {
        "forwardRep": extractRepresentations(forwardModel, n, device),
        "randomRep": extractRepresentations(randomModel, n, device),
        "rawBits": stateToBits(np.arange(size), n),
    }
    targets = {
        "GoE (binaire)": (goeLabels(field), 2),
        "in-degré (0/1/2+)": (inDegreeLabels(field), 3),
    }
    probeKinds = {"linéaire": LinearProbe, "MLP": MLPProbe}

    perm = rng.permutation(size)
    cut = int(size * 0.7)
    trainIdx, testIdx = perm[:cut], perm[cut:]

    for targetName, (labels, numClasses) in targets.items():
        majority = majorityBaseline(labels, trainIdx, testIdx, numClasses)
        print(f"\n=== cible: {targetName} | baseline classe majoritaire {majority:.4f} ===")
        print(f"{'sonde':>11} {'features':>12} {'testAcc':>9} {'macroF1':>9}")
        print("-" * 45)
        for probeName, probeFactory in probeKinds.items():
            for featName, feats in featureSets.items():
                result = trainProbe(
                    feats, labels, trainIdx, testIdx, numClasses,
                    np.random.default_rng(1), device=device, probeFactory=probeFactory,
                )
                print(
                    f"{probeName:>11} {featName:>12} {result['testAccuracy']:>9.4f} "
                    f"{result['testMacroF1']:>9.4f}"
                )

    print(
        "\nLecture:"
        "\n  forwardRep vs rawBits (linéaire) => l'entraînement forward désintrique-t-il"
        "\n  la structure causale (la rend-il linéairement lisible) ?"
        "\n  randomRep = même réseau avant entraînement => isole l'apport de l'apprentissage."
    )


if __name__ == "__main__":
    bundlePath = sys.argv[1] if len(sys.argv) > 1 else "data/rule110_n14.npz"
    if not Path(bundlePath).exists():
        sys.exit(f"bundle introuvable: {bundlePath} — lancer scripts/build_data.py")
    main(bundlePath)
