"""Phase 3 — sonde de représentation, test de H1.

Entraîne le modèle A forward jusqu'à convergence, gèle ses poids, puis entraîne
des sondes (linéaire et MLP) à décoder des faits causaux inverses (garden-of-Eden,
in-degré) depuis trois jeux de features:
  - forwardRep : représentation cachée du GRU forward convergé
  - randomRep  : représentation d'un GRU non entraîné (même architecture)
  - rawBits    : les bits bruts de l'état

Comparaison décisive, à sonde fixée: forwardRep vs randomRep. Si l'entraînement
forward n'ajoute rien au-delà des features aléatoires, la compétence forward ne
porte pas la structure causale -> H1 confirmé."""

import sys
from pathlib import Path

import numpy as np

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.models.sequence_model import SequenceModel, stateToBits
from ddlab_bench.probes import (
    LinearProbe,
    MLPProbe,
    goeLabels,
    inDegreeLabels,
    majorityBaseline,
    trainProbe,
)
from ddlab_bench.training import forwardAccuracy, trainSequenceModel

FORWARD_EPOCHS = 45
HIDDEN = 128


def extractAllRepresentations(model, n, device, chunk=4096):
    """Représentation cachée du modèle pour tous les états de l'espace."""
    size = 1 << n
    feats = np.empty((size, model.gru.hidden_size), dtype=np.float32)
    for start in range(0, size, chunk):
        states = np.arange(start, min(start + chunk, size))
        feats[states] = model.representation(states, device)
    return feats


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

    randomModel = SequenceModel(n, hidden=HIDDEN).to(device)
    featureSets = {
        "forwardRep": extractAllRepresentations(forwardModel, n, device),
        "randomRep": extractAllRepresentations(randomModel, n, device),
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
                probeRng = np.random.default_rng(1)
                result = trainProbe(
                    feats, labels, trainIdx, testIdx, numClasses, probeRng,
                    device=device, probeFactory=probeFactory,
                )
                print(
                    f"{probeName:>11} {featName:>12} {result['testAccuracy']:>9.4f} "
                    f"{result['testMacroF1']:>9.4f}"
                )

    print(
        "\nLecture:"
        "\n  Comparaison décisive, à sonde fixée: forwardRep vs randomRep."
        "\n  linéaire, forwardRep >> randomRep => structure causale linéairement lisible"
        "\n                                      (apport de l'entraînement forward)."
        "\n  MLP forwardRep vs forwardRep lin. => part de structure causale non-linéaire."
        "\n  MLP rawBits élevé est attendu (le MLP recalcule la cible depuis l'état);"
        "\n  seul forwardRep vs randomRep, à sonde égale, isole l'apport de l'apprentissage."
    )


if __name__ == "__main__":
    bundlePath = sys.argv[1] if len(sys.argv) > 1 else "data/rule110_n14.npz"
    if not Path(bundlePath).exists():
        sys.exit(f"bundle introuvable: {bundlePath} — lancer scripts/build_data.py")
    main(bundlePath)
