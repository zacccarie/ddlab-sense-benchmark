"""Phase 3 — homomorphisme de bassins (métrique 2).

Le partitionnement de l'espace de features (k-means, k = nombre de bassins)
reproduit-il le vrai découpage en bassins d'attraction ? AMI élevé = la
représentation regroupe les états par bassin.

Nuance: l'appartenance à un bassin est une propriété à LONG horizon (le destin
final de l'état), alors que la représentation est calculée en un pas. Ce test
mesure donc si l'entraînement forward un-pas capte une structure long-horizon."""

import sys
from pathlib import Path

import numpy as np
import torch

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.classify import zParameter
from ddlab_bench.metrics import basinHomomorphism
from ddlab_bench.models.sequence_model import SequenceModel, stateToBits
from ddlab_bench.probes import extractRepresentations
from ddlab_bench.training import forwardAccuracy, trainSequenceModel

FORWARD_EPOCHS = 45
HIDDEN = 128
N = 14
RULES = [250, 110, 30]


def runRule(rule, device):
    field, data = loadBundle(f"data/rule{rule}_n{N}.npz")
    n = field.n
    size = 1 << n
    rng = np.random.default_rng(0)

    forwardModel = trainSequenceModel(
        data["trainTraj"], n, hidden=HIDDEN, epochs=FORWARD_EPOCHS, rng=rng, device=device,
    )
    torch.manual_seed(0)
    randomModel = SequenceModel(n, hidden=HIDDEN).to(device)

    featureSets = {
        "forwardRep": extractRepresentations(forwardModel, n, device),
        "randomRep": extractRepresentations(randomModel, n, device),
        "rawBits": stateToBits(np.arange(size), n),
    }
    row = {
        "z": zParameter(rule)["z"],
        "basinCount": len(field.attractors),
        "fwdAcc": forwardAccuracy(forwardModel, np.arange(size), field.succ, device),
    }
    for featName, feats in featureSets.items():
        row[featName] = basinHomomorphism(feats, field.attractorId, seed=0)
    return row


def main():
    device = "cpu"
    print(f"homomorphisme de bassins | {FORWARD_EPOCHS} epochs forward, k-means\n")
    rows = {}
    for rule in RULES:
        print(f"  règle {rule}...")
        rows[rule] = runRule(rule, device)

    header = (
        f"{'règle':>6} {'Z':>5} {'#bassins':>9} {'fwdAcc':>8} "
        f"{'forwardRep AMI':>15} {'randomRep AMI':>14} {'rawBits AMI':>13}"
    )
    print("\n" + header)
    print("-" * len(header))
    for rule in RULES:
        row = rows[rule]
        print(
            f"{rule:>6} {row['z']:>5.2f} {row['basinCount']:>9} {row['fwdAcc']:>8.4f} "
            f"{row['forwardRep']['ami']:>15.4f} {row['randomRep']['ami']:>14.4f} "
            f"{row['rawBits']['ami']:>13.4f}"
        )
    print(
        "\nAMI = Adjusted Mutual Information entre clusters k-means et vrais bassins."
        "\nforwardRep >> rawBits => la représentation forward regroupe les états par bassin."
    )


if __name__ == "__main__":
    missing = [r for r in RULES if not Path(f"data/rule{r}_n{N}.npz").exists()]
    if missing:
        sys.exit(f"bundles manquants {missing} — lancer scripts/build_data.py")
    main()
