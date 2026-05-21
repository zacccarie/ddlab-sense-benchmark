"""Phase 6 — généralisation aux réseaux booléens aléatoires (RBN).

Rejoue la sonde de représentation sur des RBN (topologie IRRÉGULIÈRE, modèle de
Kauffman) à connectivité K croissante : K=1 ordonné, K=2 critique (bord du chaos),
K=3 chaotique. Question : le résultat edge-of-chaos obtenu sur les ECA — la
désintrication causale piquée au régime complexe — survit-il à une topologie
irrégulière ?

Un seul RBN par K (le RBN est aléatoire) : résultat indicatif, à moyenner sur
plusieurs instances pour conclure."""

import numpy as np

from ddlab_bench.basins import basinFieldFromSuccessor
from ddlab_bench.classify import structuralSummary
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.probes import extractRepresentations, goeLabels, inDegreeLabels, trainProbe
from ddlab_bench.rbn import RandomBooleanNetwork
from ddlab_bench.trajectories import generateTrajectories
from ddlab_bench.training import forwardAccuracy, trainSequenceModel

N = 14
K_VALUES = [1, 2, 3]
REGIMES = {1: "ordonné", 2: "critique", 3: "chaotique"}
RBN_SEED = 0
FORWARD_EPOCHS = 45
HIDDEN = 128
TRAJ_COUNT = 4000
TRAJ_LENGTH = 24


def probeMacroF1(features, labels, trainIdx, testIdx, numClasses):
    return trainProbe(
        features, labels, trainIdx, testIdx, numClasses, np.random.default_rng(1),
    )["testMacroF1"]


def runRbn(k, device):
    rbn = RandomBooleanNetwork(N, k, seed=RBN_SEED)
    field = basinFieldFromSuccessor(rbn.successorArray(), N, RBN_SEED)
    size = 1 << N
    rng = np.random.default_rng(0)

    traj = generateTrajectories(field, TRAJ_COUNT, TRAJ_LENGTH, rng)
    forwardModel = trainSequenceModel(
        traj, N, hidden=HIDDEN, epochs=FORWARD_EPOCHS, rng=rng, device=device,
    )
    fwdAcc = forwardAccuracy(forwardModel, np.arange(size), field.succ, device)
    forwardRep = extractRepresentations(forwardModel, N, device)
    rawBits = stateToBits(np.arange(size), N)

    perm = rng.permutation(size)
    cut = int(size * 0.7)
    trainIdx, testIdx = perm[:cut], perm[cut:]

    row = {
        "basinCount": len(field.attractors),
        "fwdAcc": fwdAcc,
        "structure": structuralSummary(field),
    }
    for name, labels, numClasses in [
        ("goe", goeLabels(field), 2),
        ("indeg", inDegreeLabels(field), 3),
    ]:
        row[name] = {
            "raw": probeMacroF1(rawBits, labels, trainIdx, testIdx, numClasses),
            "forward": probeMacroF1(forwardRep, labels, trainIdx, testIdx, numClasses),
        }
    return row


def main():
    device = "cpu"
    print(f"RBN n={N}, balayage K | {FORWARD_EPOCHS} epochs forward, sonde linéaire (macroF1)\n")
    rows = {}
    for k in K_VALUES:
        print(f"  K={k}...")
        rows[k] = runRbn(k, device)

    header = (
        f"{'K':>3} {'régime':>10} {'#bassins':>9} {'GoE%':>7} {'fwdAcc':>8} "
        f"{'GoE: raw→fwd (lift)':>26} {'inDeg: raw→fwd (lift)':>26}"
    )
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        row = rows[k]
        goe, indeg = row["goe"], row["indeg"]
        goeCell = f"{goe['raw']:.3f}→{goe['forward']:.3f} ({goe['forward'] - goe['raw']:+.3f})"
        indegCell = (
            f"{indeg['raw']:.3f}→{indeg['forward']:.3f} "
            f"({indeg['forward'] - indeg['raw']:+.3f})"
        )
        print(
            f"{k:>3} {REGIMES[k]:>10} {row['basinCount']:>9} "
            f"{row['structure']['goeFraction'] * 100:>6.1f}% {row['fwdAcc']:>8.4f} "
            f"{goeCell:>26} {indegCell:>26}"
        )
    print(
        "\nLift macroF1 = forwardRep - rawBits (désintrication causale)."
        "\nH — le lift culmine-t-il à K=2 (bord du chaos RBN), comme au régime"
        "\ncomplexe des ECA ? Si oui, le résultat edge-of-chaos ne dépend pas de"
        "\nla topologie régulière de l'automate cellulaire."
    )


if __name__ == "__main__":
    main()
