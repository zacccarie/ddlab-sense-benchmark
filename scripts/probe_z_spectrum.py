"""Phase 3 — balayage du spectre de complexité (H2).

Rejoue la sonde de représentation linéaire sur des règles ECA de régimes
différents (ordonné -> complexe -> chaotique). Question H2: la désintrication
forward->causal (lift = forwardRep - rawBits) varie-t-elle avec la complexité ?

Reporte accuracy ET macroF1: aux régimes extrêmes l'étiquette GoE est très
déséquilibrée et l'accuracy sature au taux de base — le macroF1 est la mesure
honnête."""

import sys
from pathlib import Path

import numpy as np

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.classify import structuralSummary, zParameter
from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.probes import extractRepresentations, goeLabels, inDegreeLabels, trainProbe
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
    fwdAcc = forwardAccuracy(forwardModel, np.arange(size), field.succ, device)
    forwardRep = extractRepresentations(forwardModel, n, device)
    rawBits = stateToBits(np.arange(size), n)

    perm = rng.permutation(size)
    cut = int(size * 0.7)
    trainIdx, testIdx = perm[:cut], perm[cut:]

    row = {"z": zParameter(rule)["z"], "structure": structuralSummary(field), "fwdAcc": fwdAcc}
    for name, labels, numClasses in [
        ("goe", goeLabels(field), 2),
        ("indeg", inDegreeLabels(field), 3),
    ]:
        raw = trainProbe(rawBits, labels, trainIdx, testIdx, numClasses, np.random.default_rng(1))
        fwd = trainProbe(forwardRep, labels, trainIdx, testIdx, numClasses, np.random.default_rng(1))
        row[name] = {
            "rawAcc": raw["testAccuracy"], "fwdRepAcc": fwd["testAccuracy"],
            "rawF1": raw["testMacroF1"], "fwdRepF1": fwd["testMacroF1"],
        }
    return row


def _liftCell(rawValue, fwdValue):
    return f"{rawValue:.3f}→{fwdValue:.3f} ({fwdValue - rawValue:+.3f})"


def _printTable(rows, title, rawKey, fwdKey):
    header = (
        f"{'règle':>6} {'Z':>5} {'GoE%':>7} {'fwdAcc':>8} "
        f"{'GoE: raw→fwd (lift)':>26} {'inDeg: raw→fwd (lift)':>26}"
    )
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for rule in RULES:
        row = rows[rule]
        print(
            f"{rule:>6} {row['z']:>5.2f} {row['structure']['goeFraction'] * 100:>6.1f}% "
            f"{row['fwdAcc']:>8.4f} "
            f"{_liftCell(row['goe'][rawKey], row['goe'][fwdKey]):>26} "
            f"{_liftCell(row['indeg'][rawKey], row['indeg'][fwdKey]):>26}"
        )


def main():
    device = "cpu"
    print(f"balayage du spectre Z | {FORWARD_EPOCHS} epochs forward, sonde linéaire\n")
    rows = {}
    for rule in RULES:
        print(f"  règle {rule}...")
        rows[rule] = runRule(rule, device)

    _printTable(rows, "lift accuracy (forwardRep - rawBits)", "rawAcc", "fwdRepAcc")
    _printTable(rows, "lift macroF1 (forwardRep - rawBits)", "rawF1", "fwdRepF1")
    print(
        "\nLe macroF1 est la mesure honnête: aux régimes 250 (84% GoE) et 30 (5% GoE)"
        "\nl'accuracy sature au taux de base. H2 — le lift varie-t-il avec la complexité ?"
    )


if __name__ == "__main__":
    missing = [r for r in RULES if not Path(f"data/rule{r}_n{N}.npz").exists()]
    if missing:
        sys.exit(f"bundles manquants {missing} — lancer scripts/build_data.py")
    main()
