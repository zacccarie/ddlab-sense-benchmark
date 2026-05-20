"""Phase 3 — diagnostic du graphe de transition appris (qualité forward + collapse).

ATTENTION — ce balayage ne teste PAS H1. La qualité de la structure inverse d'un
graphe induit est mécaniquement liée à sa qualité forward: une arête fausse
corrompt au plus deux ensembles de pré-images, donc la précision inverse reste
bornée à ±2x l'erreur forward. Aucune décorrélation possible par construction.

Ce que ce balayage mesure réellement: la qualité de la carte forward apprise et
l'effondrement (collapse) des modèles sous-entraînés ou trop petits — un modèle
effondré mappe presque tout vers un état quasi-constant, le graphe induit devient
une étoile, et goePrecision retombe au taux de base GoE.

Le vrai test de H1 est la sonde de représentation (cf. §7 de la note de projet)."""

import sys
from pathlib import Path

import numpy as np

from ddlab_bench.artifacts import loadBundle
from ddlab_bench.metrics import (
    forwardEdgeAccuracy,
    goeDetectionMetrics,
    inducedSuccessor,
    preimageSetMetrics,
)
from ddlab_bench.training import parameterCount, trainSequenceModel

HIDDEN_SWEEP = [2, 4, 8, 16, 32, 64]
EPOCHS = 18


def main(bundlePath):
    device = "cpu"
    field, data = loadBundle(bundlePath)
    n = field.n
    print(
        f"règle {field.ruleNumber}, n={n} | balayage de capacité, "
        f"{EPOCHS} epochs par modèle\n"
    )

    header = (
        f"{'hidden':>7} {'params':>8} {'fwdEdge':>9} {'preImgExact':>12} "
        f"{'preImgJacc':>11} {'goeF1':>8} {'goePrec':>8} {'goeRec':>8}"
    )
    print(header)
    print("-" * len(header))
    for hidden in HIDDEN_SWEEP:
        rng = np.random.default_rng(0)
        model = trainSequenceModel(
            data["trainTraj"], n, hidden=hidden, epochs=EPOCHS, rng=rng, device=device,
        )
        induced = inducedSuccessor(model, n, device)
        fwd = forwardEdgeAccuracy(induced, field.succ)
        pre = preimageSetMetrics(induced, field.succ)
        goe = goeDetectionMetrics(induced, field.succ)
        print(
            f"{hidden:>7} {parameterCount(model):>8} {fwd:>9.4f} "
            f"{pre['preimageExactMatch']:>12.4f} {pre['preimageJaccardMean']:>11.4f} "
            f"{goe['goeF1']:>8.4f} {goe['goePrecision']:>8.4f} {goe['goeRecall']:>8.4f}"
        )

    print(
        "\nLecture — diagnostic, pas un test de H1:"
        "\n  goePrecision proche du taux de base GoE => modèle effondré (graphe en étoile)."
        "\n  Tester H1 demande la sonde de représentation, pas l'inversion du graphe induit."
    )


if __name__ == "__main__":
    bundlePath = sys.argv[1] if len(sys.argv) > 1 else "data/rule110_n14.npz"
    if not Path(bundlePath).exists():
        sys.exit(f"bundle introuvable: {bundlePath} — lancer scripts/build_data.py")
    main(bundlePath)
