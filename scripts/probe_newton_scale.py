"""Substrat de Newton — invariance d'échelle du shatter (le §9, rendu exact).

La frontière de Julia de Newton est de mesure nulle (dimension fractale < 2) :
le mismatch GLOBAL (moyenne sur tout le plan) s'annule quand la résolution
croît. Mais le shatter LOCAL ne s'annule pas — sur la frontière, les trois
bassins se touchent (PDF Sahari-Djellit : J = dA1 = dA2 = dA3). On mesure le
nombre moyen de bassins distincts autour d'une cellule-frontière, à résolution
croissante.

Mismatch global qui s'évanouit + éclatement local invariant d'échelle = le
« perceptuellement éclaté » du §9, exact et sans échelle."""

import numpy as np

from ddlab_bench.newton import basinGrid, boundaryBasinCount, perceptualMismatch

LAM = 1.0
RESOLUTIONS = [128, 256, 512, 1024]


def main():
    print(f"Bassins de Newton (lambda={LAM}, z^3=1) — invariance d'échelle\n")
    print(f"{'résolution':>11} {'mismatch global':>17} {'bassins/frontière':>19}")
    print("-" * 49)
    for res in RESOLUTIONS:
        labels = basinGrid(LAM, res)
        print(
            f"{res:>11} {perceptualMismatch(labels):>17.5f} "
            f"{boundaryBasinCount(labels):>19.3f}"
        )
    print(
        "\nLecture : mismatch global -> 0 (la frontière fractale est de mesure"
        "\nnulle) ; bassins/frontière reste ~3 — les trois bassins se touchent à"
        "\ntoute échelle. Le shatter est réel, exact et sans échelle, localisé"
        "\nsur l'ensemble de Julia."
    )


if __name__ == "__main__":
    main()
