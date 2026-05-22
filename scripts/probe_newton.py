"""Substrat de Newton — bassins continus à frontière fractale (le §9 en continu).

Balaye lambda (le bouton de complexité, analogue du Z de Wuensche) ; pour chaque
valeur : désaccord apparence/destin sur la grille (le §9) et dimension fractale
de la frontière des bassins (la méthode du PDF Sahari-Djellit).

Question : la frontière fractale est-elle le « perceptuellement éclaté » du §9
rendu exact et sans échelle ? Le bassin est une vérité-terrain exacte (itération
de la carte continue), pas une quantification — contrairement au CML du §13."""

import numpy as np

from ddlab_bench.newton import basinGrid, boxCountingDimension, perceptualMismatch

RESOLUTION = 512
LAMBDAS = [0.5, 0.7, 1.0, 1.3, 1.6, 1.8]


def main():
    print(f"Bassins de Newton modifié — grille {RESOLUTION}x{RESOLUTION}, z^3 = 1\n")
    print(f"{'lambda':>8} {'mismatch':>10} {'dim fractale':>14} {'non-conv.':>11}")
    print("-" * 46)
    for lam in LAMBDAS:
        labels = basinGrid(lam, RESOLUTION)
        mismatch = perceptualMismatch(labels)
        dimension = boxCountingDimension(labels)
        nonConv = float(np.mean(labels < 0)) * 100.0
        print(
            f"{lam:>8.2f} {mismatch:>10.4f} {dimension:>14.3f} {nonConv:>10.1f}%"
        )
    print(
        "\nLecture : la dimension fractale croît avec lambda (PDF Sahari-Djellit) ;"
        "\nmismatch > 0 = bassins perceptuellement éclatés — le §9, en continu et"
        "\nexact. Frontière fractale => le mismatch ne s'annule pas en raffinant"
        "\nla grille : l'éclatement est sans échelle."
    )


if __name__ == "__main__":
    main()
