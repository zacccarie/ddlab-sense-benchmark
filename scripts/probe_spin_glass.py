"""Verre de spin — la hiérarchie émerge-t-elle EX NIHILO ?

§16 plante les taux, §18 plante la topologie. Ici rien n'est planté : un Ising
à couplages aléatoires i.i.d., dynamique de Glauber. Le paysage d'énergie — ses
minima métastables, leur organisation — émerge des seuls couplages aléatoires.
Énumérable (2^n états) -> vérité-terrain exacte.

Test : en abaissant la température, une hiérarchie métastable multi-niveaux
apparaît-elle dans le spectre de transfert ? (les gaps de temps de relaxation
sont les niveaux). C'est la frontière — émergence sans plantation."""

import numpy as np

from ddlab_bench.markov import spinGlassChain

N = 10
TEMPERATURES = [2.0, 1.0, 0.5, 0.25]
SEED = 0


def gapCount(timescales, factor=3.0):
    """Nombre de gaps (ratio > factor entre temps de relaxation consécutifs)
    parmi les modes lents — estimation de la profondeur de la hiérarchie."""
    slow = np.sort(timescales)[::-1][1:13]  # hors mode trivial
    ratios = slow[:-1] / np.maximum(slow[1:], 1e-9)
    return int(np.sum(ratios > factor))


def main():
    size = 1 << N
    print(f"Verre de spin — émergence ex nihilo | n={N} spins, {size} états\n")

    # minima d'énergie locaux — états métastables, propriété du paysage (hors T)
    _, energy = spinGlassChain(N, 1.0, seed=SEED)
    isMin = np.array([
        all(energy[s] <= energy[s ^ (1 << i)] for i in range(N))
        for s in range(size)
    ])
    print(f"minima d'énergie locaux (états métastables) : {int(isMin.sum())}\n")

    print(f"  {'T':>6} {'#gaps':>6}  temps de relaxation, modes lents")
    print("  " + "-" * 56)
    for T in TEMPERATURES:
        chain, _ = spinGlassChain(N, T, seed=SEED)
        ts = chain.impliedTimescales()
        slow = np.sort(ts)[::-1][1:7]
        print(f"  {T:>6.2f} {gapCount(ts):>6}  "
              + " ".join(f"{x:8.1f}" for x in slow))
    print(
        "\nLecture : T haute -> mélange rapide, pas de métastabilité."
        "\n T basse -> si des gaps de temps de relaxation apparaissent, une"
        "\n hiérarchie métastable multi-niveaux a ÉMERGÉ des seuls couplages"
        "\n aléatoires — émergence ex nihilo. Peu de gaps -> paysage plat à"
        "\n cette taille de système."
    )


if __name__ == "__main__":
    main()
