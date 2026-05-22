"""Bifurcations du quotient — quand le paysage se déforme (§25, l'axe des paramètres).

Le §16-24 audite un graphe de transition FIXE. Le concept d'homéodynamique
(Petitjean, Finck & Schmoll 2025) distingue le déplacement DANS un paysage de la
déformation DU paysage. Bifurcations, hystérésis, transitions critiques vivent
sur l'axe des PARAMÈTRES — l'axe que le banc d'essai n'avait jamais balayé.

À chaque valeur du paramètre, le spectre de l'opérateur de transfert donne la
structure métastable EXACTE (chaîne énumérable). Le nombre d'ensembles
métastables = nombre de modes lents (|lambda| > 0.99) ; une BIFURCATION = un
saut de ce nombre — un bassin se scinde, une barrière apparaît.

Deux substrats, et le résultat est un contraste :
  A. ECA bruité, paramètre epsilon — paysage métastable plat (cf. §17) ;
  B. verre de spin, paramètre température — paysage rugueux.
Comme une hiérarchie (papier 2), une cascade de bifurcations a besoin d'un
paysage : seul B en produit une."""

import numpy as np

from ddlab_bench.markov import noisyEcaChain, spinGlassChain

N = 9
SLOW = 0.99  # seuil |lambda| pour qu'un mode compte comme métastable
EPSILONS = np.linspace(0.0, 0.30, 16)
TEMPERATURES = np.geomspace(4.0, 0.15, 18)
SEEDS = [0, 1, 2]


def metastableCount(chain):
    """Nombre de modes lents — valeurs propres de transfert de module > SLOW."""
    values, _ = chain.transferSpectrum()
    return int(np.sum(np.abs(values) > SLOW))


def bifurcations(params, counts):
    """Indices où le nombre métastable saute — les bifurcations du balayage."""
    return np.where(np.diff(counts) != 0)[0]


def main():
    print(f"Bifurcations du quotient — balayage de paramètre | n={N}, "
          f"{1 << N} états, seuil métastable |lambda|>{SLOW}\n")

    # A. ECA bruité — le contraste : paysage métastable plat
    print("A. ECA bruité — paramètre epsilon (le contraste)")
    for rule, regime in [(110, "complexe"), (30, "chaotique")]:
        counts = np.array([metastableCount(noisyEcaChain(rule, N, float(e)))
                           for e in EPSILONS])
        jumps = bifurcations(EPSILONS, counts)
        bif = (", ".join(f"eps~{EPSILONS[j + 1]:.3f}[{counts[j]}>{counts[j + 1]}]"
                          for j in jumps) if len(jumps) else "aucune")
        print(f"  règle {rule:>3} ({regime:>9}) : "
              f"{counts[0]} (eps=0) -> {counts[-1]} (eps={EPSILONS[-1]:.2f})"
              f" | bifurcations : {bif}")

    # B. verre de spin — le vrai paysage rugueux
    print("\nB. verre de spin — paramètre température (le vrai paysage)")
    print(f"  {'T':>6} {'#métastables (seeds 0,1,2)':>28} {'modes lents |l2..l4| (seed 0)':>32}")
    print("  " + "-" * 68)
    cascades = {seed: [] for seed in SEEDS}
    for temp in TEMPERATURES:
        row = []
        for seed in SEEDS:
            chain, _ = spinGlassChain(N, float(temp), seed=seed)
            count = metastableCount(chain)
            cascades[seed].append(count)
            row.append(count)
        chain0, _ = spinGlassChain(N, float(temp), seed=0)
        mags = np.sort(np.abs(chain0.transferSpectrum()[0]))[::-1]
        slow = " ".join(f"{m:.4f}" for m in mags[1:4])
        print(f"  {temp:6.2f} {str(row):>28} {slow:>32}")
    for seed in SEEDS:
        counts = np.array(cascades[seed])
        jumps = bifurcations(TEMPERATURES, counts)
        bif = ", ".join(f"T~{TEMPERATURES[j + 1]:.2f}[{counts[j]}>{counts[j + 1]}]"
                        for j in jumps)
        print(f"  seed {seed} : cascade {counts[0]} -> {counts[-1]} ; "
              f"bifurcations : {bif}")

    print(
        "\nLecture : balayer un paramètre DÉFORME le paysage métastable. Chaque\n"
        "saut du nombre d'ensembles métastables est une bifurcation exacte — un\n"
        "bassin se scinde, une barrière apparaît. L'ECA bruité reste plat (le\n"
        "§17 : pas de tour métastable) ; le verre de spin, lui, déploie une\n"
        "cascade de bifurcations en abaissant la température. Comme une\n"
        "hiérarchie, une cascade de bifurcations a besoin d'un paysage rugueux —\n"
        "et le régime énumérable la rend exactement mesurable, là où\n"
        "l'homéodynamique la dit centrale mais difficile à saisir."
    )


if __name__ == "__main__":
    main()
