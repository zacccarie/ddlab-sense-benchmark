"""ECA bruité — la tour métastable d'un système naturel.

§16 a montré une hiérarchie PLANTÉE, profonde, récupérée. Ici la hiérarchie
n'est plus plantée : un ECA bruité — la règle déterministe avec proba 1-eps, un
voisin perceptuel avec proba eps. Les bassins déterministes deviennent des
ensembles métastables ; eps -> 0 retrouve l'automate du §1-§15.

Balayage de eps : le spectre de transfert (combien de modes lents = ensembles
métastables ; temps d'échappement) et l'IB agglomératif (récupère-t-il les
bassins comme niveau métastable ?)."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ddlab_bench import computeBasinField
from ddlab_bench.agglomerative_ib import agglomerativeIB, partitionAt
from ddlab_bench.markov import noisyEcaChain

N = 8
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]
EPS_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4]
HORIZON = 30  # >> relaxation intra-bassin, << temps d'échappement aux petits eps


def slowModeCount(mags, threshold=0.5):
    """Nombre de modes lents : valeurs propres de module > seuil (temps de
    relaxation supérieur à ~1.4 pas)."""
    return int(np.sum(mags > threshold))


def main():
    print(f"ECA bruité — tour métastable | n={N}, horizon IB={HORIZON}\n")
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        nBasins = len(field.attractors)
        print(f"règle {rule} ({regime}) — {nBasins} bassins vrais")
        print(f"  {'eps':>6} {'#modes lents':>13} {'t échappement':>14} "
              f"{'AMI(IB @ bassins)':>18}")
        for eps in EPS_VALUES:
            chain = noisyEcaChain(rule, N, eps)
            mags = np.abs(chain.transferSpectrum()[0])
            nSlow = slowModeCount(mags)
            tEscape = chain.impliedTimescales()[1]
            _, assignments = agglomerativeIB(chain.multiStep(HORIZON))
            emerged = partitionAt(assignments, nBasins)
            ami = adjusted_mutual_info_score(field.attractorId, emerged)
            print(f"  {eps:>6.2f} {nSlow:>13} {tEscape:>14.1f} {ami:>18.3f}")
        print()
    print(
        "Lecture — honnête :"
        "\n  t échappement ~ 1/eps : le bruit fixe bien l'échelle d'échappement."
        "\n  Mais l'IB ne récupère les bassins QUE pour la règle 250 (ordonné,"
        "\n  attracteurs courts). Règles 110 / 30 : AMI médiocre voire nul."
        "\n  Deux obstructions structurelles :"
        "\n   (1) attracteurs cycliques — la phase du cycle est une coordonnée"
        "\n       interne LENTE (mélange ~ p/eps, comparable au temps"
        "\n       d'échappement) : pas de séparation d'échelles interne/externe,"
        "\n       les bassins ne sont pas des ensembles métastables nets."
        "\n   (2) §9 — le bruit perceptuel fuit à travers les frontières de"
        "\n       bassin perceptuellement éclatées ; les bassins ne sont pas"
        "\n       robustes au bruit perceptuel."
        "\n  Le noisy ECA n'hérite PAS d'une tour propre. La tour profonde"
        "\n  récupérable reste la chaîne plantée du §16."
    )


if __name__ == "__main__":
    main()
