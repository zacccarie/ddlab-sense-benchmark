"""Observables, horizon, contraction — quand le quotient récupère les bassins.

Le §15 a montré que le quotient relatif à la densité de l'état (horizon nul) ne
s'aligne pas sur les bassins. Étude systématique : plusieurs règles, plusieurs
seeds (RBN), deux observables, balayage de l'horizon — l'observable RETARDÉ
ell_H(s) = ell(succ^H(s)), l'observable du futur à H pas.

Deux boutons attendus : l'HORIZON fait monter l'AMI (regarder plus loin dans le
futur informe le quotient du destin) ; la RICHESSE de l'observable fixe le
plafond (combien d'attracteurs ell sait distinguer). Métriques contre la
partition en bassins : AMI (les égale-t-il ?) et homogénéité (les raffine-t-il
sans les traverser ?)."""

from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score

from ddlab_bench import computeBasinField
from ddlab_bench.basins import basinFieldFromSuccessor, iterateSuccessor
from ddlab_bench.rbn import RandomBooleanNetwork
from ddlab_bench.renormalization import bisimulationQuotient, densityLabel, windowLabel

N = 10
HORIZONS = [0, 4, 16, 64, 256]
RULES = [
    (250, "ordonné"), (232, "ordonné"),
    (110, "complexe"), (54, "complexe"), (108, "complexe"),
    (30, "chaotique"), (90, "chaotique"), (150, "chaotique"),
]


def sweep(succ, basins, label):
    """AMI par horizon, et homogénéité au plateau, du quotient relatif à ell_H."""
    amis, quotient = [], None
    for h in HORIZONS:
        quotient = bisimulationQuotient(succ, label[iterateSuccessor(succ, h)])
        amis.append(adjusted_mutual_info_score(basins, quotient))
    return amis, homogeneity_score(basins, quotient)  # dernier H = plateau


def table(label, observableName):
    head = "  {:>6} {:>10} {:>5}".format("règle", "régime", "maxT") \
        + "".join(f" {'H=' + str(h):>7}" for h in HORIZONS) + f" {'homog':>7}"
    print(f"{observableName} — AMI(quotient, bassins) par horizon ; homog. au plateau :")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        amis, hom = sweep(field.succ, field.attractorId, label)
        print(f"  {rule:>6} {regime:>10} {int(field.transient.max()):>5}"
              + "".join(f" {a:>7.3f}" for a in amis) + f" {hom:>7.3f}")


def main():
    print(f"Observable retardé ell_H = ell(succ^H) — récupération des bassins | n={N}\n")
    table(densityLabel(N), "Densité")
    print()
    table(windowLabel(N, 6), "Fenêtre 6 bits (observable plus riche)")

    print("\nMulti-seed — RBN K=2, densité (AMI à H=0 puis au plateau) :")
    for seed in range(5):
        rbn = RandomBooleanNetwork(N, 2, seed=seed)
        field = basinFieldFromSuccessor(rbn.successorArray(), N, seed)
        amis, _ = sweep(field.succ, field.attractorId, densityLabel(N))
        print(f"  seed {seed} : AMI {amis[0]:.3f} (H=0)  ->  {amis[-1]:.3f} (plateau)")

    print(
        "\nLecture :"
        "\n  L'HORIZON récupère les bassins — AMI(quotient relatif à ell_H, bassins)"
        "\n  CROÎT avec H puis plafonne ; l'horizon de plateau ~ la profondeur de"
        "\n  transitoire (la contraction de la dynamique) : passé le transitoire,"
        "\n  succ^H a posé chaque état sur son attracteur, ell_H se fige."
        "\n  La RICHESSE de l'observable contrôle la PROPRETÉ — la fenêtre 6 bits"
        "\n  atteint une homogénéité ~1 (le quotient raffine les bassins sans les"
        "\n  traverser) ; la densité, trop grossière, fusionne les attracteurs"
        "\n  qu'elle ne distingue pas (homogénéité ~0.5). Deux boutons : l'horizon"
        "\n  monte la récupération, la richesse de ell fixe sa propreté."
    )


if __name__ == "__main__":
    main()
