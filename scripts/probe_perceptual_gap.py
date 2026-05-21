"""Phase 11 (axe C) — la barrière théorique : apparence contre destin.

Pourquoi les 5 objectifs appris (Phases 7-10b) échouent à découvrir le quotient
dynamique. Thèse : dans un système complexe, le bassin est une fonction
NON-LOCALE de l'observation — un changement perceptuel minime (1 bit) peut
changer le bassin. Les objectifs basés sur la proximité d'observation
(reconstruction, prédiction perceptuelle, contrastif d'instance) rapprochent les
entrées perceptuellement proches : biais de localité. Ils sont donc
structurellement opposés au quotient.

Mesure directe : parmi les voisins à 1 bit d'un état, quelle fraction change de
bassin (mismatch) ? Comparé au mismatch d'une assignation ALÉATOIRE de mêmes
tailles. Ratio ~ 1 => les bassins sont perceptuellement éclatés (aussi dispersés
que le hasard) => aucun objectif perceptuel ne peut les récupérer : échec
structurel, pas un échec d'entraînement."""

from ddlab_bench import computeBasinField
from ddlab_bench.classify import perceptualBasinMismatch, randomBasinMismatch, zParameter

N = 9
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]


def main():
    print(f"axe C — désaccord apparence / destin | n={N}\n")
    header = (f"{'règle':>7} {'Z':>5} {'régime':>10} {'#bassins':>9} "
              f"{'mismatch':>10} {'hasard':>9} {'ratio':>7}")
    print(header)
    print("-" * len(header))
    for rule, regime in RULES:
        field = computeBasinField(rule, N)
        z = zParameter(rule)["z"]
        actual = perceptualBasinMismatch(field)
        baseline = randomBasinMismatch(field)
        ratio = actual / baseline if baseline > 0 else 0.0
        print(f"{rule:>7} {z:>5.2f} {regime:>10} {len(field.attractors):>9} "
              f"{actual:>10.3f} {baseline:>9.3f} {ratio:>7.3f}")
    print(
        "\nLecture — ratio mismatch / hasard :"
        "\n  ~ 1  => bassins perceptuellement éclatés, aussi dispersés qu'au hasard"
        "\n          => la proximité d'observation ne porte aucune info de bassin"
        "\n          => tout objectif perceptuel est structurellement incapable de"
        "\n             récupérer le quotient. Les 5 échecs ne sont pas des échecs"
        "\n             d'entraînement — c'est une barrière."
        "\n  << 1 => bassins perceptuellement contigus => un objectif perceptuel"
        "\n          pourrait marcher (la thèse serait fausse)."
    )


if __name__ == "__main__":
    main()
