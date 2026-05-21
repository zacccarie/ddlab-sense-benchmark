"""CML discrétisé — première application des protocoles au substrat continu.

Le pipeline ddlab-sense-benchmark exige un espace d'états énumérable. Un Coupled
Map Lattice est continu ; on le discrétise en m niveaux par cellule -> m^n états
énumérables, vérité-terrain exacte préservée.

Balaye le plan de contrôle (r, eps) — qui remplace le Z de Wuensche. Pour chaque
régime : champ de bassins exact, désaccord apparence/destin (maryBasinMismatch,
le test du §9 porté en m-aire), quotient de bisimulation relatif au champ moyen
et sa complexité statistique C_mu (§11-§12).

Question : le résultat « bassins perceptuellement éclatés » (ratio ~ 1, obtenu
sur ECA) survit-il à un substrat doté de cohérence spatiale (structures, ondes) ?"""

from ddlab_bench import basinFieldFromSuccessor
from ddlab_bench.classify import maryBasinMismatch, randomBasinMismatch, structuralSummary
from ddlab_bench.cml import CoupledMapLattice
from ddlab_bench.renormalization import bisimulationQuotient, statisticalComplexity

N = 7
M = 5
# (r, eps, régime) — points du diagramme de phase de Kaneko
GRID = [
    (3.6, 0.10, "periodique faible-couplage"),
    (3.9, 0.10, "chaos local faible-couplage"),
    (3.9, 0.30, "bord du chaos spatio-temporel"),
    (4.0, 0.50, "chaos fort-couplage"),
]


def runPoint(r, epsilon):
    cml = CoupledMapLattice(N, M, r, epsilon)
    succ = cml.successorArray()
    field = basinFieldFromSuccessor(succ, N, f"cml-r{r}-e{epsilon}")
    summary = structuralSummary(field)
    mismatch = maryBasinMismatch(field, M)
    baseline = randomBasinMismatch(field)
    quotient = bisimulationQuotient(succ, cml.densityLabel())
    return {
        "basins": len(field.attractors),
        "goe": summary["goeFraction"],
        "mismatch": mismatch,
        "baseline": baseline,
        "ratio": mismatch / baseline if baseline > 0 else 0.0,
        "quotientBlocks": int(quotient.max()) + 1,
        "cmu": statisticalComplexity(quotient),
    }


def main():
    size = M ** N
    print(f"CML discrétisé — n={N} cellules, m={M} niveaux, {size} états énumérés\n")
    header = (
        f"{'regime':>30} {'r':>5} {'eps':>5} {'#bas':>5} {'GoE%':>6} "
        f"{'mismatch':>9} {'hasard':>8} {'ratio':>7} {'#quot':>6} {'Cmu':>6}"
    )
    print(header)
    print("-" * len(header))
    for r, epsilon, regime in GRID:
        p = runPoint(r, epsilon)
        print(
            f"{regime:>30} {r:>5.2f} {epsilon:>5.2f} {p['basins']:>5} "
            f"{p['goe'] * 100:>5.1f}% {p['mismatch']:>9.3f} {p['baseline']:>8.3f} "
            f"{p['ratio']:>7.3f} {p['quotientBlocks']:>6} {p['cmu']:>6.2f}"
        )
    print(
        "\nLecture — ratio mismatch / hasard :"
        "\n  ~ 1  => bassins perceptuellement éclatés, comme sur les ECA — la barrière"
        "\n          perceptuelle du §9 n'est pas propre aux automates cellulaires."
        "\n  << 1 => la cohérence spatiale du CML rend les bassins perceptuellement"
        "\n          contigus — un objectif perceptuel pourrait y réussir."
        "\nC_mu = complexité statistique du quotient de bisimulation (relatif au champ"
        "\nmoyen) ; attendue maximale au bord du chaos (§12)."
    )


if __name__ == "__main__":
    main()
