"""Balayage edge-of-chaos — renormalisation dynamique (§11-§12 de la note).

Pour un ensemble de règles ECA classées par Z de Wuensche, calcule la
complexité statistique (C_mu, entropie des blocs) du quotient de bisimulation,
le nombre d'états causaux, et le désaccord apparence/destin. Teste deux
choses : (1) la structure causale culmine-t-elle au régime complexe ?
(2) le ratio de mismatch perceptuel est-il universel à travers les régimes ?"""

from ddlab_bench.basins import computeBasinField
from ddlab_bench.classify import (
    perceptualBasinMismatch,
    randomBasinMismatch,
    regimeHint,
    zParameter,
)
from ddlab_bench.renormalization import (
    bisimulationQuotient,
    densityLabel,
    statisticalComplexity,
)

N = 12
RULES = [250, 232, 76, 110, 54, 73, 22, 90, 150, 30, 45]


def main():
    label = densityLabel(N)
    rows = []
    for rule in RULES:
        z = zParameter(rule)["z"]
        field = computeBasinField(rule, N)
        nBasins = len(field.attractors)
        cmuBasin = statisticalComplexity(field.attractorId)
        quotient = bisimulationQuotient(field.succ, label)
        nCausal = int(quotient.max()) + 1
        cmuCausal = statisticalComplexity(quotient)
        mismatch = perceptualBasinMismatch(field)
        random = randomBasinMismatch(field)
        ratio = mismatch / random if random > 0 else float("nan")
        rows.append(
            (z, rule, regimeHint(z), nBasins, cmuBasin, nCausal, cmuCausal, mismatch, ratio)
        )

    rows.sort()
    print(f"ECA, n = {N} — balayage par Z de Wuensche\n")
    print(
        f"{'Z':>5} {'règle':>6} {'régime':>11} {'#bassins':>9} {'Cμ(bassin)':>11} "
        f"{'#causal':>8} {'Cμ(causal)':>11} {'mismatch':>9} {'ratio':>6}"
    )
    for z, rule, regime, nBasins, cmuBasin, nCausal, cmuCausal, mismatch, ratio in rows:
        print(
            f"{z:5.2f} {rule:6d} {regime:>11} {nBasins:9d} {cmuBasin:11.3f} "
            f"{nCausal:8d} {cmuCausal:11.3f} {mismatch:9.4f} {ratio:6.2f}"
        )

    complexRows = [r for r in rows if r[2] == "complexe"]
    if complexRows:
        peakCmu = max(r[4] for r in rows)
        peakIsComplex = max(rows, key=lambda r: r[4])[2] == "complexe"
        print(
            f"\nCμ(bassin) max = {peakCmu:.3f} — au régime complexe : {peakIsComplex}"
        )
    print(f"ratio min/max = {min(r[8] for r in rows):.2f} / {max(r[8] for r in rows):.2f}")


if __name__ == "__main__":
    main()
