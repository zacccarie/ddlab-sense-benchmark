"""Atlas des quantités — du langage à la théorie (§26, le tableau).

Le §25 a clos l'inventaire : huit mots décrivent le quotient causal —
recoverability, éclatement, hiérarchie, séparation spectrale, invariance
d'échelle, métastabilité, bifurcation, hystérésis. Mais huit mots co-localisés
font un LANGAGE, pas une théorie. La théorie commence quand les mots se relient
— quand deux d'entre eux se révèlent une seule quantité.

Ce probe construit le tableau : une LIGNE par système exactement soluble (les
256 ECA, n=9), une COLONNE par quantité exacte. La structure du tableau — sa
matrice de corrélation, son rang PCA — dit si le langage est une théorie en
attente (rang bas, les mots fusionnent) ou des axes genuinement indépendants.

Sous-tableau GRATUIT : colonnes structurelles seules, aucun entraînement. Les
colonnes de recoverability (entraîner les objectifs) sont la phase 2.

Colonnes spectrales : sur l'encastrement bruité (epsilon=0.02), un spectre de
transfert déterministe étant dégénéré sur le cercle unité."""

import numpy as np
from sklearn.decomposition import PCA

from ddlab_bench import buildSuccessorArray
from ddlab_bench.markov import noisyEcaChain
from ddlab_bench.renormalization import (
    bisimulationQuotient,
    densityLabel,
    statisticalComplexity,
)

N = 9
EPS_LOW, EPS_HIGH = 0.02, 0.10
SLOW = 0.99
MAX_MODES = 16
COLUMNS = ["nBlocks", "shatter", "cmu", "specGap", "hierDepth", "metaCount", "bifSens"]


def shattering(quotient, n):
    """Éclatement perceptuel : fraction des paires de Hamming-voisins traversant
    une frontière du quotient, normalisée par une partition aléatoire de mêmes
    tailles de blocs. ~1 = aussi dispersé qu'un partage aléatoire (quotient
    perceptuellement éclaté, §9) ; <1 = blocs contigus en espace d'observation."""
    size = len(quotient)
    if int(quotient.max()) == 0:
        return 0.0
    states = np.arange(size)
    cross = sum(int(np.sum(quotient != quotient[states ^ (1 << b)])) for b in range(n))
    observed = cross / (size * n)
    _, counts = np.unique(quotient, return_counts=True)
    randomCross = 1.0 - float(np.sum((counts / size) ** 2))
    return observed / randomCross if randomCross > 1e-12 else 0.0


def metastableCount(mags):
    """Nombre de modes lents — valeurs propres de transfert de module > SLOW."""
    return int(np.sum(mags > SLOW))


def gapCount(mags):
    """Profondeur de hiérarchie ~ nombre d'écarts spectraux nets (ratio > 1.5)
    parmi les modes lents."""
    slow = mags[:MAX_MODES]
    ratios = slow[:-1] / np.maximum(slow[1:], 1e-12)
    return int(np.sum(ratios > 1.5))


def specGap(mags):
    """Écart spectral dominant parmi les modes lents."""
    slow = mags[:MAX_MODES]
    return float((slow[:-1] - slow[1:]).max())


def buildTable():
    """Tableau exact (256 ECA) x (7 quantités structurelles)."""
    label = densityLabel(N)
    rows = []
    for rule in range(256):
        succ = buildSuccessorArray(rule, N)
        quotient = bisimulationQuotient(succ, label)
        magsLo = np.sort(np.abs(
            noisyEcaChain(rule, N, EPS_LOW).transferSpectrum()[0]))[::-1]
        magsHi = np.sort(np.abs(
            noisyEcaChain(rule, N, EPS_HIGH).transferSpectrum()[0]))[::-1]
        metaLo = metastableCount(magsLo)
        rows.append([
            int(quotient.max()) + 1,            # nBlocks
            shattering(quotient, N),             # shatter
            statisticalComplexity(quotient),     # cmu
            specGap(magsLo),                     # specGap
            gapCount(magsLo),                    # hierDepth
            metaLo,                              # metaCount
            abs(metaLo - metastableCount(magsHi)),  # bifSens
        ])
    return np.array(rows, dtype=np.float64)


def main():
    print(f"Atlas des quantités — {256} ECA, n={N} | tableau exact, sans entraînement\n")
    table = buildTable()

    print("  quantité     min      moy      max     écart-type")
    print("  " + "-" * 52)
    for j, name in enumerate(COLUMNS):
        col = table[:, j]
        print(f"  {name:>10} {col.min():8.3f} {col.mean():8.3f} "
              f"{col.max():8.3f} {col.std():10.3f}")

    # colonnes à variance non nulle -> standardisation
    keep = [j for j in range(len(COLUMNS)) if table[:, j].std() > 1e-9]
    names = [COLUMNS[j] for j in keep]
    sub = table[:, keep]
    standardized = (sub - sub.mean(axis=0)) / sub.std(axis=0)

    corr = np.corrcoef(standardized, rowvar=False)
    print("\n  corrélations |r| > 0.6 — mots candidats à fusionner :")
    found = False
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if abs(corr[a, b]) > 0.6:
                print(f"    {names[a]:>10} ~ {names[b]:<10} r = {corr[a, b]:+.3f}")
                found = True
    if not found:
        print("    aucune — les colonnes sont mutuellement faibles")

    pca = PCA().fit(standardized)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    rank90 = int(np.searchsorted(cumulative, 0.90)) + 1
    print(f"\n  PCA ({len(names)} colonnes standardisées) :")
    print("    variance par composante : "
          + " ".join(f"{r * 100:.0f}%" for r in ratio))
    print(f"    variance cumulée 90% atteinte à K = {rank90} composante(s)"
          f" sur {len(names)}")

    print(
        f"\nLecture : {len(names)} quantités exactes mesurées, dimensionnalité"
        f"\neffective K={rank90}. K bas -> les mots fusionnent, le langage est une"
        "\nthéorie en attente. K proche du nombre de colonnes -> autant d'axes"
        "\nindépendants, langage genuinement multi-dimensionnel. Sous-tableau"
        "\ngratuit ; les colonnes de recoverability (phase 2, entraînement des"
        "\nobjectifs) restent à ajouter — elles diront si l'apprenabilité se"
        "\nréduit elle aussi à ces axes structurels."
    )


if __name__ == "__main__":
    main()
