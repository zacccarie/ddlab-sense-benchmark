"""Substrat continu — bassins de Newton, frontières fractales.

L'algorithme de Newton modifié  z_+ = z + lambda*(1 - z^3)/(3 z^2)  résout
z^3 = 1 : trois racines, trois bassins d'attraction, frontière fractale
(ensemble de Julia). Référence : Sahari & Djellit, ESAIM Proc. 20 (2007).

Substrat CONTINU à vérité-terrain EXACTE : le bassin d'un point se calcule en
itérant la carte jusqu'à une racine — sans énumérer ni discrétiser l'espace
d'états. Contrairement au CML du §13 (dont la quantification détruisait le
système), la grille ne fait qu'échantillonner OÙ évaluer le bassin ; chaque
étiquette vient de la carte continue exacte.

lambda joue le rôle du Z de Wuensche : la dimension fractale de la frontière
croît avec lambda. La frontière fractale est le §9 rendu exact — deux points
arbitrairement proches peuvent appartenir à des bassins différents, à toute
échelle."""

import numpy as np

# les trois racines cubiques de l'unité — les attracteurs
ROOTS = np.array([1.0 + 0j, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)])


def newtonStep(z, lam):
    """Un pas de Newton modifié pour z^3 = 1 : z + lambda*(1 - z^3)/(3 z^2)."""
    return z + lam * (1.0 - z ** 3) / (3.0 * z ** 2)


def basinLabel(z, lam, maxIter=200, tol=1e-6):
    """Bassin (0/1/2) vers lequel z converge ; -1 si pas de convergence.

    Vérité-terrain exacte : itère la carte continue jusqu'à approcher une
    racine. `z` est un array complexe (vectorisé sur tout l'array)."""
    z = np.asarray(z, dtype=np.complex128).copy()
    label = np.full(z.shape, -1, dtype=np.int64)
    active = np.ones(z.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(maxIter):
            stepped = newtonStep(z, lam)
            moving = active & np.isfinite(stepped)
            z = np.where(moving, stepped, z)
            for r, root in enumerate(ROOTS):
                hit = active & (np.abs(z - root) < tol)
                label[hit] = r
                active[hit] = False
            if not active.any():
                break
    return label


def basinGrid(lam, resolution, extent=2.0, maxIter=200, tol=1e-6):
    """Grille `resolution` x `resolution` du carré [-extent, extent]^2 du plan
    complexe, chaque cellule étiquetée par son bassin exact."""
    axis = np.linspace(-extent, extent, resolution)
    real, imag = np.meshgrid(axis, axis)
    return basinLabel(real + 1j * imag, lam, maxIter=maxIter, tol=tol)


def perceptualMismatch(labels):
    """Fraction des paires de cellules voisines (4-voisinage) de bassins
    différents — le désaccord apparence/destin du §9, sur la grille de Newton.

    Élevé => bassins perceptuellement éclatés. Sur une frontière fractale, ne
    décroît pas quand la résolution augmente : l'éclatement est sans échelle."""
    labels = np.asarray(labels)
    valid = labels >= 0
    diff = total = 0
    for sliceA, sliceB in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        a, b = labels[sliceA], labels[sliceB]
        pairValid = valid[sliceA] & valid[sliceB]
        diff += int(np.sum(pairValid & (a != b)))
        total += int(np.sum(pairValid))
    return diff / total if total else 0.0


def boxCountingDimension(labels):
    """Dimension fractale (box-counting) de la frontière des bassins.

    Frontière = cellules ayant un voisin (4-voisinage) de bassin différent. On
    compte le nombre N(s) de boîtes de côté s qui la contiennent, pour
    s = 1, 2, 4, ... ; la pente de log N(s) vs log(1/s) estime la dimension
    (méthode du §6 du PDF Sahari-Djellit)."""
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    vert = labels[:-1, :] != labels[1:, :]
    boundary[:-1, :] |= vert
    boundary[1:, :] |= vert
    horiz = labels[:, :-1] != labels[:, 1:]
    boundary[:, :-1] |= horiz
    boundary[:, 1:] |= horiz
    if not boundary.any():
        return 0.0

    res = labels.shape[0]
    sizes, counts = [], []
    s = 1
    while s <= res // 2:
        trimmed = res - (res % s)
        blocks = boundary[:trimmed, :trimmed].reshape(
            trimmed // s, s, trimmed // s, s
        )
        sizes.append(s)
        counts.append(int(blocks.any(axis=(1, 3)).sum()))
        s *= 2
    sizes = np.array(sizes, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)
    return float(np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)[0])


def boundaryBasinCount(labels, window=3):
    """Nombre moyen de bassins DISTINCTS dans une fenêtre autour de chaque
    cellule-frontière.

    Cellule-frontière = ayant un voisin (4-voisinage) de bassin différent. Sur
    la frontière de Julia de Newton les trois bassins se touchent (PDF
    Sahari-Djellit : J = dA1 = dA2 = dA3) ; le compte tend vers 3 et reste
    stable à toute résolution — alors que le mismatch GLOBAL s'annule (la
    frontière fractale est de mesure nulle). C'est le « perceptuellement
    éclaté » du §9 rendu exact et sans échelle : un mismatch global qui
    s'évanouit mais un éclatement local invariant d'échelle."""
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    vert = labels[:-1, :] != labels[1:, :]
    boundary[:-1, :] |= vert
    boundary[1:, :] |= vert
    horiz = labels[:, :-1] != labels[:, 1:]
    boundary[:, :-1] |= horiz
    boundary[:, 1:] |= horiz

    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return 0.0
    half, res = window // 2, labels.shape[0]
    counts = []
    for r, c in zip(rows, cols):
        block = labels[max(0, r - half):min(res, r + half + 1),
                       max(0, c - half):min(res, c + half + 1)]
        counts.append(len(set(block[block >= 0].tolist())))
    return float(np.mean(counts))
