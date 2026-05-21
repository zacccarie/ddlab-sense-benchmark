"""Phase 7 — rendu d'un état d'automate en image.

Le Pont 1 : un état de n cellules devient une image 2D. Un encodeur devra
re-découvrir l'état depuis les pixels — c'est la difficulté de perception qu'on
ajoute en quittant les systèmes jouets parfaitement observables."""

import numpy as np


def renderState(state, n, gridSide, cellSize):
    """Rend un état (n bits) en image 2D : grille gridSide × gridSide de cellules,
    chaque cellule un bloc plein cellSize × cellSize, valeur 0 ou 1."""
    side = gridSide * cellSize
    image = np.zeros((side, side), dtype=np.float32)
    for i in range(n):
        if (state >> i) & 1:
            row, col = divmod(i, gridSide)
            image[row * cellSize:(row + 1) * cellSize,
                  col * cellSize:(col + 1) * cellSize] = 1.0
    return image


def renderAll(n, cellSize=8):
    """Rend les 2^n états en images. n doit être un carré parfait (grille carrée)."""
    gridSide = int(round(n ** 0.5))
    if gridSide * gridSide != n:
        raise ValueError(f"n={n} n'est pas un carré parfait — grille impossible")
    size = 1 << n
    side = gridSide * cellSize
    images = np.empty((size, side, side), dtype=np.float32)
    for state in range(size):
        images[state] = renderState(state, n, gridSide, cellSize)
    return images
