"""Information Bottleneck agglomératif — la hiérarchie de renormalisation émergente.

Le §11-§14 CHOISIT l'observable de base ℓ puis calcule le quotient. Ici on
laisse la hiérarchie ÉMERGER. Étant donné une variable pertinente Y (le futur
de l'état), l'IB agglomératif (Slonim & Tishby 2000) part de chaque état dans
son propre bloc et fusionne gloutonnement la paire dont la fusion perd le moins
d'information pertinente I(Z;Y). La suite des fusions est un dendrogramme — la
hiérarchie Q_0 ⊃ Q_1 ⊃ ... Aucun ℓ n'est imposé : ℓ émerge comme la projection
IB-pertinente, à chaque échelle.

L'espace d'états étant énumérable, p(x,y) est exacte — I(Z;Y) est calculée
EXACTEMENT, sans estimateur variationnel."""

import numpy as np

_EPS = 1e-12


def _klRows(p, q):
    """D_KL(p || q) bit, ligne par ligne ; p, q de même forme (k, |Y|).
    Convention 0 log 0 = 0."""
    safe = p > 0
    ratio = np.where(safe, p / np.maximum(q, _EPS), 1.0)
    return np.sum(np.where(safe, p * np.log2(ratio), 0.0), axis=-1)


def mergeCost(pZi, pYi, pZj, pYj):
    """Perte d'information pertinente d'une fusion (Slonim & Tishby) :
    Delta I(Z;Y) = (p_i + p_j) * JS_Pi(p(y|z_i), p(y|z_j)),
    JS la divergence de Jensen-Shannon pondérée par les masses."""
    pZ = pZi + pZj
    wi, wj = pZi / pZ, pZj / pZ
    pYi, pYj = np.asarray(pYi, float), np.asarray(pYj, float)
    pYbar = wi * pYi + wj * pYj
    js = wi * _klRows(pYi[None], pYbar[None])[0] + wj * _klRows(pYj[None], pYbar[None])[0]
    return float(pZ * js)


def _costRow(i, pZ, pYZ, alive):
    """Coût de fusion du bloc i contre tous les blocs (vectorisé) ; les blocs
    morts et i lui-même sont mis à l'infini."""
    pZsum = pZ[i] + pZ
    wi = pZ[i] / pZsum
    wj = pZ / pZsum
    pYbar = wi[:, None] * pYZ[i][None, :] + wj[:, None] * pYZ
    klI = _klRows(np.broadcast_to(pYZ[i], pYZ.shape), pYbar)
    klJ = _klRows(pYZ, pYbar)
    cost = pZsum * (wi * klI + wj * klJ)
    cost[~alive] = np.inf
    cost[i] = np.inf
    return cost


def agglomerativeIB(pYgivenX, pX=None):
    """IB agglomératif : fusionne les états jusqu'à un seul bloc.

    pYgivenX : (sizeX, sizeY) — p(y | x), une ligne par état.
    pX       : (sizeX,) prior sur les états ; uniforme par défaut.

    Retourne (merges, assignments) :
      merges[k]      = (cost, infoRemaining) de la k-ième fusion, du plus fin
                       au plus grossier ; cost = Delta I(Z;Y) >= 0 ;
      assignments[k] = partition à (sizeX - k) blocs ; assignments[0] est
                       l'identité, assignments[-1] le bloc unique."""
    pYgivenX = np.asarray(pYgivenX, dtype=np.float64)
    sizeX = len(pYgivenX)
    if pX is None:
        pX = np.full(sizeX, 1.0 / sizeX)
    else:
        pX = np.asarray(pX, dtype=np.float64)
        pX = pX / pX.sum()

    pZ = pX.copy()
    rowSums = np.maximum(pYgivenX.sum(axis=1, keepdims=True), _EPS)
    pYZ = pYgivenX / rowSums
    members = [{x} for x in range(sizeX)]
    alive = np.ones(sizeX, dtype=bool)

    pY = (pZ[:, None] * pYZ).sum(axis=0)
    info = float(np.sum(pZ * _klRows(pYZ, np.broadcast_to(pY, pYZ.shape))))

    cost = np.full((sizeX, sizeX), np.inf)
    for i in range(sizeX):
        cost[i] = _costRow(i, pZ, pYZ, alive)

    assignments = [np.arange(sizeX, dtype=np.int64)]
    merges = []
    while alive.sum() > 1:
        i, j = divmod(int(np.argmin(cost)), sizeX)
        c = float(cost[i, j])
        # fusion du bloc j dans le bloc i
        newPZ = pZ[i] + pZ[j]
        pYZ[i] = (pZ[i] * pYZ[i] + pZ[j] * pYZ[j]) / newPZ
        pZ[i] = newPZ
        members[i] |= members[j]
        alive[j] = False
        info -= c
        cost[j, :] = np.inf
        cost[:, j] = np.inf
        newRow = _costRow(i, pZ, pYZ, alive)
        cost[i, :] = newRow
        cost[:, i] = newRow
        merges.append((c, info))

        contig = np.empty(sizeX, dtype=np.int64)
        for label, blk in enumerate(np.where(alive)[0]):
            for x in members[blk]:
                contig[x] = label
        assignments.append(contig)
    return merges, assignments


def partitionAt(assignments, numBlocks):
    """Partition du dendrogramme à `numBlocks` blocs."""
    sizeX = len(assignments)
    if not 1 <= numBlocks <= sizeX:
        raise ValueError(f"numBlocks={numBlocks} hors borne [1, {sizeX}]")
    return assignments[sizeX - numBlocks]
