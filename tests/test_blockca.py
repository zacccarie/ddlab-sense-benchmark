"""Tests — CA par blocs à deux échelles (§23, le « gem »). Propriétés exactes :
produit semi-direct (couche grossière autonome) et tour de renormalisation non
plantée, vérifiées contre l'énumération exhaustive."""

import numpy as np

from ddlab_bench import bisimulationQuotient
from ddlab_bench.blockca import (
    BlockCellularAutomaton,
    coarseDensityLabel,
    coarseProjection,
    computeBlockCaField,
)
from ddlab_bench.renormalization import quotientGraph, refinePartition

# config canonique : 3 blocs de 3 cellules — couche grossière chaotique (45),
# couche fine pilotée par deux règles distinctes (30 si grossier=0, 110 sinon).
CONFIG = (3, 3, 45, 30, 110)


def test_successor_array_is_functional():
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    assert succ.shape == (1 << 9,)
    assert succ.min() >= 0 and succ.max() < (1 << 9)


def test_block_size_one_is_rejected():
    # blockSize < 2 -> aucune cellule fine -> pas un système à deux échelles
    import pytest

    with pytest.raises(ValueError):
        BlockCellularAutomaton(3, 1, 45, 30, 110)


def test_coarse_layer_is_autonomous():
    # produit semi-direct : la couche grossière de succ(x) ne dépend que de la
    # couche grossière de x.
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    proj = coarseProjection(3, 3)
    nextProj = proj[succ]
    for coarseValue in np.unique(proj):
        members = np.where(proj == coarseValue)[0]
        assert len(np.unique(nextProj[members])) == 1


def test_fine_layer_is_driven_not_constant():
    # la couche fine évolue : le système n'est pas un ECA grossier déguisé.
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    proj = coarseProjection(3, 3)
    sameCoarse = np.where(proj == proj[0])[0]
    assert len(sameCoarse) > 1
    assert len(np.unique(succ[sameCoarse])) > 1


def test_coarse_projection_is_exact_bisimulation_quotient():
    # le résultat clé du §23 : le quotient de bisimulation relatif à
    # l'observable grossier EST la partition par bits grossiers — 2^nBlocks
    # blocs, non plantée mais produite par la structure d'échelle de la règle.
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    proj = coarseProjection(3, 3)
    q = bisimulationQuotient(succ, proj)
    assert int(q.max()) + 1 == 1 << 3
    _, projRelabeled = np.unique(proj, return_inverse=True)
    assert np.array_equal(q, projRelabeled)


def test_coarse_quotient_is_dynamics_compatible():
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    q = bisimulationQuotient(succ, coarseProjection(3, 3))
    quotientSucc, _ = quotientGraph(succ, q)
    assert np.array_equal(q[succ], quotientSucc[q])
    # point fixe de l'opérateur de renormalisation R
    assert int(refinePartition(succ, q).max()) == int(q.max())


def test_renormalization_tower_has_three_strict_levels():
    # état complet  >  partition grossière  >  quotient densité-grossière
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    full = 1 << 9
    coarse = int(bisimulationQuotient(succ, coarseProjection(3, 3)).max()) + 1
    density = int(bisimulationQuotient(succ, coarseDensityLabel(3, 3)).max()) + 1
    assert full > coarse > density > 1


def test_coarse_quotient_refines_density_quotient():
    # la tour est emboîtée : chaque bloc grossier vit dans un seul bloc densité.
    ca = BlockCellularAutomaton(*CONFIG)
    succ = ca.successorArray()
    coarseQ = bisimulationQuotient(succ, coarseProjection(3, 3))
    densityQ = bisimulationQuotient(succ, coarseDensityLabel(3, 3))
    for block in np.unique(coarseQ):
        members = np.where(coarseQ == block)[0]
        assert len(np.unique(densityQ[members])) == 1


def test_compute_field_matches_class():
    field = computeBlockCaField(*CONFIG)
    ca = BlockCellularAutomaton(*CONFIG)
    assert np.array_equal(field.succ, ca.successorArray())
    assert field.n == 9
    # graphe fonctionnel : somme des in-degrés = nombre d'états
    assert sum(len(pre) for pre in field.preimages) == (1 << 9)
