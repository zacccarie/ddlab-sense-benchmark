"""Tests — substrat stochastique : chaînes de Markov énumérables."""

import numpy as np

from ddlab_bench.eca import buildSuccessorArray
from ddlab_bench.markov import (
    MarkovChain,
    hierarchicalChain,
    hierarchicalGraphChain,
    noisyEcaChain,
    spinGlassChain,
)


def test_rows_normalized():
    chain = MarkovChain(np.array([[2.0, 2.0], [1.0, 3.0]]))
    assert np.allclose(chain.P.sum(axis=1), 1.0)


def test_multi_step_stochastic():
    chain = noisyEcaChain(110, 6, epsilon=0.1)
    assert np.allclose(chain.multiStep(1), chain.P)
    assert np.allclose(chain.multiStep(3).sum(axis=1), 1.0)


def test_stationary_is_invariant():
    chain = noisyEcaChain(110, 6, epsilon=0.1)
    pi = chain.stationary()
    assert abs(pi.sum() - 1.0) < 1e-9
    assert np.allclose(pi @ chain.P, pi, atol=1e-8)


def test_transfer_spectrum_sorted_with_unit_eigenvalue():
    chain = noisyEcaChain(30, 6, epsilon=0.15)
    values, _ = chain.transferSpectrum()
    mags = np.abs(values)
    assert mags[0] > 0.999                 # mode trivial, valeur propre 1
    assert np.all(np.diff(mags) <= 1e-9)   # trié par module décroissant


def test_noisy_eca_zero_noise_is_deterministic():
    chain = noisyEcaChain(110, 6, epsilon=0.0)
    succ = buildSuccessorArray(110, 6)
    for s in (0, 1, 17, 63):
        assert chain.P[s, int(succ[s])] == 1.0


def test_hierarchical_chain_size_and_nesting():
    chain, levels = hierarchicalChain(2, 3, 4, [1.0, 0.2, 0.04, 0.008], seed=0)
    assert chain.size == 2 ** 3 * 4
    assert np.allclose(chain.P.sum(axis=1), 1.0)
    assert len(levels) == 3
    # chaque niveau raffine le suivant : un bloc de d est inclus dans un bloc de d+1
    for d in range(2):
        for block in range(int(levels[d].max()) + 1):
            members = np.where(levels[d] == block)[0]
            assert len(np.unique(levels[d + 1][members])) == 1
    assert int(levels[2].max()) + 1 == 2   # niveau le plus grossier = branching blocs


def test_hierarchical_rates_validation():
    import pytest
    with pytest.raises(ValueError):
        hierarchicalChain(2, 3, 4, [1.0, 0.2, 0.04], seed=0)  # depth+1 attendu


def test_hierarchical_graph_chain():
    chain, levels = hierarchicalGraphChain(2, 3, 4, [0.9, 0.3, 0.06, 0.01], seed=0)
    assert chain.size == 2 ** 3 * 4
    assert np.allclose(chain.P.sum(axis=1), 1.0)  # auto-boucles -> aucun nœud isolé
    assert len(levels) == 3
    for d in range(2):
        for block in range(int(levels[d].max()) + 1):
            members = np.where(levels[d] == block)[0]
            assert len(np.unique(levels[d + 1][members])) == 1


def test_spin_glass_energy_flip_symmetry():
    # E(-s) = E(s) : retourner tous les spins préserve l'énergie
    _, energy = spinGlassChain(8, temperature=1.0, seed=0)
    assert np.allclose(energy, energy[::-1])


def test_spin_glass_boltzmann_stationary():
    # la dynamique de Glauber-Metropolis a la distribution de Boltzmann pour
    # mesure stationnaire (bilan détaillé)
    chain, energy = spinGlassChain(8, temperature=1.0, seed=0)
    boltzmann = np.exp(-energy / 1.0)
    boltzmann /= boltzmann.sum()
    assert np.allclose(chain.stationary(), boltzmann, atol=1e-6)
