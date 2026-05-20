"""Construit les bundles de benchmark (champ de bassins + trajectoires) pour un
panel de règles couvrant le spectre ordre -> complexe -> chaos."""

import numpy as np

from ddlab_bench import computeBasinField
from ddlab_bench.artifacts import buildBundle

N = 14
RULES = [110, 54, 30, 250]
TRAJ_CONFIG = {"count": 4000, "length": 24, "holdoutFraction": 0.3}


def main():
    rng = np.random.default_rng(7)
    for rule in RULES:
        field = computeBasinField(rule, N)
        if len(field.attractors) < 2:
            print(f"règle {rule}: bassin unique, ignorée")
            continue
        path = buildBundle(field, TRAJ_CONFIG, rng, f"data/rule{rule}_n{N}.npz")
        print(f"règle {rule:>3} -> {path} | {len(field.attractors)} bassins")


if __name__ == "__main__":
    main()
