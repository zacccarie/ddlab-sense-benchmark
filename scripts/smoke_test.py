"""Smoke test Phase 0 + 1.

Phase 0: champ de bassins exhaustif + classification (Z, structure) sur un panel
de règles ECA couvrant le spectre ordre -> complexe -> chaos.
Phase 1: génération de trajectoires + split OOD par bassins sur la règle 110."""

import numpy as np

from ddlab_bench import (
    computeBasinField,
    generateTrajectories,
    regimeHint,
    splitByBasins,
    structuralSummary,
    zParameter,
)

N = 14
RULES = [0, 204, 232, 250, 110, 54, 62, 124, 90, 150, 30, 45, 105, 22]


def runPhase0():
    print(f"\n=== Phase 0 — champ de bassins, n={N} ({1 << N} états) ===\n")
    columns = (
        f"{'règle':>6} {'zL':>5} {'zR':>5} {'Z':>5} {'régime':>11} "
        f"{'#attr':>6} {'périodeMax':>11} {'transMax':>9} {'GoE%':>7} {'bassinMax%':>11}"
    )
    print(columns)
    print("-" * len(columns))
    fields = {}
    for rule in RULES:
        field = computeBasinField(rule, N)
        fields[rule] = field
        z = zParameter(rule)
        summary = structuralSummary(field)
        print(
            f"{rule:>6} {z['zLeft']:>5.2f} {z['zRight']:>5.2f} {z['z']:>5.2f} "
            f"{regimeHint(z['z']):>11} {summary['attractorCount']:>6} "
            f"{summary['maxAttractorPeriod']:>11} {summary['maxTransient']:>9} "
            f"{summary['goeFraction'] * 100:>6.1f}% {summary['maxBasinFraction'] * 100:>10.1f}%"
        )
    return fields


def runPhase1(field):
    print(f"\n=== Phase 1 — trajectoires + split OOD, règle {field.ruleNumber} ===\n")
    rng = np.random.default_rng(42)
    traj = generateTrajectories(field, count=2000, length=32, rng=rng)
    print(f"trajectoires générées : {traj.shape[0]} × {traj.shape[1]}")

    split = splitByBasins(field, holdoutFraction=0.3, rng=rng)
    if not split["oodPossible"]:
        print("split OOD impossible : règle à bassin unique (structure triviale)")
        return
    print(
        f"bassins   train : {len(split['trainBasins']):>5}   "
        f"test OOD : {len(split['testBasins'])}"
    )
    print(
        f"états     train : {len(split['trainStates']):>5}   "
        f"test OOD : {len(split['testStates'])}"
    )


if __name__ == "__main__":
    fields = runPhase0()
    runPhase1(fields[110])
