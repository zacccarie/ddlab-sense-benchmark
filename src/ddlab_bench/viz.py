"""Phase 5 — préparation des données de visualisation du champ de bassins.

Convertit un champ de bassins (vérité-terrain) en graphe sérialisable que le
front D3 consomme: nœuds = états (avec bassin, type, transitoire), arêtes =
transitions du graphe fonctionnel."""

from ddlab_bench.classify import structuralSummary, zParameter


def buildBasinGraph(field):
    """Champ de bassins -> dict sérialisable (nœuds + arêtes + métadonnées) pour D3."""
    size = 1 << field.n
    onCycle = {state for cycle in field.attractors for state in cycle}
    goeSet = set(field.goe)

    nodes = []
    for state in range(size):
        if state in goeSet:
            kind = "goe"
        elif state in onCycle:
            kind = "attractor"
        else:
            kind = "transient"
        nodes.append({
            "id": state,
            "basin": int(field.attractorId[state]),
            "kind": kind,
            "transient": int(field.transient[state]),
        })

    edges = [{"source": state, "target": int(field.succ[state])} for state in range(size)]
    summary = structuralSummary(field)
    return {
        "rule": field.ruleNumber,
        "n": field.n,
        "z": zParameter(field.ruleNumber)["z"],
        "basinCount": len(field.attractors),
        "goePercent": round(summary["goeFraction"] * 100, 1),
        "maxPeriod": summary["maxAttractorPeriod"],
        "nodes": nodes,
        "edges": edges,
    }
