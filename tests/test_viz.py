"""Tests Phase 5 — préparation des données de visualisation."""

from ddlab_bench import computeBasinField
from ddlab_bench.viz import buildBasinGraph


def testBasinGraphStructure():
    field = computeBasinField(110, 8)
    graph = buildBasinGraph(field)
    assert len(graph["nodes"]) == 256
    assert len(graph["edges"]) == 256
    assert graph["rule"] == 110
    assert all(node["kind"] in ("goe", "attractor", "transient") for node in graph["nodes"])
    assert all(node["basin"] >= 0 for node in graph["nodes"])


def testBasinGraphKindCountsMatchField():
    field = computeBasinField(30, 9)
    graph = buildBasinGraph(field)
    goeNodes = [node for node in graph["nodes"] if node["kind"] == "goe"]
    assert len(goeNodes) == len(field.goe)
    attractorNodes = [node for node in graph["nodes"] if node["kind"] == "attractor"]
    assert len(attractorNodes) == sum(len(cycle) for cycle in field.attractors)


def testBasinGraphEdgesFollowSuccessor():
    field = computeBasinField(110, 8)
    graph = buildBasinGraph(field)
    for edge in graph["edges"]:
        assert int(field.succ[edge["source"]]) == edge["target"]
