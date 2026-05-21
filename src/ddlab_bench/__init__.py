"""ddlab-sense-benchmark.

DDLab fournit la vérité-terrain exacte d'un système dynamique discret (bassins,
attracteurs, garden-of-Eden). On mesure si un modèle d'apprentissage retrouve
cette structure causale à partir des seules trajectoires observées."""

from ddlab_bench.eca import ruleTable, stepState, buildSuccessorArray
from ddlab_bench.basins import (
    BasinField,
    computeBasinField,
    basinFieldFromSuccessor,
    buildPreimageArray,
    iterateSuccessor,
    findAttractors,
    gardenOfEden,
    transientLengths,
)
from ddlab_bench.rbn import RandomBooleanNetwork
from ddlab_bench.cml import CoupledMapLattice, logisticMap
from ddlab_bench.classify import zParameter, structuralSummary, regimeHint
from ddlab_bench.trajectories import generateTrajectories, splitByBasins
from ddlab_bench.artifacts import buildBundle, loadBundle
from ddlab_bench.renormalization import (
    refinePartition,
    bisimulationQuotient,
    quotientGraph,
    statisticalComplexity,
    densityLabel,
    windowLabel,
    bisimulationMetric,
)

__all__ = [
    "ruleTable",
    "stepState",
    "buildSuccessorArray",
    "BasinField",
    "computeBasinField",
    "basinFieldFromSuccessor",
    "RandomBooleanNetwork",
    "CoupledMapLattice",
    "logisticMap",
    "buildPreimageArray",
    "iterateSuccessor",
    "findAttractors",
    "gardenOfEden",
    "transientLengths",
    "zParameter",
    "structuralSummary",
    "regimeHint",
    "generateTrajectories",
    "splitByBasins",
    "buildBundle",
    "loadBundle",
    "refinePartition",
    "bisimulationQuotient",
    "quotientGraph",
    "statisticalComplexity",
    "densityLabel",
    "windowLabel",
    "bisimulationMetric",
]
