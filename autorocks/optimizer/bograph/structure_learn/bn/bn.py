from dataclasses import dataclass

from dataclasses_json import dataclass_json
from optimizer.models.bograph.dag_options import StructureDiscoveryStrategy
from optimizer.models.bograph.structure_learn.bn.score import (
    BayesianModelSelection,
    ScoreStrategy,
)
from optimizer.models.bograph.structure_learn.bn.search import (
    GreedySearchStrategy,
    SearchStrategy,
)


@dataclass_json
@dataclass
class BayesianNetworkStructureLearning(StructureDiscoveryStrategy):
    score_strategy: ScoreStrategy = BayesianModelSelection()
    search_strategy: SearchStrategy = GreedySearchStrategy()
