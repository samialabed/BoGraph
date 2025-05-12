from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from optimizer.models.bograph.dag_model import DAGModel


@dataclass_json
@dataclass
class ScoreStrategy(ABC):
    # abstract file that scores a graph
    # different implementations: BD score, BC score, etc...
    @abstractmethod
    def score(self, dag: DAGModel) -> float:
        pass


class BICScoreStrategy(ScoreStrategy):
    def score(self, dag: DAGModel) -> float:
        # This uses MLL and prior
        pass


class BayesianModelSelection(ScoreStrategy):
    def score(self, dag: DAGModel) -> float:
        # This uses MLL and prior
        pass


class LikelihoodScore(ScoreStrategy):
    def score(self, dag: DAGModel) -> float:
        pass
