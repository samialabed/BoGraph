import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses_json import dataclass_json
from optimizer.models.bograph.dag_model import DAGModel


class EdgeConstraints(enum.Enum):
    ALLOW = "allow"
    FORCE = "force"
    BLOCK = "block"


@dataclass_json
@dataclass
class SearchStrategy(ABC):
    # max_num_dims:  maximum allowed dimension for each node

    max_num_dims: Optional[int] = None
    # edges_constraints: any edges that should be forced/ignored

    edges_constraints: Dict[str, Dict[str, EdgeConstraints]] = None

    @abstractmethod
    def search(self, dag_model: DAGModel):
        pass


class GreedySearchStrategy(SearchStrategy):
    def search(self, dag_model: DAGModel):
        pass
