from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
from dataclasses_json import dataclass_json

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas


@dataclass_json
@dataclass
class StructureDiscoveryStrategy(ABC):
    @abstractmethod
    def learn_structure(self, data: BoGraphDataPandas, *args, **kwargs) -> nx.DiGraph:
        pass
