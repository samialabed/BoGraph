import networkx as nx

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.structure_learn.structure_learn_abc import (
    StructureDiscoveryStrategy,
)


class StaticDAG(StructureDiscoveryStrategy):
    def __init__(self, known_dag: nx.DiGraph):
        super().__init__()
        self.dag = known_dag

    def learn_structure(self, data: BoGraphDataPandas, *args, **kwargs) -> nx.DiGraph:
        return self.dag
