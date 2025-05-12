from dataclasses import dataclass
from typing import Optional, Set, Tuple

import networkx as nx
from causalnex.structure.pytorch import from_pandas
from dataclasses_json import dataclass_json

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.dag_options import StructureDiscoveryStrategy


@dataclass_json
@dataclass
class NoTears(StructureDiscoveryStrategy):
    def __init__(
        self,
        childfree_nodes: Optional[Set[str]] = None,
        parents_free_nodes: Optional[Set[str]] = None,
        tabu_edges: Optional[Set[Tuple[str, str]]] = None,
    ):
        """
        Args:
            childfree_nodes: nodes not allowed to have children
            parents_free_nodes: nodes not allowed to be parents
        """
        super().__init__()
        self._childfree_nodes = childfree_nodes
        self._parents_free_nodes = parents_free_nodes
        self._tabu_edges = tabu_edges

    def learn_structure(self, data: BoGraphDataPandas, *args, **kwargs) -> nx.DiGraph:
        return from_pandas(
            X=data.to_combi_pandas(),
            tabu_parent_nodes=self._childfree_nodes,
            tabu_child_nodes=self._parents_free_nodes,
            tabu_edges=self._tabu_edges,
            w_threshold=0.01,
        )
