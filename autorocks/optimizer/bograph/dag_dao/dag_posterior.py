from typing import Dict, List, Optional

import networkx as nx
import torch
from botorch.posteriors import Posterior
from torch import Tensor


class DAGPosterior(Posterior):
    def __init__(
        self,
        dag_structure: nx.DiGraph,
        posterior_list: Dict[str, Posterior],
    ):
        self._dag_structure = dag_structure
        self._posterior_list = posterior_list

    @property
    def device(self) -> torch.device:
        pass

    @property
    def dtype(self) -> torch.dtype:
        pass

    @property
    def event_shape(self) -> torch.Size:
        pass

    @property
    def mean(self) -> Tensor:
        pass

    @property
    def variance(self) -> Tensor:
        pass

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        cache: Dict[str, Tensor] = {}
        for node in nx.topological_sort(self._dag_structure):
            if node in cache:
                # Skip already computed results
                continue

            # Collect the parents cached results
            parents_vals = {}
            for parent in self._dag_structure.pred[node]:
                parent_posterior = cache[parent]
                parent_prediction = parent_posterior.rsample().unsqueeze(-1)
                # dict_to_numpy stored parents data according to function stored on edge
                parents_vals[parent] = parent_prediction
            node_output = bograph_node.model.forward(parents_vals)
            cache[node] = node_output
