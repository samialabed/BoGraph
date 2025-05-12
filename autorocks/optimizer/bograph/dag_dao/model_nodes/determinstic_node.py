from typing import Dict, Union

import torch
from botorch.models.deterministic import DeterministicModel
from botorch.posteriors import DeterministicPosterior, Posterior
from torch import Tensor

from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import ModelNode


class DeterminsticModelNode(ModelNode, DeterministicModel):
    def mll(self) -> Tensor:
        return torch.zeros()

    def _fit(self) -> Tensor:
        pass

    def forward(self, parents_vals: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        pass

    def posterior(
        self,
        X: Union[Dict[str, Tensor], Tensor],
        *args,
        **kwargs,
    ) -> Posterior:
        return DeterministicPosterior(values=self.forward(X))
