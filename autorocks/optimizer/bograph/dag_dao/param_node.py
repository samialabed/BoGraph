import numpy as np
import torch
from sysgym.params.boxes import ParamBox
from torch import Tensor

from autorocks.optimizer.bograph.dag_dao.node_abc import Node


class ParameterNode(Node):
    def __init__(self, name: str, param: ParamBox, values: np.ndarray):
        """Objective to watch for.
        Name of the node (name of objective or parameter)
        Should contain the historical values for this node.
        """
        super().__init__(name=name, values=values)
        # Use non-scaled bounds to avoid suggesting parameters outside the "scaled"
        # region
        self.bounds = torch.tensor(param.bounds).unsqueeze(1)

    @property
    def observed_values(self) -> Tensor:
        """Return normalized Torch friendly training values"""
        return self._values
