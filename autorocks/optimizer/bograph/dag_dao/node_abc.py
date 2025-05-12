from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from autorocks.global_flags import DEVICE
from autorocks.project import ExperimentManager


class Node(ABC):
    """Stateless Node: doesn't update observations after creation."""

    def __init__(self, name: str, values: np.ndarray):
        self.ctx = ExperimentManager()
        self._values = torch.tensor(
            values, dtype=torch.double, device=DEVICE
        ).unsqueeze(-1)
        self.name = name

    @property
    @abstractmethod
    def observed_values(self) -> Tensor:
        pass
