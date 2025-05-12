from abc import ABC, abstractmethod
from typing import Optional

from botorch.models import transforms
from botorch.models.model import Model
from torch import Tensor


class BoTorchModel(ABC):
    name: str

    @abstractmethod
    def model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        input_transform: Optional[transforms.input.InputTransform] = None,
        outcome_transform: Optional[transforms.outcome.OutcomeTransform] = None,
    ) -> Model:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return self.name
