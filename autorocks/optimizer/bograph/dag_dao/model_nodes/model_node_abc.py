from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Union

import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from autorocks.global_flags import DEVICE


class PredictionResult(NamedTuple):
    mean: Tensor
    lower: Tensor
    upper: Tensor

    def as_numpy(self) -> "PredictionResult":
        return PredictionResult(
            mean=self.mean.squeeze().cpu().numpy(),
            lower=self.lower.squeeze().cpu().numpy(),
            upper=self.upper.squeeze().cpu().numpy(),
        )


class ModelNode(Model, ABC):

    num_outputs: int

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
    ):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y

        self.train_x.to(DEVICE)
        self.train_y.to(DEVICE)

    @abstractmethod
    def mll(self) -> Tensor:
        """Return the MLL of the model"""

    @abstractmethod
    def _fit(self) -> Tensor:
        """Return the MLL after fitting the node's model"""

    @abstractmethod
    def posterior(
        self,
        X: Union[Dict[str, Tensor], Tensor],
        *args,
        **kwargs,
    ) -> Posterior:
        pass

    @abstractmethod
    def forward(
        self, parents_vals: Dict[str, Tensor], *args, **kwargs
    ) -> MultivariateNormal:
        pass

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    def predict(self, X: Union[Dict[str, Tensor], Tensor]) -> PredictionResult:
        with torch.no_grad():
            posterior = self.posterior(X)
            predicted_mean = posterior.mean
            std = torch.sqrt(posterior.variance) * 2
            lower, upper = predicted_mean - std, predicted_mean + std
        return PredictionResult(lower=lower, mean=predicted_mean, upper=upper)
