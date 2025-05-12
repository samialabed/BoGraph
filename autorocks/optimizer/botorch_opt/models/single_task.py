from typing import Optional

import botorch
import gpytorch
from botorch.models import transforms
from torch import Tensor

from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel


class SingleTaskModel(BoTorchModel):
    name: str = "SingleTaskGP"

    def __repr__(self) -> str:
        return "SingleTaskGP()"

    def model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        input_transform: Optional[transforms.input.InputTransform] = None,
        outcome_transform: Optional[transforms.outcome.OutcomeTransform] = None,
    ) -> botorch.models.SingleTaskGP:
        model = botorch.models.SingleTaskGP(train_x, train_y)
        mll = gpytorch.ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        botorch.fit_gpytorch_mll(mll)

        return model
