from typing import Optional

import botorch
import gpytorch
import torch
from botorch.models import transforms

from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel


class MultiTaskModel(BoTorchModel):
    name: str = "MultiTaskGP"

    def __repr__(self) -> str:
        return "MultiTaskGP()"

    def model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        input_transform: Optional[transforms.input.InputTransform] = None,
        outcome_transform: Optional[transforms.outcome.OutcomeTransform] = None,
    ) -> botorch.models.KroneckerMultiTaskGP:

        model = botorch.models.KroneckerMultiTaskGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            linear=True,
        )
        mll = gpytorch.ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        botorch.fit_gpytorch_mll(mll)

        return model
