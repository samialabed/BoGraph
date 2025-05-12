import warnings
from typing import Any, Callable, Mapping

import botorch
import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.posteriors import GPyTorchPosterior
from gpytorch import ExactMarginalLogLikelihood

from autorocks.optimizer.bograph.bobn import BoBn

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op", category=UserWarning
)


def run_model(
    m: BoBn, observations: Mapping[str, Any]
) -> Callable[[torch.Tensor], GPyTorchPosterior]:

    return lambda test_x: m.posterior(
        test_x,
        preprocess_observations=True,
        training_observations=observations,
    )


def botorch_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    tkwargs: Mapping[str, Any],
) -> Callable[[torch.Tensor], GPyTorchPosterior]:
    model = SingleTaskGP(
        train_X=train_x, train_Y=train_y, outcome_transform=Standardize(1)
    ).to(**tkwargs)
    # Fit and train the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit_gpytorch_mll(mll)
    model.eval()
    return lambda test_x: model.posterior(test_x)
