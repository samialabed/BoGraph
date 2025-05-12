from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from torch import Tensor

from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel


class TurboModel(BoTorchModel):
    """Using https://botorch.org/tutorials/turbo_1"""

    name: str = "TurboModel"

    def __repr__(self) -> str:
        return "TurboModel "

    def __init__(self, dim: int):
        self._likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        self._covar_module = ScaleKernel(
            # Use the same length-scale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        self.state = None

    def model(self, train_x: Tensor, train_y: Tensor) -> Model:
        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=self._covar_module,
            likelihood=self._likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        fit_gpytorch_model(mll)
        return model
