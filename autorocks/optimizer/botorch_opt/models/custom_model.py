from botorch import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ExactGP
from torch import Tensor


class CustomSurrogateModel(ExactGP, GPyTorchModel):
    def __init__(
        self,
        num_outputs: int,
        train_x: Tensor,
        train_y: Tensor,
        mean_module: Mean,
        covar_module: Kernel,
    ):
        self._num_outputs = num_outputs
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_x, train_y.squeeze(-1), GaussianLikelihood())
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor, **kwargs) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x: Tensor) -> MarginalLogLikelihood:
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.to(train_x)
        fit_gpytorch_model(mll)
        return mll
