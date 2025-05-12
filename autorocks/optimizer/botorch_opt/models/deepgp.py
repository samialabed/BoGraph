from typing import Optional

import gpytorch
import gpytorch.means
import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.optim import ExpMAStoppingCriterion
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor
from torch.utils.data import DataLoader

from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel
from autorocks.project import ExperimentManager


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int] = None,
        num_inducing: int = 64,
        mean_type: str = "constant",
        num_samples: int = 10,
    ):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        if mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape, ard_num_dims=input_dims
            ),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

        self.num_samples = num_samples

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGPNetwork(DeepGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(
        self,
        input_dims: int,
        likelihood: Likelihood,
        hidden_layer: int,
    ):
        self.ctx = ExperimentManager()
        self.logger = self.ctx.logger

        # TODO: do we need this?
        # self.train_inputs = train_x
        # self.train_targets = train_y

        hidden_layer = DeepGPHiddenLayer(
            input_dims=input_dims,
            output_dims=hidden_layer,
            mean_type="linear",
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type="constant",
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = likelihood

    def forward(self, x: Tensor):
        hidden_rep1 = self.hidden_layer(x)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader: DataLoader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return (
            torch.cat(mus, dim=-1),
            torch.cat(variances, dim=-1),
            torch.cat(lls, dim=-1),
        )

    def __call__(self, *args, **kwargs):
        predictive_dis = super().__call__(*args, **kwargs)
        # mixture of gaussian output
        average_of_mixtures = gpytorch.distributions.MultivariateNormal(
            predictive_dis.mean.mean(0), predictive_dis.covariance_matrix.mean(0)
        )
        return average_of_mixtures

    def fit_model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        training_iterations: int,
        num_samples: int,
    ):
        self.likelihood.train()
        self.train()

        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=self.ctx.debug
        )

        mll = DeepApproximateMLL(
            VariationalELBO(self.likelihood, self, train_y.numel())
        )
        mll.to(train_x)

        convergence = ExpMAStoppingCriterion(maxiter=training_iterations)
        losses = []

        for epoch in range(training_iterations):
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = self(train_x)
                loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
            if epoch % 100 == 0:
                self.logger.debug(f"Iter: {epoch}, Loss:{loss.item()}")
            optimizer.step()
            scheduler.step(loss)

            if convergence.evaluate(loss.detach()):
                self.logger.debug(
                    f"Early stopping - Iter: {epoch}, Loss: {loss.item()}"
                )
                break

        with open(self.ctx.model_checkpoint_dir / "deepgp_loss.csv", "wb") as f:
            np.savetxt(f, np.asarray(losses), fmt="%s")

        self.eval()
        self.likelihood.eval()


class DeepGPModel(BoTorchModel):
    name: str = "DeepGPModel"

    def __repr__(self) -> str:
        return (
            f"DeepGPModel("
            f"likelihood={self._likelihood}, "
            f"hidden_layer:{self._hidden_layer}, "
            f"fit_training_iterations={self._fit_training_iterations},"
            f" fit_num_samples={self._fit_num_samples}"
            f")"
        )

    def __init__(
        self,
        likelihood: Likelihood = gpytorch.likelihoods.GaussianLikelihood(),
        hidden_layer: int = 2,
        fit_training_iterations: int = 10000,
        fit_num_samples: int = 10,
    ):
        self._likelihood = likelihood
        self._hidden_layer = hidden_layer
        self._fit_num_samples = fit_num_samples
        self._fit_training_iterations = fit_training_iterations

    def model(self, train_x: Tensor, train_y: Tensor) -> Model:
        input_dims = train_x.shape[-1]
        deepgp_model = DeepGPNetwork(
            input_dims=input_dims,
            likelihood=self._likelihood,
            hidden_layer=self._hidden_layer,
        )
        deepgp_model.to(train_x)
        deepgp_model.fit_model(
            train_x=train_x,
            train_y=train_y,
            training_iterations=self._fit_training_iterations,
            num_samples=self._fit_num_samples,
        )

        return deepgp_model
