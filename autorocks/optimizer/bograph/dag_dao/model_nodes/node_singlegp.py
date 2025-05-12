from typing import Any, Dict, Optional, Union

import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import InputTransform
from botorch.posteriors import GPyTorchPosterior, Posterior
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import LinearMean
from torch import Tensor

from autorocks.global_flags import DEVICE
from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import ModelNode


class SingleTaskGPModelNode(ModelNode):
    num_outputs = 1

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        covar_module=None,
        mean_module=None,
        input_transform: Optional[InputTransform] = None,
        small_noise: bool = True,
    ):
        super().__init__(train_x, train_y)
        # TODO: linear when it is combining multiple metrics, otherwise use constant?
        if mean_module is None:
            mean_module = LinearMean(input_size=train_x.shape[-1])

        if small_noise:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        else:
            likelihood = None
        self.model = SingleTaskGP(
            # train_X should be: [Batch size, n samples, d dim]
            # We don't use batch size in bograph so assume it is 1
            train_X=self.train_x,
            train_Y=self.train_y,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(m=1),
            # assume small noise
            likelihood=likelihood,
            # TODO: use Normalize input transform in graph structure
            input_transform=input_transform,
        )

        self._mll = self._fit()

    def mll(self) -> Tensor:
        return self._mll

    def _fit(self) -> Tensor:
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        mll.to(DEVICE)
        mll = fit_gpytorch_model(mll)
        return mll  # mll(self.train_x, self.train_y)

    def forward(
        self, parents_vals: Union[Dict[str, Tensor], Tensor], *args, **kwargs
    ) -> MultivariateNormal:
        if isinstance(parents_vals, Dict):
            samples = torch.cat(list(parents_vals.values()), -1)
        else:
            samples = parents_vals
        model_prediction = self.model.forward(samples)
        return model_prediction

    def eval(self):
        self.model.eval()

    def train(self, mode: bool = True) -> Model:
        r"""Puts the model in `train` mode and reverts to the original inputs.

        Args:
            mode: A boolean denoting whether to put in `train` or `eval` mode.
                If `False`, model is put in `eval` mode.
        """
        return self.model.train(mode)

    def subset_output(self, *args, **kwargs) -> Model:
        return self.model.subset_output(*args, **kwargs)

    def condition_on_observations(self, *args, **kwargs) -> Model:
        return self.model.condition_on_observations(*args, **kwargs)

    @classmethod
    def construct_inputs(cls, *args, **kwargs) -> Dict[str, Any]:
        return SingleTaskGP.construct_inputs(*args, **kwargs)

    def posterior(
        self,
        X: Union[Dict[str, Tensor], Tensor],
        *args,
        **kwargs,
    ) -> Posterior:

        if isinstance(X, Dict):
            parent_tensor_pred = []
            for p in X.values():
                if isinstance(p, GPyTorchPosterior):
                    parent_tensor_pred.append(p.rsample(sample_shape=torch.Size([])))
                elif isinstance(p, Tensor):
                    parent_tensor_pred.append(p)
                else:
                    raise Exception(f"Unrecognised type: {type(p)}")

            X = torch.cat(parent_tensor_pred, dim=-1)
            # # TODO: Hack just to check if jointly sampling improve posterior
            # mvns = []
            # # TODO: track the index it is being added at and then concat at that level
            # tensors = []
            # for p in parent_posterior.values():
            #     if isinstance(p, GPyTorchPosterior):
            #         mvns.append(p.mvn)
            #     elif isinstance(p, Tensor):
            #         tensors.append(p)
            #     else:
            #         raise Exception(f"Unrecognised type: {type(p)}")
            #
            # if mvns:
            #     joint_mvns = MultitaskMultivariateNormal.from_independent_mvns(
            #         mvns=mvns
            #     )
            #     # jointly sample parents
            #     parent_posterior = joint_mvns.rsample()
            # if tensors:
            #     # TODO: this wouldn't work when you have a mix of tensors and model
            #     #  we need to ensure the order is respected when concating them
            #     parent_posterior = torch.cat(tensors, -1)
            # # parent_posterior = torch.cat(list(parent_posterior.values()), -1)
        elif not isinstance(X, Tensor):
            raise Exception(
                "Condition on either X: Tensor, "
                "or parents_distribution Dict[str, Tensor]"
            )

        if X.device != DEVICE:
            X = X.to(DEVICE)

        return self.model.posterior(X, *args, **kwargs)
