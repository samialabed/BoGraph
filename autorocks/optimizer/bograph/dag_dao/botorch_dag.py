from typing import Any, Callable, Dict, List, Optional, Set, Union

import networkx as nx
import pandas as pd
import torch
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior, Posterior
from gpytorch.distributions import MultitaskMultivariateNormal
from sysgym.params import ParamsSpace
from torch import Tensor

from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import ModelNode
from autorocks.optimizer.bograph.dag_dao.pdag import ProbabilisticDAG


class BoTorchDAG(ProbabilisticDAG, Model):
    """
    Wrapper over BoTorch API to allow integrating the BoGraph (gpytorch) DAG.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        p_space: ParamsSpace,
        structure: nx.DiGraph,
        known_models: Dict[str, Callable[[], ModelNode]],
        opt_obj: Set[str],
    ):

        ProbabilisticDAG.__init__(
            self,
            data=data,
            p_space=p_space,
            structure=structure,
            known_models=known_models,
            opt_obj=opt_obj,
        )
        Model.__init__(self)

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return self.train_x[0].shape[:-2]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self.train_y.shape[-1]

    # TODO: consider overriding the forward method as well The forward method should
    #  go though the graph,  and call mean_module, then combine the mean vectors
    #  using LinearMean()and add the covar of each model using AdditiveKernel

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """

        # TODO: I need to consider redesign of the whole engine
        self.eval()
        posterior_cache: Dict[str, Union[Tensor, Posterior]] = {}
        posterior_cache.update(self._params_dispenser(X))
        target_objective_posterior = []
        for node in nx.topological_sort(self._bograph):
            if node in posterior_cache:
                # Skip already computed results
                continue

            # use dictionary to allow easier access to parameters at the model level
            parents_vals = {}
            for parent in self._bograph.pred[node]:
                # Collect the parents cached results
                parent_samples = posterior_cache[parent]
                # if isinstance(parent_samples, Posterior):
                # Sample the parents distributions
                # parent_samples = parent_samples.rsample().squeeze(0)
                # dict_to_numpy stored parents data according to function stored on edge
                parents_vals[parent] = parent_samples
            bograph_node = self.bograph_attr(self._bograph, node)
            node_output = bograph_node.model.posterior(parents_vals)
            posterior_cache[node] = node_output
            if bograph_node.opt_target:
                target_objective_posterior.append(node_output)

        if len(target_objective_posterior) > 1:
            mvn = MultitaskMultivariateNormal.from_independent_mvns(
                mvns=[posterior.mvn for posterior in target_objective_posterior]
            )
        else:
            return target_objective_posterior[0]

        posterior = GPyTorchPosterior(mvn=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    @property
    def observed_inputs(self) -> Tensor:
        return self.train_x

    @property
    def observed_targets(self) -> Tensor:
        return self.train_y

    @property
    def in_dimensions(self) -> int:
        return self.train_x.shape[-1]
