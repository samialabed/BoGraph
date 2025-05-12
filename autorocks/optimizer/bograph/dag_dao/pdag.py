from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Union

import gpytorch
import networkx as nx
import pandas as pd
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from sysgym.params import ParamsSpace
from torch import Tensor

from autorocks.global_flags import DEVICE
from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import ModelNode
from autorocks.optimizer.bograph.dag_dao.model_nodes.node_singlegp import (
    SingleTaskGPModelNode,
)
from autorocks.optimizer.bograph.dag_dao.node_abc import Node
from autorocks.optimizer.bograph.dag_dao.obj_node import ObjectiveNode
from autorocks.optimizer.bograph.dag_dao.param_node import ParameterNode
from autorocks.project import ExperimentManager


@dataclass
class BoGraphNodeAttr:
    node: Node
    model: Optional[ModelNode] = None
    opt_target: bool = False


class ParametersDispenser:
    def __init__(self, p_space: ParamsSpace):
        """Simplify moving from samples to parameter and vice-versa"""
        self._param_to_idx = {}

        params = p_space.keys()
        for idx, p in enumerate(params):
            index = torch.tensor(idx, device=DEVICE)
            self._param_to_idx[p] = index

        self._params = set(params)

    def forward(self, samples: Tensor) -> Dict[str, Tensor]:
        # knows which parameter to forward the samples to
        res = {}
        for p, idx in self._param_to_idx.items():
            res[p] = torch.index_select(samples, dim=-1, index=idx)
        return res

    def __call__(self, samples: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        return self.forward(samples)

    @property
    def parameters(self) -> Set[str]:
        return self._params


class ProbabilisticDAG:
    """


    Stateless DAGDao: Assumes observations only given at creation time
    no need observation given after.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        p_space: ParamsSpace,
        structure: nx.DiGraph,
        known_models: Dict[str, Callable[[], ModelNode]],
        opt_obj: Set[str],
    ):
        self.ctx = ExperimentManager()
        gpytorch.settings.debug(self.ctx.debug)
        self._params_dispenser = ParametersDispenser(p_space=p_space)
        self._bograph = self.build_prob_dag(
            data=data,
            structure=structure,
            known_models=known_models,
            opt_obj=opt_obj,
            p_space=p_space,
        )

        # TODO: this seems needlessly inefficient
        train_x = []
        train_y = []
        bounds = []

        for param in p_space.keys():
            bograph_node = self.bograph_attr(self._bograph, param).node
            assert isinstance(bograph_node, ParameterNode)
            bounds.append(bograph_node.bounds)
            train_x.append(bograph_node.observed_values)
        for obj in opt_obj:
            bograph_node = self.bograph_attr(self._bograph, obj).node
            train_y.append(bograph_node.observed_values)

        self.train_x = torch.cat(train_x, -1)
        self.train_y = torch.cat(train_y, -1)
        self.bounds = torch.cat(bounds, -1)

    def forward(self, samples: Tensor) -> MultivariateNormal:
        """
        X is the sampled points to test. The optimizer structure has transformers on
        the edge that dict_to_numpy the output of previous node to one suitable to
        the current node.

        Go through the graph in topological order skipping any orphans (no parents).
        Using the cached results from parents node, perform the dict_to_numpy operation
            stored in the edge.
        Then cache the results of this node not transformed to be used in final node.

        Args:
            samples: Sample points to explore the surrogate optimizer

        Returns: MultivariateNormal of the surrogate optimizer


        """
        # TODO: this doesn't seems to do anything
        cache: Dict[str, Union[Tensor, MultivariateNormal]] = {}
        cache.update(self._params_dispenser(samples))

        target_objective_mvn = []

        for node in nx.topological_sort(self._bograph):
            if node in cache:
                # Skip already computed results
                continue

            # Collect the parents cached results
            parents_vals = {}
            for parent in self._bograph.pred[node]:
                parent_prediction = cache[parent]
                if isinstance(parent_prediction, MultivariateNormal):
                    # Sample the parents distributions
                    parent_prediction = parent_prediction.rsample().unsqueeze(-1)
                # dict_to_numpy stored parents data according to function stored on edge
                parents_vals[parent] = parent_prediction
            bograph_node = self.bograph_attr(self._bograph, node)
            node_output = bograph_node.model.forward(parents_vals)
            cache[node] = node_output
            if bograph_node.opt_target:
                target_objective_mvn.append(node_output)

        if len(target_objective_mvn) > 1:
            mvn = MultitaskMultivariateNormal.from_independent_mvns(
                target_objective_mvn
            )
        else:
            mvn = target_objective_mvn[0]
        return mvn

    def mll(self) -> Tensor:
        """
        train models independently and sequentially.
        and return the sum of their MLL
        """
        mlls = []
        for node in self._bograph.nodes(data=True):
            bograph_node = self.bograph_attr(structure=self._bograph, node=node)
            mlls.append(bograph_node.model.mll())
        mll = torch.sum(torch.cat(mlls, -1), -1)
        return mll

    def save_model(self):
        nx.write_gpickle(
            self._bograph, self.ctx.model_checkpoint_dir / "model_dag.gpickle"
        )

    def load_model(self):
        self._bograph = nx.read_gpickle(
            self.ctx.model_checkpoint_dir / "model_dag.gpickle"
        )

    @staticmethod
    def build_prob_dag(
        data: pd.DataFrame,
        p_space: ParamsSpace,
        structure: nx.DiGraph,
        known_models: Dict[str, Callable[[], ModelNode]],
        opt_obj: Set[str],
    ) -> nx.DiGraph:
        parameters = set(p_space.keys())

        for node in nx.topological_sort(structure):
            # if node in known_models:
            #     model = known_models[node]
            values_ = data[node].to_numpy()

            if node in parameters:
                bograph_node = ParameterNode(
                    name=node, param=p_space[node], values=values_
                )
            else:
                bograph_node = ObjectiveNode(name=node, values=values_)

            # if a node has a parents then it should contain a model
            if structure.pred[node]:
                # get the parent inputs
                parents_vals = []
                for parent in structure.pred[node]:
                    parent_node = ProbabilisticDAG.bograph_attr(
                        structure=structure, node=parent
                    )
                    parents_vals.append(parent_node.node.observed_values)
                training_samples = torch.cat(parents_vals, -1)

                if node in known_models:
                    bograph_model = known_models[node]
                else:
                    bograph_model = SingleTaskGPModelNode
                # TODO: allow passing options
                bograph_model = bograph_model(
                    train_x=training_samples, train_y=bograph_node.observed_values
                )
            else:
                # node has no model
                bograph_model = None

            bograph_attr = BoGraphNodeAttr(
                node=bograph_node,
                opt_target=node in opt_obj,
                model=bograph_model,
            )
            ProbabilisticDAG.set_bograph_attr(
                structure=structure, node=node, attr=bograph_attr
            )

        return structure

    @staticmethod
    def bograph_attr(structure: nx.DiGraph, node: str) -> BoGraphNodeAttr:
        return structure.nodes[node]["bograph_attr"]

    @staticmethod
    def set_bograph_attr(
        structure: nx.DiGraph, node: str, attr: BoGraphNodeAttr
    ) -> None:
        structure.nodes[node]["bograph_attr"] = attr

    def eval(self):
        """ensure all nodes in the graph are in eval"""
        for node in self._bograph:
            node_attr = self.bograph_attr(structure=self._bograph, node=node)
            if node_attr.model:
                node_attr.model.eval()

    def train(self, mode: bool = True):
        for node in self._bograph:
            node_attr = self.bograph_attr(structure=self._bograph, node=node)
            if node_attr.model:
                node_attr.model = node_attr.model.train(mode)
