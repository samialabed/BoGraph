from typing import Dict

import networkx as nx
import torch
from botorch.models.transforms import Normalize
from gpytorch import kernels, means
from torch import Tensor

import autorocks.optimizer.acqf as acqf_wrapper
from autorocks.envs.synthetic.funcs.ackley import Ackley6DParametersSpace
from autorocks.envs.synthetic.synth_objective_dao import TargetFuncObjective
from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.bograph.dag_dao.model_nodes.determinstic_node import (
    DeterminsticModelNode,
)
from autorocks.optimizer.bograph.dag_dao.model_nodes.node_singlegp import (
    SingleTaskGPModelNode,
)
from autorocks.optimizer.bograph.dag_options import BoGraphConfig, DAGPrior
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.preprocessor.normalizer import ParamNormalizerProcessor
from autorocks.optimizer.bograph.preprocessor.standardizer import (
    MetricsStandardizerProcessor,
)
from autorocks.optimizer.bograph.structure_learn.notears.notears import NoTears
from autorocks.optimizer.bograph.structure_learn.scheduler import StaticUpdateStrategy
from autorocks.optimizer.bograph.structure_learn.static_dag import StaticDAG
from autorocks.optimizer.opt_configs import OptimizerConfig

__OBJECTIVE = [TargetFuncObjective()]


class RhDMidDebugTerm(SingleTaskGPModelNode):
    def __init__(self, train_x: Tensor, train_y: Tensor):
        super().__init__(
            train_x,
            train_y,
            covar_module=kernels.ScaleKernel(
                kernels.PeriodicKernel() + kernels.LinearKernel(),
            ),
        )


class LhDMidDebugTerm(SingleTaskGPModelNode):
    def __init__(self, train_x: Tensor, train_y: Tensor):
        super().__init__(
            train_x,
            train_y,
            covar_module=kernels.ScaleKernel(kernels.PeriodicKernel()),
        )


class LinearCombinationOfTerms(SingleTaskGPModelNode):
    def __init__(self, train_x: Tensor, train_y: Tensor):
        super().__init__(
            train_x,
            train_y,
            mean_module=means.LinearMean(input_size=2),
            covar_module=kernels.ScaleKernel(kernels.PeriodicKernel()),
            input_transform=Normalize(d=2),
        )


class AckleyHigh(DeterminsticModelNode):
    def forward(self, parents_vals: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        lhd: Tensor = parents_vals["lhd"]
        rhd: Tensor = parents_vals["rhd"]
        # TODO: make this full bayesian
        y = -lhd - rhd
        return y


def auto(param_space: Ackley6DParametersSpace) -> OptimizerConfig:
    # TODO: should the structure discovery should tabu param space automatically?
    return BoGraphConfig(
        name="AutoBoGraph",
        param_space=param_space,
        opt_objectives=__OBJECTIVE,
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=10),
        structure_discovery_strategy=NoTears(
            parents_free_nodes=set(param_space.keys()),
            tabu_edges={("lhd", "rhd"), ("rhd", "lhd")},
        ),
        acquisition_function=acqf_wrapper.UpperConfidentBoundWrapper(
            beta=0.2, optimizer_cfg=AcqfOptimizerCfg(dim=param_space.dimensions)
        ),
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
    )


def static_low_level(param_space: Ackley6DParametersSpace) -> OptimizerConfig:
    known_dag = nx.DiGraph()
    known_dag.add_edges_from([(p, f"cos({p})") for p in param_space.keys()])
    known_dag.add_edges_from([(p, f"pow({p})") for p in param_space.keys()])

    known_dag.add_edges_from([(f"pow({p})", "lhd") for p in param_space.keys()])
    known_dag.add_edges_from([(f"cos({p})", "rhd") for p in param_space.keys()])

    known_dag.add_edges_from([("lhd", "target"), ["rhd", "target"]])
    # TODO: should include linear term for pow and cos too
    dag_prior = DAGPrior(
        known_models={"target": LinearCombinationOfTerms},
        initial_dag=known_dag,
    )

    return BoGraphConfig(
        name="StaticBoGraphStandardizeLow",
        param_space=param_space,
        opt_objectives=__OBJECTIVE,
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=20),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        acquisition_function=acqf_wrapper.UpperConfidentBoundWrapper(
            beta=0.2, optimizer_cfg=AcqfOptimizerCfg(dim=param_space.dimensions)
        ),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
    )


def static_mid_level(param_space: Ackley6DParametersSpace) -> OptimizerConfig:
    """Bad relationship"""
    known_dag = nx.DiGraph()
    known_dag.add_edges_from([(p, "lhd") for p in param_space.keys()])
    known_dag.add_edges_from([(p, "rhd") for p in param_space.keys()])
    known_dag.add_edges_from([("lhd", "target"), ["rhd", "target"]])

    dag_prior = DAGPrior(
        known_models={
            "target": LinearCombinationOfTerms,
            "lhd": LhDMidDebugTerm,
            "rhd": RhDMidDebugTerm,
        },
        initial_dag=known_dag,
    )

    return BoGraphConfig(
        name="StaticBoGraphStandardizeMid",
        param_space=param_space,
        opt_objectives=__OBJECTIVE,
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=20),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        acquisition_function=acqf_wrapper.UpperConfidentBoundWrapper(
            beta=0.2,
        ),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
    )


def static_high_level(param_space: Ackley6DParametersSpace) -> OptimizerConfig:

    known_dag = nx.DiGraph()
    # known_dag.add_edges_from([(p, "lhd") for p in param_space.keys()])
    # known_dag.add_edges_from([(p, "rhd") for p in param_space.keys()])
    # known_dag.add_edges_from([("lhd", "target"), ["rhd", "target"]])
    #
    known_dag.add_edges_from([(p, "target") for p in param_space.keys()])

    dag_prior = DAGPrior(
        # known_models={"target": AckleyHigh},
        initial_dag=known_dag,
    )

    return BoGraphConfig(
        name="StaticBoGraphStandardizeHigh",
        param_space=param_space,
        opt_objectives=__OBJECTIVE,
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=20),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        acquisition_function=acqf_wrapper.qLowerBoundMaxValueEntropyWrapper(
            candidate_set_size=1000,
            bounds=torch.tensor(param_space.bounds(True).T),
            optimizer_cfg=AcqfOptimizerCfg(dim=param_space.dimensions)
            # sampler=SobolQMCNormalSampler(num_samples=16384),
            # beta=0.2,
        ),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
    )
