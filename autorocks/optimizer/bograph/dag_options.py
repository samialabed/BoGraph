from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
from dataclasses_json import config, dataclass_json
from sysgym.params import ParamsSpace

from autorocks.envs.objective_dao import OptimizationObjective
from autorocks.optimizer.acqf.acqf_abc import AcquisitionFunctionWrapperABC
from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import ModelNode
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.structure_learn.scheduler import UpdateStrategy
from autorocks.optimizer.bograph.structure_learn.structure_learn_abc import (
    StructureDiscoveryStrategy,
)
from autorocks.optimizer.opt_configs import OptimizerConfig


def initial_dag_from_env(
    param_space: ParamsSpace, targets: List[OptimizationObjective]
) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_edges_from([(p, t.name) for p, t in zip(param_space.keys(), targets)])

    for target in targets:
        # TODO: Nodes in nx.DiGraph are key: str, value: NamedTuple(Sink:True)
        dag.nodes(data=True)[target]["sink"] = True
    return dag


@dataclass_json
@dataclass
class DAGPrior:
    # known_edges: Edges that will always be used, despite what learning algorithm does
    known_edges: Optional[List[Tuple[str, str]]] = None

    # known_models: models that will always be used, overriding the GP priors
    known_models: Dict[str, ModelNode] = field(
        metadata=config(encoder=str), default_factory=dict
    )

    # initial_dag: starting DAG before learning new structure
    initial_dag: Optional[nx.DiGraph] = field(
        metadata=config(encoder=str), default=None
    )
    # tabu_edges: the edges that aren't allowed
    tabu_edges: Optional[List[Tuple[str, str]]] = None


@dataclass_json
@dataclass
class BoBnConfig(OptimizerConfig):
    random_iter: int  # number of random iterations before starting the optimizer
    retry: int  # number of retries when BO optimization fail
    dag: nx.DiGraph
    seed: Optional[int] = None
    conservative_mode: bool = True
    use_turbo: bool = False


# TODO: Update to match the new interface.
@dataclass_json
@dataclass
class BoGraphConfig(OptimizerConfig):
    random_iter: int  # number of random iterations before starting the optimizer
    retry: int  # number of retries when BO optimization fail
    update_strategy: UpdateStrategy
    structure_discovery_strategy: StructureDiscoveryStrategy
    acquisition_function: AcquisitionFunctionWrapperABC = field(
        metadata=config(encoder=repr)
    )
    preprocessing_pipeline: PreprocessingPipeline = field(metadata=config(encoder=str))
    seed: Optional[int] = None
    restore_from_checkpoint: bool = False
    prior: DAGPrior = DAGPrior()
