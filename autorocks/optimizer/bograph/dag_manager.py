import logging
from typing import Set

import networkx as nx
from sysgym.params import ParamsSpace

from autorocks.logging_util import BOGRAPH_LOGGER
from autorocks.optimizer.bograph.bograph_dao import BoGraphIntermediateData
from autorocks.optimizer.bograph.dag_dao.botorch_dag import BoTorchDAG
from autorocks.optimizer.bograph.dag_options import BoGraphConfig
from autorocks.optimizer.bograph.dag_postprocessor import postprocess_structure
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.project import ExperimentManager

LOG = logging.getLogger(BOGRAPH_LOGGER)


class DAGManager:
    """

    DAGManager responsibilities:
        * Takes in the BoGraphConfig: needs it for prior, method, and update schedule
        * Refit all models
        * It gets called all the time to create a DAG optimizer with the full dataset.
        * decides whether to update the edges or use previous edges
        * the preprocessing pipeline should be saved and reused if no updates needed


    * TODO: Time how long it takes to perform observation update, processing, etc...


    Responsibilities:

    Calls the updater to know when to update


    Parse the dag optimizer configurations

    Gets env config to know what are the parameter space and what are the objectives

    Creates initial DAG too


    get_dag():
    If needs update: update the edges and apply prior
    if no need for update: fit the graph anyway, and provide the dag

    """

    def __init__(
        self,
        model_cfg: BoGraphConfig,
        objs: Set[str],
        preprocessor: PreprocessingPipeline,
        param_space: ParamsSpace,
    ):
        self.ctx = ExperimentManager()
        self.prior = model_cfg.prior
        self.scheduler = model_cfg.update_strategy
        self.structure_learner = model_cfg.structure_discovery_strategy
        self.preprocessor = preprocessor
        self._objs = objs
        self._dag = model_cfg.prior.initial_dag
        self.param_space = param_space

    def get_dag(self, historic_obs: BoGraphIntermediateData) -> BoTorchDAG:
        # preprocessing pipeline
        processed_data = self.preprocessor.fit_transform(historic_obs.to_pandas())

        if self.scheduler.eval():
            # learn structure
            dag = self.structure_learner.learn_structure(processed_data)
            # checkpoint a structure if debug is on (should be a schedule?)
            if self.ctx.debug:
                with open(
                    self.ctx.model_checkpoint_dir / "learned_dag_edgelist.txt", "wb"
                ) as f:
                    nx.write_edgelist(dag, f)

            sources = set(processed_data.params.columns.to_list())
            if processed_data.objs.ndim == 1:
                sinks = processed_data.objs.to_frame().columns.to_list()
            else:
                sinks = set(processed_data.objs.columns.to_list())
            dag = postprocess_structure(dag, sources, sinks, self.prior.known_edges)

            self._dag = dag
            # Save dag after postprocessing
            with open(self.ctx.model_checkpoint_dir / "structure.gz", "wb") as f:
                nx.write_gpickle(dag, f)
        if self._dag is None:
            # No DAG produced assume all param linked to the objectives
            self._dag = nx.DiGraph()
            for obj in self._objs:
                self._dag.add_edges_from([(p, obj) for p in self.param_space.keys()])

        # TODO: at every iteration at random connect all parameters to the objective
        dag_nodes = self._dag.nodes.keys()
        subset_data = processed_data.to_combi_pandas()[dag_nodes]

        params_to_keep = set()
        for p in self.param_space.keys():
            if p in dag_nodes:
                params_to_keep.add(p)

        self.param_space = self.param_space.subset(params_to_keep)

        p_dag = BoTorchDAG(
            data=subset_data,
            p_space=self.param_space,
            structure=self._dag,
            known_models=self.prior.known_models,
            opt_obj=self._objs,
        )
        return p_dag
