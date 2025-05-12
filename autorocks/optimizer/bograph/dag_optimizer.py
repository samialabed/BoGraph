import logging
import re
from typing import Dict, Mapping

import pandas as pd
import torch
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from tenacity import before_sleep_log, retry, stop_after_attempt
from torch import Tensor

from autorocks.envs.env_state import EnvState
from autorocks.optimizer.bograph.bobn import BoBn
from autorocks.optimizer.bograph.bograph_dao import BoGraphIntermediateData
from autorocks.optimizer.bograph.dag_dao.botorch_dag import BoTorchDAG
from autorocks.optimizer.bograph.dag_manager import DAGManager
from autorocks.optimizer.bograph.dag_options import BoBnConfig, BoGraphConfig
from autorocks.optimizer.objective_manager import ObjectiveManager
from autorocks.optimizer.optimizer_abc import Optimizer
from autorocks.project import ExperimentManager


def _clean_df_for_rocksdb(df: pd.DataFrame):
    # TODO: make this more generic not only for rocksdb.
    df.rename(columns=lambda x: re.sub(r"^db_bench\.\w*\.", "", x), inplace=True)
    df.rename(columns=lambda x: re.sub("rocksdb_?", "", x), inplace=True)
    df.rename(columns=lambda x: re.sub(r"_statistics", "", x), inplace=True)
    df.rename(columns=lambda x: re.sub(r"_stats", "", x), inplace=True)
    df.rename(columns=lambda x: re.sub(r"^statistics.", "", x), inplace=True)


class BoBnOptimizer(Optimizer):
    def __init__(self, cfg: BoBnConfig):
        super().__init__(cfg)
        self.ctx = ExperimentManager()
        self.cfg = cfg
        self.param_space = cfg.param_space
        self.obj_manager = ObjectiveManager(cfg.opt_objectives)
        self.logger = self.ctx.logger
        self.observed_states: pd.DataFrame = pd.DataFrame()

        self._batch_size = 1

        self._bounds = torch.tensor(
            self.param_space.bounds().T,
            dtype=torch.double,
        )

        self._bobn = BoBn(
            dag=cfg.dag,
            params=self.param_space,
            objectives=set([obj.name for obj in cfg.opt_objectives]),
            conservative_mode=cfg.conservative_mode,
            use_turbo=cfg.use_turbo,
        )
        self._initial_points = self._bobn.generate_initial_points(cfg.random_iter)
        self._initial_points.append(self._generate_defaults())

        self.bo_loop = retry(
            stop=stop_after_attempt(self.cfg.retry),
            reraise=True,
            before_sleep=before_sleep_log(self.logger, logging.DEBUG),
        )(self.optimize_space)

        self.logger.info("Using GPU: %s.", torch.cuda.is_available())

    def optimize_space(self) -> Mapping[str, any]:
        """return parameters as suggested by the optimizer."""
        if len(self._initial_points) > 0:
            suggested_params = self._initial_points.pop()
        else:
            suggested_params = self._bobn.opt(self.observed_states)
        self.logger.info("Next suggested configurations: %s.", suggested_params)
        return suggested_params

    def observe_state(self, state: EnvState):
        """Feed the new measurement to the optimizer to update prior."""
        # TODO: This is too complicated, I should have just take in dataframe.
        # Exist as a workaround until full refactor.
        self._save_observation(state)
        observation = dict(state.params)
        observation.update(state.as_dict())
        df = pd.json_normalize(observation)
        _clean_df_for_rocksdb(df)
        # Override the objective to match the max/minimize loop
        objectives = self.obj_manager.measurement_to_obj_val_dict(state.measurements)
        for obj, val in objectives.items():
            df[obj] = val
        self.logger.info("Target objective: %s", objectives)
        self.observed_states = pd.concat([self.observed_states, df], ignore_index=True)


# TODO: Migrate to the new API
class DAGOptimizer(Optimizer):
    """DAG Based Bayesian Optimization

    BOGraph components:
        * DAGOptimizer performs the BO loop and maintain history.
        * DAGManager builds the DAG optimizer from data or update it
        * DAGModel inference and sampling the DAG Model

    DAGOptimizer responsibility:
        * Calls the DAG optimizer for inference
        * Pass the DAGModel to ACQF
        * Stores all previous observation in memory.
        * Should use the bograph_causal_edp.py interface to allow MOBO
    """

    def __init__(self, cfg: BoGraphConfig):
        """Contains the best found DAG so far"""
        super().__init__(cfg)
        # Everything below here should be refactored out
        self._batch_size = 1
        self._burn_in_iterations = cfg.random_iter
        self.bo_loop = retry(
            stop=stop_after_attempt(self.cfg.retry),
            reraise=True,
            before_sleep=before_sleep_log(self.logger, logging.DEBUG),
        )(self._bo_loop)

        # Reason for keeping historical_obs in a list here: * We can in the future
        # move towards a full database with calls that retrieve the history delegated
        # to dedicated db
        self.historical_obs = BoGraphIntermediateData(obj_manager=self.obj_manager)
        self.dag_manager = DAGManager(
            model_cfg=cfg,
            objs=self.obj_manager.objectives,
            preprocessor=cfg.preprocessing_pipeline,
            param_space=cfg.param_space,
        )
        self._acqf_wrapper = cfg.acquisition_function
        self._bounds = torch.tensor(
            self.dag_manager.param_space.bounds().T,
            dtype=torch.double,
        )
        self._sobol_samples = draw_sobol_samples(
            self._bounds,
            n=cfg.random_iter,
            q=1,
        ).squeeze(1)

    def optimize_space(self) -> Dict[str, any]:
        if self._burn_in_iterations > 0:
            # draw samples for parameter from random as starting point
            self._burn_in_iterations -= 1
            next_configurations = self._sobol_samples[self._burn_in_iterations]
        else:
            # get next best configuration from the acquisition function
            next_configurations = self.bo_loop()
        suggested_params = self.dag_manager.param_space.numpy_to_dict(
            next_configurations.numpy()
        )
        self.logger.info("Next suggested configurations: %s.", suggested_params)
        return suggested_params

    def observe_state(self, state: EnvState):
        # TODO: This needs simplifying
        super().observe_state(state)  # Keep this to compare to other methods
        self.historical_obs.add(state)

    def _bo_loop(self) -> Tensor:
        dag_surrogate_model = self.dag_manager.get_dag(historic_obs=self.historical_obs)
        acqf_candidate = self.acf(dag_surrogate_model)
        next_configurations = unnormalize(
            acqf_candidate.to("cpu"), bounds=dag_surrogate_model.bounds
        )
        return next_configurations

    def acf(self, model: BoTorchDAG) -> torch.Tensor:
        acqf = self._acqf_wrapper.build(
            model=model,
            observed_x=model.observed_inputs,
            observed_y=model.observed_targets,
        )
        # Allow morphing the size of the in-dimension based on what we found relevant
        # in the causal analysis
        in_dim = model.in_dimensions
        self._acqf_wrapper.opt_cfg.dim = in_dim
        candidates, _ = self._acqf_wrapper.optimize(acqf, self._bounds)

        return candidates.detach()
