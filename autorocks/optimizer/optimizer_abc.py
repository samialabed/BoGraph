import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping

import numpy as np
from dataclasses_json import dataclass_json

import autorocks.data.loader.filenames_const as fn
from autorocks.envs.env_state import EnvState
from autorocks.optimizer.objective_manager import ObjectiveManager
from autorocks.optimizer.opt_configs import OptimizerConfig
from autorocks.project import ExperimentManager


@dataclass_json
@dataclass(frozen=True)
class TotalData:
    # TODO(distributed): This should be moved away from the optimizer responsibility
    parameters: List[np.ndarray] = field(default_factory=list)
    optimization_target: List[np.ndarray] = field(default_factory=list)

    def add(self, params: np.ndarray, objective: np.ndarray):
        self.parameters.append(params)
        self.optimization_target.append(objective)


class Optimizer(ABC):
    # TODO: separate the saving logic to somewhere else
    def __init__(self, cfg: OptimizerConfig):
        self.ctx = ExperimentManager()
        self.cfg = cfg
        self.param_space = cfg.param_space
        self.obj_manager = ObjectiveManager(cfg.opt_objectives)
        self.logger = self.ctx.logger
        self.observed_states = TotalData()

    @abstractmethod
    def optimize_space(self) -> Mapping[str, any]:
        """return parameters as suggested by the optimizer."""

    def _generate_defaults(self) -> Mapping[str, any]:
        """Generates default configuration for first iteration."""
        res = {}
        for param in self.param_space.parameters():
            res[param.name] = param.box.default
        return res

    def observe_state(self, state: EnvState):
        """Feed the new measurement to the optimizer to update prior."""
        # self.logger.debug(f"New observed measurement {state}")
        # TODO: This needs to stay until we finish experiments but lets remove it
        #  and relay on saving all state in memory/disk
        self._save_observation(state)
        # TODO: this needs to be moved away from here
        target_obj = self.obj_manager.measurement_to_obj_numpy(state.measurements)
        self.logger.debug("Target objective: %s", target_obj)
        self.observed_states.add(params=state.params.as_numpy(), objective=target_obj)

    def _save_observation(self, state: EnvState):
        self.logger.debug("Saving current_state to %s", self.ctx.results_dir)
        with open(
            self.ctx.results_dir / fn.ITERATION_SYS_PARAMS, "w"
        ) as param_fp, open(
            self.ctx.results_dir / fn.ITERATION_SYS_OBSERVATIONS, "w"
        ) as state_fp:
            param_fp.write(state.params.to_json())
            json.dump(
                obj=state.measurements.as_flat_dict(),
                fp=state_fp,
            )
