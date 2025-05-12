from typing import Dict, List, Set

import numpy as np
from sysgym import EnvMetrics

from autorocks.envs.objective_dao import OptimizationObjective


class ObjectiveManager:
    def __init__(self, optimization_objective: List[OptimizationObjective]):
        self.optimization_objective = optimization_objective

    @property
    def objectives(self) -> Set[str]:
        return {obj.name for obj in self.optimization_objective}

    def measurement_to_obj_val_dict(self, measurement: EnvMetrics) -> Dict[str, float]:
        res = {}
        for objective in self.optimization_objective:
            res[objective.name] = objective.extract_opt_target_with_sign(measurement)
        return res

    def measurement_to_obj_numpy(self, measurement: EnvMetrics) -> np.ndarray:
        """Populates the objectives from env measurement.
        avoid copying the measurement object as it consumes large memory."""
        out = []
        for objective in self.optimization_objective:
            out.append(objective.extract_opt_target_with_sign(measurement))

        return np.array(out)

    def reverse_transform(self, state: np.ndarray) -> Dict[str, float]:
        out = {}
        for i, objective in enumerate(self.optimization_objective):
            out[objective.name] = state[:, i]

        return out

    def reference_points(self) -> List[float]:
        """TODO: For MOBO, quickly glued in for the paper, we need massive refactor"""
        return [x.reference_point for x in self.optimization_objective]
