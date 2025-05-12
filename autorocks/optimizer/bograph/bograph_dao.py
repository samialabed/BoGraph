from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from autorocks.envs.env_state import EnvState
from autorocks.optimizer.objective_manager import ObjectiveManager


@dataclass
class BoGraphDataPandas:
    params: pd.DataFrame
    objs: pd.DataFrame
    intermediate: pd.DataFrame

    def param_objs_df(self) -> pd.DataFrame:
        return pd.concat([self.params, self.objs], axis=1)

    def param_intermediates_df(self) -> pd.DataFrame:
        return pd.concat([self.params, self.intermediate], axis=1)

    def intermediates_objs_df(self) -> pd.DataFrame:
        return pd.concat([self.intermediate, self.objs], axis=1)

    def to_combi_pandas(self) -> pd.DataFrame:
        return pd.concat([self.params, self.objs, self.intermediate], axis=1)

    def copy(self):
        return BoGraphDataPandas(
            params=self.params.copy(),
            objs=self.objs.copy(),
            intermediate=self.intermediate.copy(),
        )


class BoGraphIntermediateData:
    parameters: List[Dict]
    optimization_target: List[Dict]
    intermediate_metrics: List[Dict]

    def __init__(self, obj_manager: ObjectiveManager):
        self.parameters = []
        self.optimization_target = []
        self.intermediate_metrics = []
        self._obj_manager = obj_manager

    def add(self, state: EnvState):
        self.parameters.append(dict(state.params))
        # Add objectives and measurements
        measurements = state.measurements
        self.optimization_target.append(
            self._obj_manager.measurement_to_obj_val_dict(measurements)
        )
        measurement_dict = measurements.as_flat_dict()
        # Make sure the objective does not appear in the intermediate value
        for obj in self._obj_manager.objectives:
            if obj in measurement_dict:
                del measurement_dict[obj]
        self.intermediate_metrics.append(measurement_dict)

    def to_pandas(self) -> BoGraphDataPandas:
        pandas = BoGraphDataPandas(
            params=pd.DataFrame(self.parameters),
            objs=pd.DataFrame(self.optimization_target),
            intermediate=pd.DataFrame(self.intermediate_metrics).fillna(0),
        )
        return pandas
