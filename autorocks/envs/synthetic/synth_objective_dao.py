from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json

from autorocks.envs.objective_dao import OptimizationObjective, OptMode
from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements

# TODO: we need to overhaul objective optimization


@dataclass_json
@dataclass(frozen=True)
class TargetFuncObjective(OptimizationObjective):
    name: str = "target"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: TestFunctionMeasurements) -> Union[float, int]:
        return env_metric.target


@dataclass_json
@dataclass(frozen=True)
class BraninMOBOFuncObjective(OptimizationObjective):
    name: str = "branin"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: TestFunctionMeasurements) -> Union[float, int]:
        return env_metric.structure["branin"]


@dataclass_json
@dataclass(frozen=True)
class CurrinMOBOFuncObjective(OptimizationObjective):
    name: str = "currin"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: TestFunctionMeasurements) -> Union[float, int]:
        return env_metric.structure["currin"]
