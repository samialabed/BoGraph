from dataclasses import dataclass, field
from typing import List

from dataclasses_json import config, dataclass_json
from sysgym.params import ParamsSpace

from autorocks.envs.objective_dao import OptimizationObjective


@dataclass_json
@dataclass
class OptimizerConfig:
    opt_objectives: List[OptimizationObjective]
    param_space: ParamsSpace = field(
        metadata=config(encoder=ParamsSpace.spaces_to_json)
    )
    name: str
