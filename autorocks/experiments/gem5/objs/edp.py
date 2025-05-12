from dataclasses import dataclass

from dataclasses_json import dataclass_json
from sysgym.envs.gem5.env_measure import Gem5Metrics

from autorocks.envs.objective_dao import OptimizationObjective, OptMode


@dataclass_json
@dataclass(frozen=True)
class EDPObjective(OptimizationObjective):
    name: str = "edp"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: Gem5Metrics) -> float:
        # https://www.eecs.umich.edu/courses/eecs470/OLD/w14/lectures/470L14W14.pdf
        return env_metric.log_epd
