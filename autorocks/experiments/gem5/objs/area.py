from dataclasses import dataclass

from dataclasses_json import dataclass_json
from sysgym.envs.gem5.env_measure import Gem5Metrics

from autorocks.envs.objective_dao import OptimizationObjective, OptMode


@dataclass_json
@dataclass(frozen=True)
class AreaObjective(OptimizationObjective):
    name: str = "area"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: Gem5Metrics) -> float:
        return env_metric.summary_stats.total_area
