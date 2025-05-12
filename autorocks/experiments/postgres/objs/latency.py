from dataclasses import dataclass

from dataclasses_json import dataclass_json

from autorocks.envs.objective_dao import OptimizationObjective, OptMode
from autorocks.envs.postgres.env_metrics import PostgresMetrics


@dataclass_json
@dataclass(frozen=True)
class LatencyP99(OptimizationObjective):
    name: str = "latency_p99"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: PostgresMetrics) -> float:
        # https://www.eecs.umich.edu/courses/eecs470/OLD/w14/lectures/470L14W14.pdf
        return env_metric.bench_metrics.latency_p99
