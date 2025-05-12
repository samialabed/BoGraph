from dataclasses import asdict, dataclass
from typing import Dict

from dataclasses_json import dataclass_json
from sysgym import EnvMetrics

from autorocks.envs.postgres.benchmarks.benchbase.dao import BenchbaseResult
from autorocks.envs.postgres.env_system_metrics import PostgresSystemMetrics


@dataclass_json
@dataclass(frozen=True)
class PostgresMetrics(EnvMetrics):
    bench_metrics: BenchbaseResult
    system_metrics: PostgresSystemMetrics

    def as_flat_dict(self) -> Dict[str, any]:
        bench_res_dict = asdict(self.bench_metrics)
        metrics_dict = self.system_metrics.as_dict()

        return {**bench_res_dict, **metrics_dict}
