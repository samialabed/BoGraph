import logging
from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json
from sysgym.envs.rocksdb.env_measure import RocksDBMeasurements

from autorocks.envs.objective_dao import OptimizationObjective, OptMode
from autorocks.logging_util import ENV_RUNNER_LOGGER

LOG = logging.getLogger(ENV_RUNNER_LOGGER)


@dataclass_json
@dataclass(frozen=True)
class LatencyObjective(OptimizationObjective):
    name: str = "latency"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        latency = []

        for db_bench in env_metric.bench_stats.db_bench.values():
            latency.append(db_bench.latency)
        avg_latency = sum(latency) / len(latency)
        LOG.info("Average latency %f", avg_latency)
        return avg_latency
