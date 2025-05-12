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
class ThroughputObjective(OptimizationObjective):
    name: str = "iops"
    opt_mode: OptMode = OptMode.maximize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        iops = []
        for db_bench in env_metric.bench_stats.db_bench.values():
            iops.append(db_bench.iops)
        avg_iops = sum(iops) / len(iops)
        LOG.info("Average IOPS throughput %f", avg_iops)
        return avg_iops
