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
class BytesPerWriteObjective(OptimizationObjective):
    name: str = "bytes_per_write"
    opt_mode: OptMode = OptMode.maximize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        avg_bytes_per_write = (
            env_metric.bench_stats.statistics.rocksdb_bytes_per_write.mean
        )
        LOG.info("Average avg_bytes_per_write throughput %f", avg_bytes_per_write)
        return avg_bytes_per_write
