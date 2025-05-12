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
class CompactionObjective(OptimizationObjective):
    name: str = "compaction_outfile_sync"
    opt_mode: OptMode = OptMode.maximize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        compaction_outfile_sync_avg = (
            env_metric.bench_stats.statistics.rocksdb_compaction_outfile_sync_micros.mean
        )
        LOG.info(
            "Average compaction_outfile_sync throughput %f", compaction_outfile_sync_avg
        )
        return compaction_outfile_sync_avg


@dataclass_json
@dataclass(frozen=True)
class DBGetObjective(OptimizationObjective):
    name: str = "db_get"
    opt_mode: OptMode = OptMode.minimize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        avg_db_get = env_metric.bench_stats.statistics.rocksdb_db_get_micros.mean
        LOG.info("Average avg_db_get throughput %f", avg_db_get)
        return avg_db_get
