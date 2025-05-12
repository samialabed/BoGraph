from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json
from sysgym.envs.rocksdb.env_measure import RocksDBMeasurements

from autorocks.envs.objective_dao import OptimizationObjective, OptMode


@dataclass_json
@dataclass(frozen=True)
class MemObjective(OptimizationObjective):
    name: str = "mem_p99"
    opt_mode: OptMode = OptMode.maximize

    def extract_obj(self, env_metric: RocksDBMeasurements) -> Union[float, int]:
        return env_metric.sysio.mem_usage.p95
