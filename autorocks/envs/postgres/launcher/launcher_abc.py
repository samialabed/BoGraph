from abc import ABC, abstractmethod
from typing import Optional

from sysgym import EnvParamsDict

from autorocks.envs.postgres.launcher.launcher_cfgs import PostgresLauncherSettings


class PostgresLauncher(ABC):
    def __init__(
        self,
        params: EnvParamsDict,
        cfgs: Optional[PostgresLauncherSettings] = None,
    ):
        self.cfgs = cfgs
        self._params = params

    def __enter__(self):
        self.start(self._params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()

    @staticmethod
    def postgres_params_dict_to_system_args(params: EnvParamsDict) -> str:
        params_dict = dict(params)

        extra_tracking_params = {
            "track_counts": "on",
            "track_functions": "all",
            "track_io_timing": "on",
            "track_wal_io_timing": "on",
            "autovacuum": "off",
        }

        params_with_tracking = {**params_dict, **extra_tracking_params}
        return " ".join([f"-c {k}={v}" for (k, v) in params_with_tracking.items()])

    @abstractmethod
    def start(self, params: EnvParamsDict):
        pass

    @abstractmethod
    def kill(self):
        pass
