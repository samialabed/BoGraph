from dataclasses import dataclass, field

from dataclasses_json import config, dataclass_json
from sysgym import EnvConfig

from autorocks.envs.postgres.benchmarks.benchbase.cfg import BenchbaseCFG
from autorocks.envs.postgres.launcher.launcher_cfgs import PostgresLauncherSettings


@dataclass_json
@dataclass(frozen=True)
class PostgresEnvConfig(EnvConfig):
    bench_cfg: BenchbaseCFG
    launcher_settings: PostgresLauncherSettings = field(
        metadata=config(encoder=lambda x: "Sensitive not saved")
    )
    # number of time to rerun the environment and average the results
    repeat_evaluation: int = 1

    @property
    def name(self) -> str:
        return "postgres"
