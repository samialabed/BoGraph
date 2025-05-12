from typing import Type

from autorocks.envs.postgres.launcher.docker.docker_cfgs import (
    DockerizedPostgresSettings,
)
from autorocks.envs.postgres.launcher.launcher_abc import PostgresLauncher
from autorocks.envs.postgres.launcher.launcher_cfgs import PostgresLauncherSettings


def get_launcher(
    launcher_cfgs: PostgresLauncherSettings,
) -> Type[PostgresLauncher]:
    if isinstance(launcher_cfgs, DockerizedPostgresSettings):
        from autorocks.envs.postgres.launcher.docker.docker_env import (
            DockerizedPostgres,
        )

        return DockerizedPostgres

    else:
        raise Exception(f"Unknown launcher setting provided: {launcher_cfgs}")
