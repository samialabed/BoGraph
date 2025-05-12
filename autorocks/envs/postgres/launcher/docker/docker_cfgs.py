from dataclasses import dataclass
from typing import Dict

from autorocks.envs.postgres.launcher.launcher_cfgs import PostgresLauncherSettings


@dataclass(frozen=True)
class DockerizedPostgresSettings(PostgresLauncherSettings):
    container_name: str
    hostname: str
    image: str
    docker_ports: Dict[str, str]
