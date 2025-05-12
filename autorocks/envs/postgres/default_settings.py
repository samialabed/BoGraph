from autorocks.envs.postgres.launcher.docker.docker_cfgs import (
    DockerizedPostgresSettings,
)
from autorocks.envs.postgres.launcher.launcher_cfgs import PostgresEnvVar


def envvar_benchbase_default() -> PostgresEnvVar:
    return PostgresEnvVar(
        POSTGRES_USER="admin", POSTGRES_PASSWORD="password", POSTGRES_DB="benchbase"
    )


def docker_launcher_default() -> DockerizedPostgresSettings:
    return DockerizedPostgresSettings(
        ip_addr="localhost",
        port=5432,
        container_name="postgres",
        hostname="postgres",
        image="postgres:alpine",
        docker_ports={"5432": "5432"},
        env_var=envvar_benchbase_default(),
    )
