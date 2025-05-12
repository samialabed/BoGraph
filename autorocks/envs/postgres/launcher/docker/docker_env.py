import logging
from time import sleep
from typing import Optional

import docker
from docker.errors import NotFound
from docker.models.containers import Container
from sysgym import EnvParamsDict
from tenacity import retry, stop_after_attempt, wait_random_exponential

from autorocks.envs.postgres.default_settings import docker_launcher_default
from autorocks.envs.postgres.launcher.docker.docker_cfgs import (
    DockerizedPostgresSettings,
)
from autorocks.envs.postgres.launcher.launcher_abc import PostgresLauncher
from autorocks.logging_util import ENV_RUNNER_LOGGER
from autorocks.project import ExperimentManager

LOG = logging.getLogger(ENV_RUNNER_LOGGER)


class DockerizedPostgres(PostgresLauncher):
    postgres_container: Container  # we used postgres 14.2

    def __init__(
        self,
        params: EnvParamsDict,
        cfgs: Optional[DockerizedPostgresSettings] = None,
    ):
        if cfgs is None:
            cfgs = docker_launcher_default()
        super().__init__(params, cfgs)
        self._ctx = ExperimentManager()
        self.cfgs = cfgs

        self._docker_cli = docker.from_env()
        self._container_name = cfgs.container_name

        try:
            gem5_container = self._docker_cli.containers.get(self._container_name)
            gem5_container.kill()
            gem5_container.remove()
        except NotFound:
            # expected behavior: no conflicting container exist
            pass
        except Exception as e:
            LOG.error(
                "Calling docker failed for container name: %s due to: %s",
                self._container_name,
                e,
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60)
    )
    def start(self, params: EnvParamsDict):
        startup_cmd = self.postgres_params_dict_to_system_args(params=params)
        self.postgres_container = self._docker_cli.containers.run(
            name=self.cfgs.container_name,
            hostname=self.cfgs.hostname,
            image=self.cfgs.image,
            command=f"postgres -N 500 {startup_cmd}",
            environment=self.cfgs.env_var.asdict(),
            ports=self.cfgs.docker_ports,
            detach=True,
            tty=True,
            remove=True,
        )
        # Sleep to give time for postgres to start
        sleep(10)

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60)
    )
    def kill(self):
        try:
            LOG.debug("Terminating docker instance and cleaning system")
            self.postgres_container.kill()
            self._docker_cli.close()
            sleep(10)  # give docker sometime to kill the instance
        except docker.errors.NotFound:
            LOG.info("Attempted to close a container but was already closed")
