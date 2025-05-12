import logging
import os
from argparse import ArgumentParser

from autorocks.envs.postgres.default_settings import docker_launcher_default
from autorocks.envs.postgres.env_cfg import PostgresEnvConfig
from autorocks.execution.manager import ExecutionPlan, runner_manager
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.experiments.postgres.cfgs import (
    available_benchmarks,
    available_optimizer,
    available_params,
)
from autorocks.logging_util import ENV_RUNNER_LOGGER

LOG = logging.getLogger(ENV_RUNNER_LOGGER)

# ugliest hack ever
postgres_addr = os.getenv("POSTGRES_ADDR", "localhost")
print(f"Using postgress address: {postgres_addr}")
if postgres_addr != "localhost":
    os.environ["DOCKER_HOST"] = f"tcp://{postgres_addr}:2376"
    os.environ["DOCKER_TLS_VERIFY"] = ""

if __name__ == "__main__":
    parser = ArgumentParser(description="Run postgres case-study")
    parser.add_argument(
        "--execution_plan",
        type=ExecutionPlan,
        help="Choose the backend manager of the optimizer learner",
        choices=list(ExecutionPlan),
        default=ExecutionPlan.LocalExecution,
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Name of the parameters preset to run against",
        choices=available_params,
        required=True,
    )
    parser.add_argument(
        "--opt",
        type=str,
        help="Name of the optimizer to use",
        choices=available_optimizer,
        required=True,
    )
    parser.add_argument(
        "--bench",
        type=str,
        help="Name of the benchmark to use",
        choices=available_benchmarks,
        required=True,
    )

    parser.add_argument(
        "--iter",
        help="Number of iterations to run the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--debug", help="Run in debug mode and create more logs", action="store_true"
    )

    parser.add_argument(
        "--exp_dir",
        help="Output path for the experiment.",
        type=str,
        required=False,
    )

    cli_options = parser.parse_args()

    if cli_options.exp_dir:
        LOG.info(f"Overriding output directory to {cli_options.exp_dir}")
    if cli_options.debug:
        LOG.info(f"Enabling Debug mode")

    bench_cfg = available_benchmarks[cli_options.bench]()
    env_cfg = PostgresEnvConfig(
        launcher_settings=docker_launcher_default(), bench_cfg=bench_cfg
    )
    param_schema = available_params[cli_options.params]()
    opt_cfg = available_optimizer[cli_options.opt](param_schema)

    config = ExperimentConfigs(
        iterations=cli_options.iter,
        opt_cfg=opt_cfg,
        env_cfg=env_cfg,
        debug=cli_options.debug,
        exp_dir=cli_options.exp_dir,
    )
    runner_manager(config=config, execution_plan=cli_options.execution_plan)
