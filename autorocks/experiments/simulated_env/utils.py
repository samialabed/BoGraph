import logging
from argparse import ArgumentParser
from typing import List

from autorocks.execution.manager import ExecutionPlan
from autorocks.logging_util import ENV_RUNNER_LOGGER

LOG = logging.getLogger(ENV_RUNNER_LOGGER)


def sim_env_cli(available_optimizer: List[str]):
    parser = ArgumentParser(description="Run Simulated env case-study")
    parser.add_argument(
        "--execution_plan",
        type=ExecutionPlan,
        help="Choose the backend manager of the optimizer learner",
        choices=list(ExecutionPlan),
        default=ExecutionPlan.LocalExecution,
    )
    parser.add_argument(
        "--opt",
        type=str,
        help="Name of the optimizer to use",
        choices=available_optimizer,
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
        LOG.info("Overriding output directory to %s", cli_options.exp_dir)
    if cli_options.debug:
        LOG.info("Enabling Debug mode")
    return cli_options
