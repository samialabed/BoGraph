from autorocks.exp_cfg import ExperimentConfigs
from autorocks.utils.enum import ExtendedEnum


class LauncherException(BaseException):
    pass


class ExecutionPlan(ExtendedEnum):
    LocalExecution = "local"
    NNI = "nni"
    Distributed = "unimplemented"  # TODO(Distributed+Docker)


def runner_manager(
    config: ExperimentConfigs,
    execution_plan: ExecutionPlan,
):

    if execution_plan == ExecutionPlan.NNI:

        from autorocks.execution.nni_launcher.nni_exp import nni_experiment

        nni_experiment(cfg=config)
    elif execution_plan == ExecutionPlan.LocalExecution:
        from autorocks.execution.local_loop import runner_loop

        runner_loop(cfg=config)
    else:
        raise LauncherException(f"Unsupported execution plan: {execution_plan}")
