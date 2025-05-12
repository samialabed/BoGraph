#!/bin/env python
from sysgym.envs.gem5.benchmarks.benchmark_tasks import MachSuiteTask
from sysgym.envs.gem5.schema import AladdinSweeper20Params

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.gem5.custom_models.bograph_static import (
    mobograph_static_pwr_lat,
)
from autorocks.experiments.gem5.objs import LatencyObjective, PowerObjective
from autorocks.experiments.gem5.utils import gem5_benchmark_plan, gem5_env
from scripts.exp_runners.experiments_runner import ExperimentsRunner, ExperimentsSkip


def experiment_config() -> ExperimentsRunner:
    exp_to_skip = ExperimentsSkip(
        skip_exp=set(),
        # skip_optimizer={"mobograph"},
        skip_bench={
            # Skip these three for taking too long
            MachSuiteTask.AES,
            MachSuiteTask.FFT_STRIDED,
            MachSuiteTask.FFT_TRANSPOSE,
            MachSuiteTask.GEMMA_NCUBED,
            MachSuiteTask.STENCIL_3D,
            # MachSuiteTask.FFT_TRANSPOSE,
        },
    )
    objective = [
        LatencyObjective(),
        PowerObjective(),
    ]
    param_schema = AladdinSweeper20Params()
    env = gem5_env
    iterations = 100

    optimizers_to_evaluate = {
        # "MOBOSingleTask": lambda: baseline_opt_cfg.mobo_botorch(
        #     param_schema=param_schema,
        #     opt_obj=objective,
        #     surrogate_model=botorch_model.SingleTaskModel(),
        # ),
        "mobograph": lambda: mobograph_static_pwr_lat(param_space=param_schema),
    }

    benchmarks_to_evaluate = {
        # MachSuiteTask.AES: gem5_benchmark_plan,
        # MachSuiteTask.FFT_TRANSPOSE: gem5_benchmark_plan,
        # MachSuiteTask.STENCIL_3D: gem5_benchmark_plan,
        # MachSuiteTask.GEMMA_NCUBED: gem5_benchmark_plan,
        # MachSuiteTask.STENCIL_2D: gem5_benchmark_plan,
        # MachSuiteTask.FFT_STRIDED: gem5_benchmark_plan,
        # MachSuiteTask.SPMV_CRS: gem5_benchmark_plan,
        MachSuiteTask.SPMV_ELLPACK: gem5_benchmark_plan,
        # MachSuiteTask.MD_KNN: gem5_benchmark_plan,
    }

    return ExperimentsRunner(
        iterations=iterations,
        optimizer_to_evaluate=optimizers_to_evaluate,
        benchmarks_to_evaluate=benchmarks_to_evaluate,
        env_callable=env,
        exp_skip=exp_to_skip,
    )


if __name__ == "__main__":
    exp_runner_cfg = experiment_config()
    exp_runner_cfg.execute()
