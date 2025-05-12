#!/bin/env python
from sysgym.envs.rocksdb.benchmarks.dbbench import (
    established_benchmarks as dbbench_benchmarks,
)
from sysgym.envs.rocksdb.benchmarks.dbbench.established_benchmarks import DBBenchTasks
from sysgym.envs.rocksdb.env_cfg import RocksDBEnvConfig

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.rocksdb import cfgs
from scripts.exp_runners.experiments_runner import ExperimentsRunner, ExperimentsSkip


def experiment_config() -> ExperimentsRunner:
    # hardcoded parameters for easier re-generating of results
    exp_to_skip = ExperimentsSkip(
        skip_exp=set(),
        skip_optimizer=set(),
        skip_bench=set(),
    )
    param_schema = cfgs.available_params["10param"]
    env = RocksDBEnvConfig
    iterations = 100
    benchmarks_to_evaluate = {
        # dbbench_benchmarks.DBBenchTasks.READ_RANDOM_WRITE_RANDOM: lambda _: DBBenchTasks.READ_RANDOM_WRITE_RANDOM.get_plan()
        dbbench_benchmarks.DBBenchTasks.ZIPPY_WORKLOAD: lambda _: DBBenchTasks.ZIPPY_WORKLOAD.get_plan()
        # dbbench_benchmarks.DBBenchTasks.FAST_ZIPPY_WORKLOAD: lambda _: DBBenchTasks.FAST_ZIPPY_WORKLOAD.get_plan()
    }

    optimizers_to_evaluate = {}
    for k in [
        # "Default",
        # "bobn",
        "turbo_bobn",
        # botorch_model.SingleTaskModel.name,
        # f"{botorch_model.MultiTaskModel.name}_ALL",
        # f"{botorch_model.MultiTaskModel.name}_IOPS_ONLY"
        # botorch_model.TurboModel.name,
        # str(NNITuner.TPE),
        # str(NNITuner.RANDOM),
        # str(NNITuner.SMAC),
        # str(NNITuner.PBT),
    ]:
        optimizers_to_evaluate[k] = cfgs.available_optimizer[k](param_schema())

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
