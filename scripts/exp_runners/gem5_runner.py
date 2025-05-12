#!/bin/env python
from sysgym.envs.gem5.benchmarks.benchmark_tasks import MachSuiteTask
from sysgym.envs.gem5.schema import AladdinSweeper20Params

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.gem5.custom_models.bograph_auto import bograph_causal_edp
from autorocks.experiments.gem5.custom_models.bograph_static import bograph_static_edp
from autorocks.experiments.gem5.objs import EDPObjective
from autorocks.experiments.gem5.utils import gem5_benchmark_plan, gem5_env
from scripts.exp_runners.experiments_runner import ExperimentsRunner, ExperimentsSkip


def experiment_config() -> ExperimentsRunner:
    # hardcoded parameters for easier re-generating of results
    exp_to_skip = ExperimentsSkip(
        skip_exp=set(),
        skip_optimizer=set(),
        skip_bench={
            MachSuiteTask.STENCIL_3D,
            MachSuiteTask.GEMMA_NCUBED,
            MachSuiteTask.STENCIL_2D,
            MachSuiteTask.FFT_STRIDED,
            MachSuiteTask.SPMV_CRS,
            MachSuiteTask.SPMV_ELLPACK,
            MachSuiteTask.MD_KNN,
        },
    )
    objective = [EDPObjective()]
    param_schema = AladdinSweeper20Params()
    env = gem5_env
    iterations = 100

    optimizers_to_evaluate = {
        "bograph_causal_edp": lambda: bograph_causal_edp(param_space=param_schema),
        "bograph_static_edp": lambda: bograph_static_edp(param_space=param_schema),
        "Default": lambda: baseline_opt_cfg.default(
            param_schema=param_schema, opt_obj=objective
        ),
        "DeepGP": lambda: baseline_opt_cfg.botorch(
            param_schema=param_schema,
            opt_obj=objective,
            surrogate_model=botorch_model.DeepGPModel(),
        ),
        "Additive": lambda: baseline_opt_cfg.botorch(
            param_schema=param_schema,
            opt_obj=objective,
            surrogate_model=botorch_model.AdditiveModel(),
        ),
        "SingleTask": lambda: baseline_opt_cfg.botorch(
            param_schema=param_schema,
            opt_obj=objective,
            surrogate_model=botorch_model.SingleTaskModel(),
        ),
        str(NNITuner.TPE): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=objective,
            tuner_name=NNITuner.TPE,
        ),
        str(NNITuner.RANDOM): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=objective,
            tuner_name=NNITuner.RANDOM,
        ),
        str(NNITuner.SMAC): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=objective,
            tuner_name=NNITuner.SMAC,
        ),
        str(NNITuner.PBT): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=objective,
            tuner_name=NNITuner.PBT,
        ),
    }

    benchmarks_to_evaluate = {
        MachSuiteTask.AES: gem5_benchmark_plan,
        MachSuiteTask.FFT_TRANSPOSE: gem5_benchmark_plan,
        MachSuiteTask.STENCIL_3D: gem5_benchmark_plan,
        MachSuiteTask.GEMMA_NCUBED: gem5_benchmark_plan,
        MachSuiteTask.STENCIL_2D: gem5_benchmark_plan,
        MachSuiteTask.FFT_STRIDED: gem5_benchmark_plan,
        MachSuiteTask.SPMV_CRS: gem5_benchmark_plan,
        MachSuiteTask.SPMV_ELLPACK: gem5_benchmark_plan,
        MachSuiteTask.MD_KNN: gem5_benchmark_plan,
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
