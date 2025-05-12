#!/bin/env python
import os

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.postgres.benchmarks.benchbase.available_benchmarks import (
    BenchmarkClass,
)
from autorocks.envs.postgres.benchmarks.benchbase.cfg import BenchbaseCFG
from autorocks.envs.postgres.default_settings import envvar_benchbase_default
from autorocks.envs.postgres.env_cfg import PostgresEnvConfig
from autorocks.envs.postgres.launcher.docker.docker_cfgs import (
    DockerizedPostgresSettings,
)
from autorocks.envs.postgres.schema import PostgresParametersCollection10
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.postgres.custom_models.bograph_static import (
    bograph_static_postgres,
)
from autorocks.experiments.postgres.objs import LatencyP99
from autorocks.experiments.postgres.utils import create_benchmark_cfg
from autorocks.optimizer.default.default_optimizer_cfg import DefaultConfig
from scripts.exp_runners.experiments_runner import (
    ExperimentsRunner,
    ExperimentsSkip,
    ExpUID,
)

__POSTGRES_ADDR = os.getenv("POSTGRES_ADDR", "localhost")
print(f"Using postgres on address: {__POSTGRES_ADDR}")
if __POSTGRES_ADDR != "localhost":
    os.environ["DOCKER_HOST"] = f"tcp://{__POSTGRES_ADDR}:2376"
    os.environ["DOCKER_TLS_VERIFY"] = ""


def experiment_config() -> ExperimentsRunner:
    # hardcoded parameters for easier re-generating of results
    exp_to_skip = ExperimentsSkip(
        skip_exp={
            ExpUID(bench_task=BenchmarkClass.YCSB, optimizer_name="Default"),
            ExpUID(bench_task=BenchmarkClass.YCSB, optimizer_name=str(NNITuner.RANDOM)),
            ExpUID(bench_task=BenchmarkClass.YCSB, optimizer_name=str(NNITuner.TPE)),
            ExpUID(bench_task=BenchmarkClass.YCSB, optimizer_name=str(NNITuner.PBT)),
        },
        skip_optimizer={
            "Default",  # done
            "OT_Static",  # done
            botorch_model.DeepGPModel.name,  # not worth the gpu hours
            botorch_model.AdditiveModel.name,  # suboptimal
            str(NNITuner.SMAC),  # doesn't work for large param space
        },
        skip_bench={
            BenchmarkClass.NOOP,
            BenchmarkClass.TPC_C,
            BenchmarkClass.TPC_H,
            BenchmarkClass.WIKI,
        },
    )
    __OBJECTIVE = [LatencyP99()]
    __ITERATIONS = 100
    __PARAM_SPACE = PostgresParametersCollection10()
    __LAUNCHER_SETTINGS = DockerizedPostgresSettings(
        ip_addr=__POSTGRES_ADDR,
        port=5432,
        container_name="postgres",
        hostname="postgres",
        image="postgres:alpine",
        docker_ports={"5432": "5432"},
        env_var=envvar_benchbase_default(),
    )

    def env(bench: BenchbaseCFG) -> PostgresEnvConfig:
        return PostgresEnvConfig(launcher_settings=__LAUNCHER_SETTINGS, bench_cfg=bench)

    benchmarks_to_evaluate = {
        BenchmarkClass.YCSB: lambda t: create_benchmark_cfg(
            t, 18000  # 18m record, ~18gb
        ),
        BenchmarkClass.WIKI: lambda t: create_benchmark_cfg(
            t, 100  # 100k articles  ∼20 Gb
        ),
        BenchmarkClass.TPC_C: lambda t: create_benchmark_cfg(
            t, 200
        ),  # 200 warehouse, ~18gb
        BenchmarkClass.TPC_H: lambda t: create_benchmark_cfg(t, 10),  # ~10gb
        BenchmarkClass.NOOP: lambda t: create_benchmark_cfg(t, 1),
    }

    optimizers_to_evaluate = {
        "bograph": lambda: bograph_static_postgres(param_space=__PARAM_SPACE),
        "OT_Static": lambda: DefaultConfig(
            param_space=__PARAM_SPACE, opt_objectives=__OBJECTIVE, name="OtterTune"
        ),
        "Default": lambda: DefaultConfig(
            param_space=__PARAM_SPACE, opt_objectives=__OBJECTIVE
        ),
        botorch_model.DeepGPModel.name: lambda: baseline_opt_cfg.botorch(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            surrogate_model=botorch_model.DeepGPModel(),
        ),
        botorch_model.AdditiveModel.name: lambda: baseline_opt_cfg.botorch(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            surrogate_model=botorch_model.AdditiveModel(),
        ),
        botorch_model.SingleTaskModel.name: lambda: baseline_opt_cfg.botorch(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            surrogate_model=botorch_model.SingleTaskModel(),
        ),
        str(NNITuner.TPE): lambda: baseline_opt_cfg.nni_opt(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.TPE,
        ),
        str(NNITuner.RANDOM): lambda: baseline_opt_cfg.nni_opt(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.RANDOM,
        ),
        str(NNITuner.SMAC): lambda: baseline_opt_cfg.nni_opt(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.SMAC,
        ),
        str(NNITuner.PBT): lambda: baseline_opt_cfg.nni_opt(
            param_schema=__PARAM_SPACE,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.PBT,
        ),
    }

    return ExperimentsRunner(
        iterations=__ITERATIONS,
        optimizer_to_evaluate=optimizers_to_evaluate,
        benchmarks_to_evaluate=benchmarks_to_evaluate,
        env_callable=env,
        exp_skip=exp_to_skip,
    )


if __name__ == "__main__":
    exp_runner_cfg = experiment_config()
    exp_runner_cfg.execute()
