import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.postgres.benchmarks.benchbase.available_benchmarks import (
    BenchmarkClass,
)
from autorocks.envs.postgres.schema import PostgresParametersCollection10
from autorocks.envs.postgres.schema.ottertune_best import OtterTuneParamSchema
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.postgres.custom_models.bograph_static import (
    bograph_static_postgres,
)
from autorocks.experiments.postgres.objs import LatencyP99
from autorocks.experiments.postgres.utils import create_benchmark_cfg

OBJECTIVE = [LatencyP99()]
available_benchmarks = {
    "ot_ycsb": lambda: create_benchmark_cfg(
        BenchmarkClass.YCSB, 18000  # 18m record, ~18gb
    ),
    "ot_wiki": lambda: create_benchmark_cfg(
        BenchmarkClass.WIKI, 100  # 100k articles  ∼20 Gb
    ),
    "ot_tpcc": lambda: create_benchmark_cfg(
        BenchmarkClass.TPC_C, 200
    ),  # 200 warehouse, ~18gb
    "ot_tpch": lambda: create_benchmark_cfg(BenchmarkClass.TPC_H, 10),  # ~10gb
    "noop": lambda: create_benchmark_cfg(BenchmarkClass.NOOP, 1),
}
available_params = {
    "10params": PostgresParametersCollection10,
}

available_optimizer = {
    "bograph": bograph_static_postgres,
    "OT_static": lambda param_schema: baseline_opt_cfg.default(
        param_schema=OtterTuneParamSchema(), opt_obj=OBJECTIVE
    ),
    "Default": lambda param_schema: baseline_opt_cfg.default(
        param_schema=param_schema, opt_obj=OBJECTIVE
    ),
    botorch_model.TurboModel.name: lambda param_schema: baseline_opt_cfg.turbo(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.TurboModel(param_schema.dimensions),
    ),
    botorch_model.DeepGPModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.DeepGPModel(),
    ),
    botorch_model.AdditiveModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.AdditiveModel(),
    ),
    botorch_model.SingleTaskModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.SingleTaskModel(),
    ),
    str(NNITuner.TPE): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        tuner_name=NNITuner.TPE,
    ),
    str(NNITuner.RANDOM): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        tuner_name=NNITuner.RANDOM,
    ),
    str(NNITuner.SMAC): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        tuner_name=NNITuner.SMAC,
    ),
    str(NNITuner.PBT): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        tuner_name=NNITuner.PBT,
    ),
}
