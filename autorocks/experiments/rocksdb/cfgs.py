import networkx as nx
import torch
from sysgym.envs.rocksdb.schema import RocksDB10Params, RocksDB17Params

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.rocksdb import objs
from autorocks.global_flags import DEVICE
from autorocks.optimizer.bograph import dag_options

MT_OBJECTIVES = [
    objs.CompactionObjective(),
    objs.DBGetObjective(),
    objs.BytesPerWriteObjective(),
    objs.ThroughputObjective(),
]

OBJECTIVE = [objs.ThroughputObjective()]
available_params = {
    "10param": RocksDB10Params,
    "17param": RocksDB17Params,
}


def _rocksdb_expert_10param() -> nx.DiGraph:
    expert = nx.DiGraph()

    expert.add_edges_from(
        [
            ("write_buffer_size", "db_write_micros.p95"),
            ("max_write_buffer_number", "db_write_micros.p95"),
            ("min_write_buffer_number_to_merge", "db_write_micros.p95"),
            ("block_size", "db_write_micros.p95"),
            ("db_write_micros.p95", "iops"),
            ("level0_file_num_compaction_trigger", "compaction_times_micros.p95"),
            ("level0_slowdown_writes_trigger", "compaction_times_micros.p95"),
            ("level0_stop_writes_trigger", "compaction_times_micros.p95"),
            ("max_bytes_for_level_multiplier", "compaction_times_micros.p95"),
            ("max_background_compactions", "compaction_times_micros.p95"),
            ("max_background_flushes", "compaction_times_micros.p95"),
            ("compaction_times_micros.p95", "iops"),
            ("write_buffer_size", "iops"),
        ]
    )
    return expert


available_optimizer = {
    "Default": lambda param_schema: baseline_opt_cfg.default(
        param_schema=param_schema, opt_obj=OBJECTIVE
    ),
    "bobn": lambda param_schema: dag_options.BoBnConfig(
        name="BoBn",
        param_space=param_schema,
        opt_objectives=OBJECTIVE,
        random_iter=10,
        retry=3,
        dag=_rocksdb_expert_10param(),
        use_turbo=False,
        conservative_mode=True,
    ),
    "turbo_bobn": lambda param_schema: dag_options.BoBnConfig(
        name="TurboBoBn",
        param_space=param_schema,
        opt_objectives=OBJECTIVE,
        random_iter=10,
        retry=3,
        dag=_rocksdb_expert_10param(),
        use_turbo=True,
        conservative_mode=True,
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
    f"{botorch_model.MultiTaskModel.name}_ALL": lambda param_schema: baseline_opt_cfg.botorch_mt_all(
        param_schema=param_schema,
        opt_obj=MT_OBJECTIVES,
        surrogate_model=botorch_model.MultiTaskModel(),
        weighting=torch.tensor(
            [0.4, 1.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE
        ),
    ),
    f"{botorch_model.MultiTaskModel.name}_IOPS_ONLY": lambda param_schema: baseline_opt_cfg.botorch_mt_all(
        param_schema=param_schema,
        opt_obj=MT_OBJECTIVES,
        surrogate_model=botorch_model.MultiTaskModel(),
        weighting=torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=DEVICE
        ),
    ),
    botorch_model.TurboModel.name: lambda param_schema: baseline_opt_cfg.turbo(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.TurboModel(param_schema.dimensions),
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
