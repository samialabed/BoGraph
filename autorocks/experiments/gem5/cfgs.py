from sysgym.envs.gem5.schema import AladdinSweeper20Params

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.gem5.custom_models.bograph_auto import bograph_causal_edp
from autorocks.experiments.gem5.custom_models.bograph_static import (
    bograph_static_edp,
    mobograph_static_pwr_lat,
)
from autorocks.experiments.gem5.objs.edp import EDPObjective
from autorocks.experiments.gem5.objs.latency import LatencyObjective
from autorocks.experiments.gem5.objs.power import PowerObjective

EDP_OBJ = [EDPObjective()]
available_optimizer_edp = {
    "bograph_causal_edp": bograph_causal_edp,
    "bograph_static_edp": bograph_static_edp,
    "Default": lambda param_schema: baseline_opt_cfg.default(
        param_schema=param_schema, opt_obj=EDP_OBJ
    ),
    botorch_model.DeepGPModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        surrogate_model=botorch_model.DeepGPModel(),
    ),
    botorch_model.AdditiveModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        surrogate_model=botorch_model.AdditiveModel(),
    ),
    botorch_model.SingleTaskModel.name: lambda param_schema: baseline_opt_cfg.botorch(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        surrogate_model=botorch_model.SingleTaskModel(),
    ),
    str(NNITuner.TPE): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        tuner_name=NNITuner.TPE,
    ),
    str(NNITuner.RANDOM): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        tuner_name=NNITuner.RANDOM,
    ),
    str(NNITuner.SMAC): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        tuner_name=NNITuner.SMAC,
    ),
    str(NNITuner.PBT): lambda param_schema: baseline_opt_cfg.nni_opt(
        param_schema=param_schema,
        opt_obj=EDP_OBJ,
        tuner_name=NNITuner.PBT,
    ),
}

LAT_ENG_OBJ = [
    LatencyObjective(),
    PowerObjective(),
]

mobo_botorch = f"MOBO{botorch_model.SingleTaskModel.name}"
available_optimizer_lat_eng = {
    # "bograph_causal_edp": bograph_causal_edp,
    mobo_botorch: lambda param_schema: baseline_opt_cfg.mobo_botorch(
        param_schema=param_schema,
        opt_obj=LAT_ENG_OBJ,
        surrogate_model=botorch_model.SingleTaskModel(),
    ),
    "mobograph": mobograph_static_pwr_lat,
}

available_optimizer = {**available_optimizer_edp, **available_optimizer_lat_eng}

available_params = {
    "20param": AladdinSweeper20Params,
}
