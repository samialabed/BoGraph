import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.synthetic.synth_objective_dao import TargetFuncObjective
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg

OBJECTIVE = [TargetFuncObjective()]

SIM_COMMON_OPTIMIZER = {
    botorch_model.TurboModel.name: lambda param_schema: baseline_opt_cfg.turbo(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.TurboModel(dim=param_schema.dimensions),
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
