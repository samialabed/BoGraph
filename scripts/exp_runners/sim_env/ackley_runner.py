#!/bin/env python
import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.synthetic.funcs.ackley import Ackley6DParametersSpace, AkcleyCfg
from autorocks.envs.synthetic.synth_objective_dao import TargetFuncObjective
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from scripts.exp_runners.experiments_runner import ExperimentsRunner, ExperimentsSkip


def experiment_config() -> ExperimentsRunner:
    # hardcoded parameters for easier re-generating of results
    exp_to_skip = ExperimentsSkip(
        skip_exp=set(), skip_optimizer=set(), skip_bench=set()
    )
    param_schema = Ackley6DParametersSpace()
    env = AkcleyCfg

    iterations = 100
    __OBJECTIVE = [TargetFuncObjective()]

    optimizers_to_evaluate = {
        # botorch_model.DeepGPModel.name": lambda: baseline_optimizers_cfg.botorch(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     surrogate_model=botorch_model.DeepGPModel(),
        # ),
        botorch_model.AdditiveModel.name: lambda: baseline_opt_cfg.botorch(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            surrogate_model=botorch_model.AdditiveModel(),
        ),
        botorch_model.SingleTaskModel.name: lambda: baseline_opt_cfg.botorch(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            surrogate_model=botorch_model.SingleTaskModel(),
        ),
        str(NNITuner.TPE): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.TPE,
        ),
        str(NNITuner.RANDOM): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.RANDOM,
        ),
        str(NNITuner.SMAC): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.SMAC,
        ),
        str(NNITuner.PBT): lambda: baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.PBT,
        ),
    }
    benchmarks_to_evaluate = {"": lambda x: x}

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
