#!/bin/env python
from argparse import ArgumentParser

import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.synthetic.funcs import levy
from autorocks.envs.synthetic.synth_objective_dao import TargetFuncObjective
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.experiments import baseline_opt_cfg
from autorocks.optimizer.bograph import dag_options
from scripts.exp_runners.experiments_runner import (
    ExperimentsRunner,
    ExperimentsSkip,
    ExpUID,
)

# _DIM_TO_ITER = {25: 100, 50: 125, 100: 200, 300: 500, 1000: 1000}
_DIM_TO_ITER = {6: 100, 25: 100, 50: 125, 100: 200, 300: 500}


def experiment_config(dim: int) -> ExperimentsRunner:
    # hardcoded parameters for easier re-generating of results
    exp_to_skip = ExperimentsSkip(
        skip_exp=set(),
        skip_optimizer=set(),
        skip_bench=set(),
    )
    param_schema = levy.make_levy_space(dim)
    env = lambda _: levy.LevyCfg(dim=dim, noise_std=0.01)

    iterations = _DIM_TO_ITER[dim]
    __OBJECTIVE = [TargetFuncObjective()]

    optimizers_to_evaluate = {
        # botorch_model.TurboModel.name: baseline_opt_cfg.turbo(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     surrogate_model=botorch_model.TurboModel(param_schema.dimensions),
        # ),
        # "bobn": dag_options.BoBnConfig(
        #     name="BoBn",
        #     param_space=param_schema,
        #     opt_objectives=__OBJECTIVE,
        #     random_iter=10,
        #     retry=3,
        #     dag=levy.make_struct(dim),
        # ),
        # botorch_model.AdditiveModel.name: baseline_opt_cfg.botorch(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     surrogate_model=botorch_model.AdditiveModel(),
        # ),
        # botorch_model.SingleTaskModel.name: baseline_opt_cfg.botorch(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     surrogate_model=botorch_model.SingleTaskModel(),
        # ),
        # str(NNITuner.TPE): baseline_opt_cfg.nni_opt(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     tuner_name=NNITuner.TPE,
        # ),
        str(NNITuner.RANDOM): baseline_opt_cfg.nni_opt(
            param_schema=param_schema,
            opt_obj=__OBJECTIVE,
            tuner_name=NNITuner.RANDOM,
        ),
        # str(NNITuner.SMAC): baseline_opt_cfg.nni_opt(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     tuner_name=NNITuner.SMAC,
        # ),
        # str(NNITuner.PBT): baseline_opt_cfg.nni_opt(
        #     param_schema=param_schema,
        #     opt_obj=__OBJECTIVE,
        #     tuner_name=NNITuner.PBT,
        # ),
    }
    benchmarks_to_evaluate = {f"Dim={dim}": lambda x: x}

    return ExperimentsRunner(
        iterations=iterations,
        optimizer_to_evaluate=optimizers_to_evaluate,
        benchmarks_to_evaluate=benchmarks_to_evaluate,
        env_callable=env,
        exp_skip=exp_to_skip,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run gem5 case-study")
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        help="Choose the backend manager of the optimizer learner",
        choices=list(_DIM_TO_ITER.keys()),
        default=25,
    )
    cli_options = parser.parse_args()

    exp_runner_cfg = experiment_config(cli_options.dim)
    exp_runner_cfg.execute()
