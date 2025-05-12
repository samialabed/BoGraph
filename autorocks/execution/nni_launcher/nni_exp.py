import dataclasses
import pickle
from pathlib import Path

from nni.experiment import Experiment

from autorocks.dir_struct import RootDir
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.execution.nni_launcher.nni_utils import conver_params_to_nni_search_space
from autorocks.exp_cfg import ExperimentConfigs


def nni_experiment(cfg: ExperimentConfigs):
    tuner_name = cfg.opt_cfg.tuner_name
    param_space = cfg.opt_cfg.param_space
    search_space = conver_params_to_nni_search_space(param_space)
    experiment = Experiment("local")

    if tuner_name == "BOHB":  # advisor
        # Always maximize, we handle minimizing by negating the value
        experiment.config.advisor.name = str(tuner_name)
        experiment.config.advisor.class_args["optimize_mode"] = "maximize"
    else:
        experiment.config.tuner.name = str(tuner_name)
        # Always maximize at NNI,
        # we handle minimizing of objective by negating the value at objective level
        if tuner_name != NNITuner.RANDOM:
            experiment.config.tuner.class_args["optimize_mode"] = "maximize"
        if tuner_name == NNITuner.PBT:
            experiment.config.tuner.class_args["population_size"] = cfg.iterations
    experiment.config.trial_concurrency = 1
    experiment.config.max_trial_number = cfg.iterations
    experiment.config.search_space = search_space

    # run each trial only once
    cfg.run_once = True

    config_tmp_dir = Path(f"/tmp/nni_exp/{cfg.env_cfg.name}/{cfg.opt_cfg.name}")
    config_tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_location = config_tmp_dir / "config.p"
    with open(cfg_location, "wb") as f:
        try:
            pickle.dump(cfg, f)
        except pickle.PicklingError:
            print("Can't pickle the object, attempting to serialize parameter space")

            serialized_space = []
            assert len(param_space.values()) == len(dataclasses.fields(param_space))
            for param, field_data in zip(
                param_space.values(), dataclasses.fields(param_space)
            ):
                serialized_space.append(
                    {
                        "name": field_data.name,
                        "default": param.default,
                        "lower_bound": param.lower_bound,
                        "upper_bound": param.upper_bound,
                        "type": field_data.type,
                    }
                )
            pickle.dump(
                # workaround to get around
                dataclasses.replace(
                    cfg,
                    opt_cfg=dataclasses.replace(
                        cfg.opt_cfg,
                        param_space={
                            "cls_name": type(param_space).__name__,
                            "fields": serialized_space,
                        },
                    ),
                ),
                f,
            )
    # TODO: this won't work in distributed setting
    trial_cmd = (
        f"python {RootDir / 'execution' / 'local_loop.py'} --config={cfg_location}"
    )
    experiment.config.trial_command = trial_cmd
    experiment.config.trial_code_directory = RootDir
    experiment.run(17735)
    # Uncomment for pausing the experiment before quitting and allow debug
    # input("Press enter to quit")
    experiment.stop()
