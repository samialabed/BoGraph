#!/usr/bin/env python

import dataclasses
import gc
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from sysgym.params import ParamsSpace

import autorocks.data.loader.filenames_const as fn
from autorocks.envs.env_factory import env_from_cfg
from autorocks.envs.env_state import EnvState
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.optimizer.optimizer_factory import optimizer_from_cfg
from autorocks.project import ExperimentManager
from sysgym import EnvParamsDict


def _clear_memory():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if "cuda" in str(obj.device):
                obj.detach().to("cpu")
                del obj
    torch.cuda.empty_cache()
    gc.collect()


def runner_loop(cfg: ExperimentConfigs):
    # TODO(distributed) rename to manager
    # TODO refactor the way we store results, we should move to a database probably
    env_cls = env_from_cfg(env_cfg=cfg.env_cfg)
    with ExperimentManager(cfg) as ctx:
        optimizer = optimizer_from_cfg(opt_cfg=cfg.opt_cfg)
        env_params_holder = EnvParamsDict(param_space=cfg.opt_cfg.param_space)
        for exp_num in ctx:
            if torch.cuda.is_available():
                _clear_memory()
            env = env_cls(env_cfg=cfg.env_cfg, artifacts_output_dir=ctx.env_output_dir)
            optimizer_query_start_time = time.monotonic()
            params = optimizer.optimize_space()

            optimizer_query_end_time = time.monotonic()
            optimizer_query_duration = (
                optimizer_query_end_time - optimizer_query_start_time
            )
            ctx.logger.info(
                "Took %s seconds to query %s",
                optimizer_query_duration,
                cfg.opt_cfg.name,
            )

            env_params_holder.update(params)

            env_execution_stime = time.monotonic()

            measurements = env.run(env_params_holder)
            env_state = EnvState(params=env_params_holder, measurements=measurements)

            env_execution_duration = time.monotonic() - env_execution_stime

            with open(ctx.results_dir / fn.ITERATION_ENV_EXECUTION_TIME, "w") as f:
                f.write(str(env_execution_duration))

            optimizer_training_start_time = time.monotonic()
            optimizer.observe_state(state=env_state)
            optimizer_training_end_time = time.monotonic()

            optimizer_training_duration = (
                optimizer_training_end_time - optimizer_training_start_time
            )

            total_optimizer_time = (
                optimizer_training_duration + optimizer_query_duration
            )

            # TODO: move the timing of the training to the optimizer itself
            with open(ctx.results_dir / fn.ITERATION_MODEL_PERFS, "w") as f:
                f.write(str(total_optimizer_time))

            ctx.logger.info(f"{cfg.opt_cfg.name}: Took {total_optimizer_time} seconds.")

            env_params_holder.reset()
            if cfg.run_once:
                ctx.logger.info(
                    "Set to run only once. Iteration: %s/%s",
                    exp_num,
                    cfg.iterations,
                )
                break


if __name__ == "__main__":
    # TODO: exist as a hack around NNI trials.
    #  Once we refactor for distributed system it will not work.
    parser = ArgumentParser(
        description="Run BO loop manually given a serialized config file."
    )

    parser.add_argument(
        "--config",
        help="Path to configuration override, pickled, mostly used as hack around NNI.",
        type=Path,
        required=False,
    )
    args = parser.parse_args()

    override_config = args.config
    with open(override_config, "rb") as cfg_path:
        config: ExperimentConfigs = pickle.load(cfg_path)
        if isinstance(config.opt_cfg.param_space, dict):
            # Deserialize the serialization of nni experiment for dynamically created
            # spaces.
            fields = []
            picked_fields = config.opt_cfg.param_space["fields"]
            print(picked_fields)
            for field in picked_fields:
                fields.append(
                    (
                        field["name"],
                        field["type"],
                        field["type"](
                            field["lower_bound"],
                            field["upper_bound"],
                            default=field["default"],
                        ),
                    ),
                )

            cls_name = config.opt_cfg.param_space["cls_name"]

            reconstructed_space = dataclasses.make_dataclass(
                cls_name, fields=fields, bases=(ParamsSpace,), init=False, frozen=True
            )()

            config = dataclasses.replace(
                config,
                opt_cfg=dataclasses.replace(
                    config.opt_cfg, param_space=reconstructed_space
                ),
            )

    runner_loop(cfg=config)
