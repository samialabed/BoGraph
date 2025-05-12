import logging
import logging.config as log_cfg
import os
from pathlib import Path

import autorocks.data.loader.filenames_const as fn
from autorocks.dir_struct import LocalResultDir
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.logging_util import BOGRAPH_LOGGER, log_config_dict
from autorocks.utils.singleton import SingleInstanceMetaClass


class ExperimentManager(metaclass=SingleInstanceMetaClass):
    """
    This class represents our project. It stores useful information about the structure.

    TODO: I should remove this singelton
    TODO: replace it with function calling to get the context if it exist
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #  Not thread safe
        # TODO: HACK that clears the singleton after the context is done.
        SingleInstanceMetaClass.clear(self.__class__)

    def __init__(self, cfg: ExperimentConfigs = None):
        assert (
            cfg
        ), "Config needs to be provided at the beginning of the experiment lifespan."
        self.max_iters = cfg.iterations

        # If experiment destination is defined,
        # otherwise save locally in experiments directory
        if cfg.exp_dir:
            self.exp_root_dir = Path(cfg.exp_dir)
        else:
            self.exp_root_dir = LocalResultDir
        self.exp_root_dir.mkdir(parents=True, exist_ok=True)

        # Experiment directory structure:
        #   /env_name/benchmark_name/num_params/exp_name(optimizer)/exp_time/_iterations
        obj_names = "_".join([n.name for n in cfg.opt_cfg.opt_objectives])
        self.experiment_relative_path = (
            Path(str(cfg.env_cfg.name)) / obj_names  # env_name  # name of objectives
        )
        if hasattr(cfg.env_cfg, "bench_cfg"):
            self.experiment_relative_path = (
                self.experiment_relative_path / cfg.env_cfg.bench_cfg.name
            )  # benchmark_container name
        self.experiment_relative_path = (
            self.experiment_relative_path
            # TODO: Maybe use name of parameter space rather than number
            / f"{len(cfg.opt_cfg.param_space)}_params"  # num params
            / f"{cfg.iterations}_iter"  # num iterations
            / str(cfg.opt_cfg.name)  # optimizer name
            / cfg.experiment_time  # exp time, TODO: do we want to use UUID?
        )
        self.experiment_dir = self.exp_root_dir / self.experiment_relative_path
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        self.debug = cfg.debug
        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        log_cfg.dictConfig(log_config_dict(str(self.logs_dir), self.debug))

        # TODO choose the logger based on the running environment
        self.logger = logging.getLogger(BOGRAPH_LOGGER)

        self.cur_iter = 0
        self._per_iteration_dir = self.experiment_dir

        # TODO parse configuration to see if seed is specified or not
        self.logger.info("Saving configuration to disk.")

        config_file_out = f"{self.experiment_dir}/{fn.CONFIG_FILE}"
        if not os.path.isfile(config_file_out):
            with open(config_file_out, "w", encoding="utf-8") as f:
                f.write(cfg.to_json(indent=2))

    @property
    def model_checkpoint_dir(self):
        """Per iteration checkpoint dir for optimizer."""
        checkpoint_dir = self._per_iteration_dir / "model_checkpoint"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        return checkpoint_dir

    @property
    def results_dir(self):
        """Per iteration results dir."""
        results_dir = self._per_iteration_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        return results_dir

    @property
    def env_output_dir(self):
        """Per iteration output dir for environment artifacts, parsers and benchmark."""
        out_dir = self._per_iteration_dir / "env_output"
        out_dir.mkdir(exist_ok=True, parents=True)
        return out_dir

    def __iter__(self):
        return self

    def __next__(self):
        """Use this to start new experiment, for now everything is single thread
        so it is safe to do this. In future we should move to a database.
        """
        self.cur_iter = sum(
            os.path.isdir(os.path.join(self.experiment_dir, i))
            for i in os.listdir(self.experiment_dir)
        )
        if self.cur_iter <= self.max_iters:
            self.logger.info("Executing iteration %s/%s", self.cur_iter, self.max_iters)
            self._per_iteration_dir = self.experiment_dir / str(self.cur_iter)
            self._per_iteration_dir.mkdir(exist_ok=True, parents=True)
            return self.cur_iter
        self.logger.info("Finished executing %s iterations.", self.max_iters)
        raise StopIteration
