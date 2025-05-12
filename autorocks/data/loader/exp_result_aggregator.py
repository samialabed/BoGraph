import glob
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas import json_normalize

from autorocks.data.loader import filenames_const as fn
from autorocks.data.loader.exp_dao import (
    ExperimentData,
    ExperimentDataIncompleteException,
)

ITERATION_NUM_REGEX = re.compile(r"(\d*)/results")

LOG = logging.getLogger()


def get_all_iterations_as_one_exp(
    path_to_experiment: Path,
    out_path: Path = None,
    force_recompute: bool = False,
    save_results: bool = True,
) -> ExperimentData:
    """
    Combine all observed data into one experiment csv file.

    force: force to recreate the csv file even if it exist

    """
    try:
        if out_path is None:
            out_path = path_to_experiment
        # check if file exist before attempting to recreate it
        if force_recompute:
            LOG.info("Force recreate all experiments results.")

        # TODO: simplify using a for-loop
        # TODO: that is dependent on moving towards a model_perfs dataclass and unify it

        if force_recompute or not (out_path / fn.EXPERIMENT_MODEL_PERFS).is_file():
            exp_model_perfs_df = _combine_iterations_training_time(
                in_path=path_to_experiment,
                out_path=out_path,
                force_recompute=force_recompute,
            )
        else:
            LOG.debug(
                "Reusing parsed results in %s",
                out_path / fn.EXPERIMENT_MODEL_PERFS,
            )
            exp_model_perfs_df = pd.read_csv(out_path / fn.EXPERIMENT_MODEL_PERFS)

        if force_recompute or not (out_path / fn.EXPERIMENT_SYS_PARAMS).is_file():
            exp_sys_params_df = _combine_iterations_params(
                in_path=path_to_experiment, out_path=out_path
            )
        else:
            LOG.debug(
                "Reusing parsed results in %s",
                out_path / fn.EXPERIMENT_SYS_PARAMS,
            )
            exp_sys_params_df = pd.read_csv(out_path / fn.EXPERIMENT_SYS_PARAMS)

        if force_recompute or not (out_path / fn.EXPERIMENT_SYS_OBSERVATIONS).is_file():
            exp_sys_observations_df = _combine_iterations_sys_measurements(
                in_path=path_to_experiment, out_path=out_path
            )
        else:
            LOG.debug(
                "Reusing parsed results in %s",
                out_path / fn.EXPERIMENT_SYS_OBSERVATIONS,
            )
            exp_sys_observations_df = pd.read_csv(
                out_path / fn.EXPERIMENT_SYS_OBSERVATIONS
            )

        if save_results:
            exp_model_perfs_df.to_csv(out_path / fn.EXPERIMENT_MODEL_PERFS, index=False)
            LOG.info("Created %s", fn.EXPERIMENT_MODEL_PERFS)

            exp_sys_params_df.to_csv(out_path / fn.EXPERIMENT_SYS_PARAMS, index=False)
            LOG.info("Created %s", fn.EXPERIMENT_SYS_PARAMS)

            exp_sys_observations_df.to_csv(
                out_path / fn.EXPERIMENT_SYS_OBSERVATIONS, index=False
            )
            LOG.info("Created %s", fn.EXPERIMENT_SYS_OBSERVATIONS)
        return ExperimentData(
            model_performance=exp_model_perfs_df,
            sys_params=exp_sys_params_df,
            sys_observations=exp_sys_observations_df,
        )
    except Exception as e:
        LOG.error("Failed to process files in %s", path_to_experiment)
        raise e


def _combine_iterations_training_time(
    in_path: Path, out_path: Path, force_recompute: bool = False
) -> pd.DataFrame:
    LOG.info("Checking %s", in_path / fn.EXPERIMENT_MODEL_PERFS)
    if not force_recompute and (in_path / fn.EXPERIMENT_MODEL_PERFS).is_file():
        return pd.read_csv(in_path / fn.EXPERIMENT_MODEL_PERFS)
    LOG.info("Creating %s", out_path / fn.EXPERIMENT_MODEL_PERFS)
    exp_model_perfs = parse_model_perf(in_path)
    if not exp_model_perfs:
        LOG.error(
            "Skip parsing %s, not found in %s.",
            fn.EXPERIMENT_MODEL_PERFS,
            out_path,
        )
        raise ExperimentDataIncompleteException(
            f"{in_path / fn.EXPERIMENT_MODEL_PERFS} Not found"
        )

    exp_model_perfs_df = pd.DataFrame(
        exp_model_perfs, columns=["step", "inference_time"]
    )
    return exp_model_perfs_df


def _combine_iterations_params(in_path: Path, out_path: Path):
    if (in_path / fn.EXPERIMENT_SYS_PARAMS).is_file():
        return pd.read_csv(in_path / fn.EXPERIMENT_SYS_PARAMS)
    LOG.info("Creating %s", out_path / fn.EXPERIMENT_SYS_PARAMS)
    all_params = parse_params(in_path)
    if not all_params:
        LOG.error(
            "Skip parsing %s, not found in %s.",
            fn.EXPERIMENT_SYS_PARAMS,
            out_path,
        )
        raise ExperimentDataIncompleteException(
            f"{in_path / fn.EXPERIMENT_SYS_PARAMS} Not found"
        )

    all_params_df = pd.DataFrame(all_params)
    all_params_df = all_params_df.sort_values(by="step", ascending=True)
    return all_params_df


def _combine_iterations_sys_measurements(in_path: Path, out_path: Path):
    if (in_path / fn.EXPERIMENT_SYS_OBSERVATIONS).is_file():
        return pd.read_csv(in_path / fn.EXPERIMENT_SYS_OBSERVATIONS)
    # EXPERIMENT_SYS_MEASUREMENTS doesn't exist, create it
    LOG.info("Creating %s", out_path / fn.EXPERIMENT_SYS_OBSERVATIONS)
    exp_measurement_dict = parse_measurement(in_path)
    if not exp_measurement_dict:
        LOG.error(
            "Skip parsing %s, not found in %s.",
            fn.EXPERIMENT_SYS_OBSERVATIONS,
            out_path,
        )
        raise ExperimentDataIncompleteException(
            f"{in_path / fn.EXPERIMENT_SYS_OBSERVATIONS} Not found"
        )

    exp_measurement_df = json_normalize(exp_measurement_dict)
    exp_measurement_df = exp_measurement_df.sort_values(by="step", ascending=True)
    return exp_measurement_df


def parse_model_perf(in_path: Path) -> List[Dict]:
    """Parse all training times and return a single array containing them all.
    Should pass the path_to_model_exp to the experiment directory."""
    all_inferences: List[Dict] = []
    inference_files_path = glob.glob(
        str(in_path / f"*/results/{fn.ITERATION_MODEL_PERFS}")
    )
    for file_path in inference_files_path:
        with open(file_path, "r") as fp:
            inference_time = fp.readline()
            step_num = int(ITERATION_NUM_REGEX.findall(file_path)[0])
            all_inferences.append(
                {"step": step_num, "inference_time": float(inference_time)}
            )
    return all_inferences


def parse_params(in_path: Path) -> List[Dict]:
    """Parse all params and return a single array containing them all.
    Should pass the path to the experiment directory."""
    all_params: List[Dict] = []
    params_files_path = glob.glob(str(in_path / f"*/results/{fn.ITERATION_SYS_PARAMS}"))
    for file_path in params_files_path:
        with open(file_path, "r") as fp:
            param_file = json.load(fp)
            step_num = int(ITERATION_NUM_REGEX.findall(file_path)[0])
            param_file["step"] = step_num
            all_params.append(param_file)
    return all_params


def parse_measurement(in_path: Path) -> List[Dict]:
    """Parse all measurements  and return a single array containing them all.
    Should pass the path to the experiment directory."""
    # Step number is extracted from file path
    all_measurement: List[Dict] = []
    measurement_files_path = glob.glob(
        str(in_path / f"*/results/{fn.ITERATION_SYS_OBSERVATIONS}")
    )
    for file_path in measurement_files_path:
        with open(file_path, "r") as fp:
            measurement_file = json.load(fp)
            step_num = int(ITERATION_NUM_REGEX.findall(file_path)[0])
            measurement_file["step"] = step_num
            all_measurement.append(measurement_file)
    return all_measurement
