import glob
import logging
import os
from pathlib import Path
from typing import List

from pandas import DataFrame

from autorocks.data.loader import filenames_const as fn

LOG = logging.getLogger()


def ls_subdir(path: Path) -> List[Path]:
    """Helper function to list only subdirectories. Returns full path"""
    return list(filter(lambda x: os.path.isdir(x), path.iterdir()))


def summary_perf(df: DataFrame) -> DataFrame:
    """Calculate the overall best min, max, median across all iterations"""
    summary_df = DataFrame()
    summary_df["min_perf"] = df.min(axis=0)
    summary_df["max_perf"] = df.max(axis=0)

    summary_df = summary_df.reset_index().rename(columns={"index": "iteration"})
    return summary_df


def convergence_summary(df: DataFrame) -> DataFrame:
    convergence_df = DataFrame()

    rolling_min = df.cummin(0)
    convergence_df["rolling_min_min"] = rolling_min.min(1)
    convergence_df["rolling_min_median"] = rolling_min.median(1)
    convergence_df["rolling_min_max"] = rolling_min.max(1)

    rolling_max = df.cummax(0)
    convergence_df["rolling_max_min"] = rolling_max.min(1)
    convergence_df["rolling_max_median"] = rolling_max.median(1)
    convergence_df["rolling_max_max"] = rolling_max.max(1)

    convergence_df = convergence_df.reset_index().rename(columns={"index": "step"})
    return convergence_df


def clean_cached_files(exp_dir: Path, dry_run: bool = False):
    for file in [
        fn.EXPERIMENT_MODEL_PERFS,
        fn.EXPERIMENT_SYS_PARAMS,
        fn.EXPERIMENT_SYS_OBSERVATIONS,
        fn.MODEL_ALL_EXP_MODEL_PERFS,
        fn.MODEL_ALL_EXP_SYS_OBSERVATIONS,
        fn.MODEL_ALL_EXP_SYS_PARAMS,
        fn.ALL_MODELS_SYS_OBSERVATIONS,
        fn.ALL_MODELS_MODEL_PERFS,
        fn.ALL_MODELS_SYS_PARAMS,
        fn.PARSED_EXP,
    ]:
        matching_files = glob.glob(str(exp_dir / "**" / file), recursive=True)
        for file_path in matching_files:
            LOG.info("Removing: %s", file_path)
            if not dry_run:
                os.remove(file_path)
