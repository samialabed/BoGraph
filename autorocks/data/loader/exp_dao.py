import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

import pandas as pd
from pandas import DataFrame

from autorocks.data.loader import filenames_const as fn

LOG = logging.getLogger()


class ExperimentDataIncompleteException(BaseException):
    pass


class DataframeNameOnDisk(NamedTuple):
    model_performance: str
    sys_observations: str
    sys_params: str


@dataclass
class BaseDataDAO(ABC):
    # TODO: make this a dataclass with various stats collected about the model
    model_performance: pd.DataFrame
    sys_observations: pd.DataFrame
    # TODO: make this a dataclass of the params
    sys_params: pd.DataFrame

    def __post_init__(self):
        rows_model_perfs = self.model_performance.shape[0]
        rows_sys_observations = self.sys_observations.shape[0]
        rows_sys_params = self.sys_params.shape[0]
        if not (rows_model_perfs == rows_sys_observations == rows_sys_params):
            logging.error(
                "Missing %s experiment.",
                set(self.model_performance["step"].values).difference(
                    set(self.sys_observations["step"].values),
                    set(self.sys_params["step"].values),
                ),
            )
        assert rows_model_perfs == rows_sys_observations == rows_sys_params, (
            f"Expected rows of: model_performance [{rows_model_perfs}]"
            f" == system_observations [{rows_sys_observations}] "
            f"== system_parameters [{rows_sys_params}] "
        )

    @staticmethod
    @abstractmethod
    def param_to_name_on_disk() -> DataframeNameOnDisk:
        """Dictionary that maps from the name of the parameter to its name on disk"""

    def copy(self, shallow=False):
        if shallow:
            return copy.copy(self)
        else:
            return copy.deepcopy(self)

    def set_col(self, col_name: str, col_val: any):
        self.model_performance[col_name] = col_val
        self.sys_observations[col_name] = col_val
        self.sys_params[col_name] = col_val

    def save(self, output_path: Path) -> None:
        """Save the experiment data on disk"""
        LOG.info("Saving ModelExperimentsData in %s", str(output_path))
        names_on_disk = self.param_to_name_on_disk()
        self.sys_observations.to_csv(output_path / names_on_disk.sys_observations)
        self.sys_params.to_csv(output_path / names_on_disk.sys_params)
        self.model_performance.to_csv(output_path / names_on_disk.model_performance)

    @classmethod
    def load(cls, cached_summary_files_path: Path):
        """Load preprocessed data from disk into memory."""
        LOG.info(
            "Reading cached summary results stored in %s",
            str(cached_summary_files_path),
        )
        names_on_disk = cls.param_to_name_on_disk()

        all_exp_sys_observations = pd.read_csv(
            cached_summary_files_path / names_on_disk.sys_observations, index_col=0
        )
        all_exp_sys_params = pd.read_csv(
            cached_summary_files_path / names_on_disk.sys_params, index_col=0
        )
        all_exp_model_perfs = pd.read_csv(
            cached_summary_files_path / names_on_disk.model_performance, index_col=0
        )

        return cls(
            model_performance=all_exp_model_perfs,
            sys_observations=all_exp_sys_observations,
            sys_params=all_exp_sys_params,
        )

    def combine_sys_params_metric(self) -> pd.DataFrame:
        """Combine all dataframes into one"""
        return pd.merge(
            left=self.sys_params,
            right=self.sys_observations,
            on=["model", "step", "iteration"],
        )


@dataclass
class ExperimentData(BaseDataDAO):
    """DAO holding the data for a single experiment."""

    @staticmethod
    def param_to_name_on_disk() -> DataframeNameOnDisk:
        return DataframeNameOnDisk(
            model_performance=fn.EXPERIMENT_MODEL_PERFS,
            sys_observations=fn.EXPERIMENT_SYS_OBSERVATIONS,
            sys_params=fn.EXPERIMENT_SYS_PARAMS,
        )


@dataclass
class ModelExperimentsData(BaseDataDAO):
    @classmethod
    def from_experiments_dataset(
        cls, experiments_dataset: List[ExperimentData]
    ) -> "ModelExperimentsData":
        model_perfs_list = []
        sys_params_list = []
        sys_obs_list = []

        for exp_res in experiments_dataset:
            model_perfs_list.append(exp_res.model_performance)
            sys_params_list.append(exp_res.sys_params)
            sys_obs_list.append(exp_res.sys_observations)

        model_perfs_df = pd.concat(model_perfs_list, ignore_index=True, axis=0)
        sys_params_df = pd.concat(sys_params_list, ignore_index=True, axis=0)
        sys_obs_df = pd.concat(sys_obs_list, ignore_index=True, axis=0)
        return cls(
            model_performance=model_perfs_df,
            sys_observations=sys_obs_df,
            sys_params=sys_params_df,
        )

    @staticmethod
    def param_to_name_on_disk() -> DataframeNameOnDisk:
        return DataframeNameOnDisk(
            model_performance=fn.MODEL_ALL_EXP_MODEL_PERFS,
            sys_observations=fn.MODEL_ALL_EXP_SYS_OBSERVATIONS,
            sys_params=fn.MODEL_ALL_EXP_SYS_PARAMS,
        )


@dataclass
class ModelsComparisonData(BaseDataDAO):
    """
    Attributes:
        model_performance: Captures the execution time of the optimizer.
        sys_observations: The observed measurements from the system.
        sys_params: The configurations proposed by the auto-tuner.
    """

    model_performance: DataFrame
    sys_observations: DataFrame
    sys_params: DataFrame

    @staticmethod
    def param_to_name_on_disk() -> DataframeNameOnDisk:
        return DataframeNameOnDisk(
            model_performance=fn.ALL_MODELS_MODEL_PERFS,
            sys_observations=fn.ALL_MODELS_SYS_OBSERVATIONS,
            sys_params=fn.ALL_MODELS_SYS_PARAMS,
        )

    def filter_for_specific_models(self, models: Set[str]) -> "ModelsComparisonData":
        """Return a copy of the dataset subset on a specific set of models."""
        return self.filter_col_for_values(col_name="model", values=models)

    def filter_col_for_values(
        self, col_name: str, values: Set[any]
    ) -> "ModelsComparisonData":
        """Return a copy of the dataset subset on a specific set of models."""
        return ModelsComparisonData(
            model_performance=self.model_performance[
                self.model_performance[col_name].isin(values)
            ].copy(),
            sys_observations=self.sys_observations[
                self.sys_observations[col_name].isin(values)
            ].copy(),
            sys_params=self.sys_params[self.sys_params[col_name].isin(values)].copy(),
        )

    def available_models_and_iterations(self) -> Dict[str, Set[int]]:
        models_to_iters = {}
        all_models = self.model_performance["model"].unique()
        for model in all_models:
            all_iterations = self.model_performance[
                self.model_performance["model"] == model
            ]["iteration"].unique()

            models_to_iters[model] = {iteration for iteration in all_iterations}

        return models_to_iters
