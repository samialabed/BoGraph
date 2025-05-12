from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from autorocks.data.loader.all_models_result_aggregator import (
    create_all_models_comparison_dataset,
)
from autorocks.dir_struct import RootDir
from autorocks.envs.gem5.benchmarks.benchmark_tasks import MachSuiteTask
from autorocks.viz import viz


def create_compare_all() -> pd.DataFrame:
    task_name_to_res = []

    for task in [
        MachSuiteTask.AES,
        MachSuiteTask.FFT_TRANSPOSE,
        MachSuiteTask.STENCIL_3D,
        MachSuiteTask.GEMMA_NCUBED,
        MachSuiteTask.STENCIL_2D,
        MachSuiteTask.FFT_STRIDED,
        MachSuiteTask.SPMV_CRS,
        MachSuiteTask.SPMV_ELLPACK,
        MachSuiteTask.MD_KNN,
    ]:
        exp_dir = RootDir.parent / f"local_execution/gem5osdi/{task}/20_params/100_iter"
        model_comparison_data = create_all_models_comparison_dataset(exp_dir)
        model_comparison_data_c = viz.unify_model_name(model_comparison_data)
        color_palette = viz.create_color_palette(
            model_comparison_data_c,
            ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"],
        )

        sys_perf = model_comparison_data_c.sys_observations
        latency = 1 / sys_perf["detailed_stats.system.sim_ticks"]

        pdp = sys_perf["bench_stats.avg_power"] * latency
        # pdp = self.avg_power * self.cycle

        edp = np.log(latency * pdp.values)
        # pdp
        sys_perf["bench_stats.edp"] = edp

        perf_df = sys_perf
        perf_df = (
            perf_df[["bench_stats.edp", "iteration", "model"]]
            .groupby(["model", "iteration"])
            .agg(str(viz.OptimizationType.MINIMIZE))
            .reset_index()
        )
        perf_df["task"] = str(task)
        perf_df.to_dict()
        task_name_to_res.append(perf_df)
    res = pd.concat([pd.DataFrame(x) for x in task_name_to_res])

    res["task"] = res["task"].apply(
        lambda x: x.replace("gemm_", "")
        .replace("stencil_", "")
        .replace("aes_", "")
        .replace("fft_", "")
        .lower()
    )
    return res


def highest_correlation(
    df: DataFrame, target_var: str, correlation_cutoff=0.5
) -> Dict[str, int]:
    # Correlation between target features
    feature_corr = df.corr()
    cor_target = abs(feature_corr[target_var])  # Selecting highly correlated features
    relevant_features = cor_target[cor_target > correlation_cutoff]
    return relevant_features


def parameters_corr_to_stats(
    df: DataFrame, parameters_df: DataFrame, correlation_cutoff=0.5
) -> Dict[str, List[str]]:
    param_statistics = parameters_df.merge(df, on=["model", "step", "iteration"])

    parameters = parameters_df.columns.to_list()[:-3]
    corr_filter = abs(param_statistics.corr()) > correlation_cutoff

    param_to_stat = {}
    for param in parameters:
        param_to_stat[param] = (
            corr_filter[param].loc[corr_filter[param]].index.to_list()
        )

    return param_to_stat
