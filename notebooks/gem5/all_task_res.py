# Viz cell

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import autorocks.viz.viz as viz
from autorocks.data.loader.all_models_result_aggregator import (
    create_all_models_comparison_dataset,
)
from autorocks.dir_struct import RootDir
from autorocks.envs.gem5.benchmarks.benchmark_tasks import MachSuiteTask

# output_location = "/Users/salabed/workspace/latex_writings/papers/mlsys21_autobo/figs"
output_location = "/home/salabed/workspace/latex/papers/eurosys22_workshop/figs"
output_format = "svg"  # pdf
if output_format == "svg":
    output_location = f"{output_location}/svg"
save_res = True

color_palette = {
    "BoGraph": "#016450",
    "BoTorch": "#02818a",
    "DeepGP": "#3690c0",
    "Default": "#f6eff7",
    "PBTTuner": "#a6bddb",
    "Random": "#67a9cf",
}

exp_names = [
    MachSuiteTask.AES,
    MachSuiteTask.FFT_TRANSPOSE,
    MachSuiteTask.STENCIL_3D,
    MachSuiteTask.GEMMA_NCUBED,
    MachSuiteTask.STENCIL_2D,
    MachSuiteTask.FFT_STRIDED,
    MachSuiteTask.SPMV_CRS,
    MachSuiteTask.SPMV_ELLPACK,
    MachSuiteTask.MD_KNN,
]
for exp in exp_names:
    exp_name = str(exp)
    print(f"Generating figure for {exp_name}")
    exp_dir = RootDir.parent / f"local_execution/gem5osdi/{exp_name}/20_params/100_iter"
    model_comparison_data = create_all_models_comparison_dataset(exp_dir)
    # HACK until we fix logging

    model_comparison_data_c = viz.unify_model_name(model_comparison_data)
    performance = model_comparison_data_c.sys_observations
    latency = 1 / performance["detailed_stats.system.sim_ticks"]
    pdp = performance["bench_stats.avg_power"] * latency
    edp = np.log(latency * pdp.values)
    performance["bench_stats.edp"] = edp

    fig = viz.model_perf_plot(
        model_perf_df=model_comparison_data_c.model_performance,
        model_palette_map=color_palette,
        comparison_col="inference_time",
    )
    if save_res:
        print(f"saving: {exp_name}_exetime.{output_format}")
        fig.savefig(
            f"{output_location}/{exp_name}_exetime.{output_format}",
            bbox_inches="tight",
            format=f"{output_format}",
            dpi=600,
        )
    fig = viz.perf_boxplot(
        perf_df=performance,
        optimization_type=viz.OptimizationType.MINIMIZE,
        ylabel="log(EDP(x))",
        comparison_col="bench_stats.edp",
        model_palette_map=color_palette,
        # horizontal_line='Default'
    )
    if save_res:
        print(f"saving: {exp_name}_reduced_epd_perf.{output_format}")

        fig.savefig(
            f"{output_location}/{exp_name}_reduced_epd_perf.{output_format}",
            bbox_inches="tight",
            format=f"{output_format}",
            dpi=600,
        )

    convergence_df = performance.copy()

    convergence_df["rolling"] = convergence_df.groupby(["model", "iteration"]).agg(
        f"cum{str(viz.OptimizationType.MINIMIZE)}"
    )["bench_stats.edp"]

    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(convergence_df["rolling"].values.reshape(-1, 1))
    convergence_df["scaled"] = pd.DataFrame(
        arr_scaled, columns=["rolling_scaled"], index=convergence_df["rolling"].index
    )

    fig = viz.convergence_lintplot_roi(
        df=convergence_df,  # model_comparison_data_c.system_performance,
        optimization_type=viz.OptimizationType.MINIMIZE,
        # ylabel="EDP in LogScale",
        model_baseline="Default",
        # column_name='bench_stats.edp',
        column_name="rolling",
        model_palette_map=color_palette,
    )
    if save_res:
        print(f"saving: {exp_name}_convergence.{output_format}")
        fig.savefig(
            f"{output_location}/{exp_name}_convergence.{output_format}",
            bbox_inches="tight",
            format=f"{output_format}",
            dpi=600,
        )
