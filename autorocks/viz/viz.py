""" Contains all visualization helper functions. """
import pathlib
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from botorch.utils.multi_objective import is_non_dominated
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from seaborn.palettes import _ColorPalette

from autorocks.data.loader.exp_dao import ModelsComparisonData
from autorocks.utils.enum import ExtendedEnum
from autorocks.viz.official_model_names import MODEL_NAME_MAPPING

_DEFAULT_OUTPUT_LOCATION = pathlib.Path("~/Workspace/latex_writings/thesis/phd_dissertation/Chapters/").expanduser()
_CHAPTERS = {
    "Background",
    "Appendix",
    "BoBn",
    "BoGraph",
    "BoGraphEval",
    "Conclusion",
    "Introduction",
    "RelatedWork"
}

NUM = Union[float, int]


class OptimizationType(ExtendedEnum):
    MINIMIZE = "min"
    MAXIMIZE = "max"


def scatter_plot_params(
    model_comparison_data: ModelsComparisonData,
    comparison_col: str,
    models_filter: Set[str] = set(),
    params_filter: Set[str] = set(),
) -> Figure:
    """Helps understanding the impact of a parameter against an objective."""
    fig, ax = plt.subplots(figsize=(4, 3))
    parameters_list = set(model_comparison_data.sys_params.columns)
    parameters_list.remove("step")
    parameters_list.remove("iteration")
    parameters_list.remove("model")

    params_and_target = model_comparison_data.sys_observations.merge(
        model_comparison_data.sys_params, on=["step", "iteration", "model"]
    )

    if models_filter:
        params_and_target = params_and_target[
            params_and_target["model"].isin(models_filter)
        ]

    color_palette = sns.color_palette("colorblind", len(parameters_list))

    for i, p in enumerate(parameters_list):
        if params_filter and p not in params_filter:
            continue

        sns.scatterplot(
            data=params_and_target, x=p, y=comparison_col, ax=ax, color=color_palette[i]
        )
    plt.close()
    return fig


def roi_boxplot(
    df: DataFrame,
    comparison_col: str,
    model_baseline: str,
    optimization_type: OptimizationType,
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    yscale: Optional[str] = None,
    fig_size: Tuple[int, int] = (4, 3),
) -> Figure:
    fig, ax = plt.subplots(figsize=fig_size)
    df = (
        df.groupby(["model", "iteration"])
        .max()
        .reset_index()[["model", comparison_col]]
    )
    baseline_df = df[df["model"] == model_baseline][comparison_col].to_numpy()
    non_baseline = df[df["model"] != model_baseline][comparison_col].to_numpy()
    if optimization_type == OptimizationType.MINIMIZE:
        roi = (baseline_df - non_baseline) / baseline_df * 100
    else:
        roi = (non_baseline - baseline_df) / baseline_df * 100
    # Order the plot
    df = df[df["model"] != model_baseline]
    df[comparison_col] = roi
    plot_order = cal_plot_order(
        df, comparison_col, ascending=optimization_type == OptimizationType.MINIMIZE
    )

    ax = sns.boxplot(
        data=df,
        x="model",
        y=comparison_col,
        order=plot_order,
        palette=model_palette_map,
        ax=ax,
        dodge=False,
    )

    ylabel = f"{ylabel} % improvement over {model_baseline}"
    _post_fig_processing(ax, comparison_col, title, ylabel, yscale)

    lines = ax.get_lines()
    for cat in ax.get_xticks():
        # every 4th line at the interval of 6 is median line 0 -> p25 1 -> p75 2 ->
        # lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4 + cat * 6].get_ydata()[0], 1)

        ax.text(
            cat,
            y,
            f"{y}%",
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="white",
            bbox=dict(facecolor="#445A64"),
        )

    return fig


def model_perf_plot(
    model_perf_df: DataFrame,
    comparison_col: str,
    yscale: Optional[str] = None,
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    ylabel: Optional[str] = None,
    title: str = "",
    horizontal_line: Optional[NUM] = None,
) -> Figure:
    """
    Produces a barplot comparing the comparison_col between various models for a given
    experiment.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    model_perf_df = model_perf_df[model_perf_df["model"] != "Default"]

    # Order the plot by model performance ascending
    model_order = (
        model_perf_df.groupby("model")[comparison_col]
        .mean()
        .sort_values(ascending=True)
    )

    plot_order = model_order.index
    sns.barplot(
        data=model_perf_df,
        x="model",
        y=comparison_col,
        order=plot_order,
        palette=model_palette_map,
        ax=ax,
        capsize=0.05,
    )
    # Human readable formatting
    # _add_num_to_plot(ax, fmt="{}s")
    for y, x in enumerate(model_order):
        plt.annotate(f"{x:.2f}s", xy=(10, y), va="center")

    _post_fig_processing(ax, comparison_col, title, ylabel, yscale, horizontal_line)
    return fig


def perf_boxplot(
    perf_df: DataFrame,
    optimization_type: OptimizationType,
    comparison_col: str,
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    ylabel: Optional[str] = None,
    yscale: Optional[str] = None,
    title: str = "",
    horizontal_line: Optional[Union[NUM, str]] = None,
    fig_size: Tuple[int, int] = (4, 3),
    add_roi: bool = False,
) -> Figure:
    """
    Display a boxplot showing the best found minimum or maximum performance target.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    perf_df = (
        perf_df[[comparison_col, "iteration", "model"]]
        .groupby(["model", "iteration"])
        .agg(str(optimization_type))
        .reset_index()
    )

    if isinstance(horizontal_line, str):
      horizontal_line_str = horizontal_line
      horizontal_line = float(perf_df[perf_df["model"] == horizontal_line_str][comparison_col].item())
      perf_df = perf_df[perf_df["model"] != horizontal_line_str]
      if model_palette_map:
        model_palette_map = model_palette_map.copy()
        model_palette_map.pop(horizontal_line_str)


    # Order the plot by what minimize/maximize objective time ascending
    plot_order = cal_plot_order(
        df=perf_df,
        comparison_col=comparison_col,
        ascending=optimization_type == optimization_type.MINIMIZE,
    )
    ax = sns.boxplot(
        data=perf_df,
        x="model",
        y=comparison_col,
        palette=model_palette_map,
        ax=ax,
        order=plot_order,
        saturation=3.95,
        linewidth=2.5,
        dodge=False,
    )

    _post_fig_processing(ax, comparison_col, title, ylabel, yscale, horizontal_line)
    ax.legend(loc="upper center")

    if add_roi:
        lines = ax.get_lines()
        for cat in ax.get_xticks():
            # every 4th line at the interval of 6 is median line 0 -> p25 1 -> p75 2 ->
            # lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
            y_pos = round(lines[4 + cat * 6].get_ydata()[0], 1)
            val = (y_pos - horizontal_line) / horizontal_line * 100
            val = round(val, 2)
            ax.text(
                cat,
                y_pos,
                f"{val}%",
                ha="center",
                va="center",
                fontweight="bold",
                size=10,
                color="white",
                bbox=dict(facecolor="#445A64"),
            )

    return fig


def _post_fig_processing(
    ax: Axes,
    comparison_col: Optional[str] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    yscale: Optional[str] = None,
    horizontal_line: Optional[NUM] = None,
):
    assert (
        comparison_col or ylabel
    ), "Please provide either a label or a comparison col to infer the label from."

    if not ylabel:
        assert comparison_col
        ylabel = comparison_col.replace("_", " ").capitalize()
    if yscale:
        ylabel = f"{ylabel} in {yscale} scale"
        ax.set(yscale=yscale)
    if title:
        ax.set_title(title)
    if horizontal_line:

        ax.axhline(horizontal_line, color="red", label="Default")

    ax.set(xlabel="Model", ylabel=ylabel)

    plt.close()


def convergence_lineplot(
    df: DataFrame,
    ylabel: str,
    column_name: str,
    convergence_plot: bool = True,
    optimization_type: Optional[OptimizationType] = None,
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    title: str = "",
    fig_size: Tuple[int, int] = (4, 3),
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    horizontal_line: Optional[str] = None,
) -> Figure:
    """
    Creates a figure that shows the median, min, and max of convergence
    towards the objective per step (where the error bars is per experiment rerun).
    """
    df = df.copy()

    horizontal_line_val = None
    if horizontal_line is not None:
        horizontal_line_val = (
            df[df["model"] == horizontal_line][column_name].mean().item()
        )
        df = df[df["model"] != horizontal_line]
        if model_palette_map and horizontal_line in model_palette_map:
            model_palette_map = model_palette_map.copy()
            model_palette_map.pop(horizontal_line)

    if convergence_plot:
        assert (
            optimization_type
        ), "Expect optimization_type to be supplied when plotting convergence plot"
        df["rolling"] = df.groupby(["model", "iteration"]).agg(
            f"cum{str(optimization_type)}"
        )[column_name]

        y = "rolling"
    else:
        y = column_name

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(
        data=df,
        x="step",
        hue="model",
        y=y,
        palette=model_palette_map,
        ax=ax,
        style="model",
        linestyle="",
        errorbar=("ci", 90),
        estimator=np.mean,
        markers=True,
        # dashes={
        #     "BoGraph": (),
        #     "Default": (5, 5),
        #     "DeepGP": (1, 1),
        #     "BoTorch": (1, 1),
        #     "Random": (1, 1),
        #     "SMAC": (1, 1),
        #     "Struct": (5, 5),
        # },
        # err_style="bars",
    )

    ax.set(xlabel="Optimization step", ylabel=ylabel, title=title)
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.seet(ylim=ylim)
    if horizontal_line_val is not None:
        ax.axhline(horizontal_line_val, color="red", label="Default")
    plt.legend(ncol=len(model_palette_map) // 2, loc="upper center", fontsize="small")
    plt.close()
    return fig


def imput_all_missing_vals(
    df: pd.DataFrame, max_steps: int, target: str
) -> pd.DataFrame:
    df = df[["model", "iteration", "step", target]]
    all_dfs = []
    all_models = df["model"].unique()
    for model in all_models:
        all_iterations = df[df["model"] == model]["iteration"].unique()
        for iteration in all_iterations:
            all_dfs.append(
                impute_missing_vals(
                    df=df,
                    max_steps=max_steps,
                    target=target,
                    model=model,
                    iteration=iteration,
                )
            )

    for imputated in all_dfs:
        df = df.append(imputated)
    return df.reset_index(drop=True).sort_values(by="step")


def impute_missing_vals(
    df: pd.DataFrame, max_steps: int, target: str, model: str, iteration: int
) -> pd.DataFrame:
    observed_steps = set(
        df[(df["model"] == model) & (df["iteration"] == iteration)]["step"].values
    )
    missing_steps = {i for i in range(1, max_steps + 1)} - observed_steps
    max_observed = df[(df["model"] == model) & (df["iteration"] == iteration)][
        target
    ].max()

    imputated_values_df = pd.DataFrame(
        [
            {
                "model": model,
                "iteration": iteration,
                "step": s,
                target: max_observed,
            }
            for s in missing_steps
        ]
    )
    # df.append(imputated_values_df).sort_values(by="step").reset_index(drop=True)
    return imputated_values_df


def convergence_lintplot_roi(
    df: DataFrame,
    # optimization_type: OptimizationType,
    model_baseline: str,
    optimization_type: OptimizationType,
    column_name: str = "rolling",
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    title: str = "",
) -> Figure:
    """
    ROI For convergence. Higher is always better
    """
    non_baseline_df = df[(df["step"] > 10) & (df["model"] != model_baseline)]
    non_baseline = non_baseline_df.groupby(["model", "step"])[column_name]
    non_baseline_median = non_baseline.median()
    non_baseline_max = non_baseline.max()
    non_baseline_mins = non_baseline.min()

    baseline = df[df["model"] == model_baseline][column_name]
    baseline_median = baseline.median()
    baseline_max = baseline.max()
    baseline_min = baseline.min()

    if optimization_type == OptimizationType.MINIMIZE:
        # roi_series_median = (
        #     (baseline_median - non_baseline_median) / baseline_median * 100
        # )
        # roi_series_mins = (baseline_min - non_baseline_mins) / baseline_min * 100
        # roi_series_maxes = (baseline_max - non_baseline_max) / baseline_max * 100
        roi_series_median = baseline_median - non_baseline_median
        roi_series_mins = baseline_min - non_baseline_mins
        roi_series_maxes = baseline_max - non_baseline_max
    else:
        roi_series_median = (
            (non_baseline_median - baseline_median) / baseline_median * 100
        )
        roi_series_mins = (non_baseline_mins - baseline_min) / baseline_min * 100
        roi_series_maxes = (non_baseline_max - baseline_max) / baseline_max * 100

    # df = roi_series_median.to_frame().reset_index()
    df = pd.DataFrame(
        pd.concat([roi_series_mins, roi_series_median, roi_series_maxes], axis=0)
    ).reset_index()
    df = df[df["model"] != model_baseline]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.lineplot(
        data=df,
        x="step",
        hue="model",
        y=column_name,
        palette=model_palette_map,
        ax=ax,
        style="model",
        # err_style="bars",
        dashes=True,
        markers=False
        # estimator=np.median,
    )

    ax.set(
        xlabel="Optimization step",
        ylabel=f"X-factor improvement over {model_baseline}",
        title=title,
    )
    plt.legend(ncol=len(model_palette_map) // 2, loc="lower center", fontsize="small")
    plt.close()

    return fig


def cal_plot_order(df: DataFrame, comparison_col: str, ascending: bool):
    return (
        df.groupby("model")
        .mean()
        .sort_values(by=comparison_col, ascending=ascending)
        .index.values
    )


def _add_num_to_plot(ax: Axes, fmt="{}"):
    for p in ax.patches:
        txt = fmt.format(str(p.get_height().round(1)))
        txt_x = p.get_x()
        txt_y = p.get_height()
        ax.text(
            txt_x,
            txt_y,
            txt,
        )


# Post processing human readable - extract to a file?
def create_color_palette(
    df: ModelsComparisonData, palette="Paired"
) -> Dict[str, _ColorPalette]:
    # palette = {deep, muted, pastel, dark, bright, colorblind}
    unique_models = df.sys_observations["model"].unique()
    palette = sns.color_palette(palette, len(unique_models), as_cmap=False)
    # Assign each model a unique color
    model_to_color_dict = dict(zip(unique_models, palette))
    return model_to_color_dict


def create_color_palette_df(
    df: pd.DataFrame, palette: List[str] = None
) -> Dict[str, _ColorPalette]:
    # palette = {deep, muted, pastel, dark, bright, colorblind}
    unique_models = df["model"].unique()
    palette = sns.color_palette(palette, len(unique_models), as_cmap=True)
    # Assign each model a unique color
    model_to_color_dict = dict(zip(unique_models, palette))
    return model_to_color_dict


def create_color_palette_for_list_df(
    unique_models: List[str], palette: List[str] = None
) -> Dict[str, _ColorPalette]:
    # palette = {deep, muted, pastel, dark, bright, colorblind}
    palette = sns.color_palette(palette, len(unique_models), as_cmap=True)
    # Assign each model a unique color
    model_to_color_dict = dict(zip(unique_models, palette))
    return model_to_color_dict


def unify_model_name(data: ModelsComparisonData) -> ModelsComparisonData:
    data = data.copy()
    data.model_performance = _unify_model_name_df(data.model_performance)
    data.sys_observations = _unify_model_name_df(data.sys_observations)
    return data


def _unify_model_name_df(df: DataFrame) -> DataFrame:
    df = df.copy()
    df["model"] = df["model"].map(MODEL_NAME_MAPPING).fillna(df["model"])
    return df


def convergence_dominated_points(
    df: DataFrame,
    optimization_type: OptimizationType,
    objectives: List[str],
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    title: str = "",
) -> Figure:
    sign = -1 if optimization_type == OptimizationType.MINIMIZE else 1
    y = torch.tensor(df[objectives].values, device="cpu") * sign
    pareto_frontier_all_models = is_non_dominated(y).cpu().numpy()
    df["is_non_dominated"] = False
    df.loc[pareto_frontier_all_models, "is_non_dominated"] = True
    df["count_of_dom"] = (
        df[["model", "iteration", "is_non_dominated"]]
        .groupby(["model", "iteration"])
        .cumsum()
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(
        data=df,
        x="step",
        y="count_of_dom",
        hue="model",
        ax=ax,
        palette=model_palette_map,
    )

    ax.set(xlabel="Step", ylabel="Count of pareto frontiers", title=title)
    plt.close()

    return fig


def pareto_frontier(
    df: DataFrame,
    optimization_type: OptimizationType,
    objectives: List[str],
    model_marker: Dict[str, str],
    model_palette_map: Optional[Dict[str, _ColorPalette]] = None,
    title: str = "",
) -> Figure:
    df = df.groupby(["step", "model"]).median().reset_index()

    model_observed_y = {}
    dominated_points = []
    sign = -1 if optimization_type == OptimizationType.MINIMIZE else 1
    for model in df.model.unique():
        model_res = df[df["model"] == model]
        model_obj = torch.tensor(model_res[objectives].values, device="cpu") * sign
        model_pareto_frontier = is_non_dominated(model_obj).numpy()
        dominated_points.append(model_res[np.invert(model_pareto_frontier)])
        model_observed_y[model] = model_pareto_frontier

    dominated_points_model_df = pd.concat(dominated_points)

    y = torch.tensor(df[objectives].values, device="cpu") * sign
    pareto_frontier_all_models = is_non_dominated(y).cpu().numpy()

    fig, ax = plt.subplots(figsize=(16, 9))

    # All points that are dominated
    ax = sns.scatterplot(
        data=dominated_points_model_df,
        x=objectives[0],
        y=objectives[1],
        palette=model_palette_map,
        hue="model",
        style="model",
        markers=model_marker,
        norm=matplotlib.colors.LogNorm(),
        # alpha={"MoBO": 0.5, "BoGraph": 1}
        # cmap=cmap,
        alpha=0.7,
        # palette=color_palette,
        ax=ax,
    )

    for model in df.model.unique():
        ax = sns.scatterplot(
            data=df[df["model"] == model][model_observed_y[model]],
            x=objectives[0],
            y=objectives[1],
            # alpha={"MoBO": 0.5, "BoGraph": 1}
            # cmap=cmap,
            # alpha=0.5,
            color=model_palette_map[model],
            label=f"{model} | Pareto optimal",
            ax=ax,
            s=50,
            marker=model_marker[model],
        )

    sns.lineplot(
        data=df[pareto_frontier_all_models],
        x=objectives[0],
        y=objectives[1],
        linestyle="--",
        label="Pareto frontier (all models)",
        alpha=1,
        ax=ax,
        color="red",
    )
    ax.set(xlabel=objectives[0], ylabel=objectives[1], title=title)
    plt.close()

    return fig


def mobo_exploration_plot(
    df: pd.DataFrame, objectives: Tuple[str, str], num_of_cols: int = 3
) -> Figure:
    models = df["model"].unique()

    fig, axes = plt.subplots(1, num_of_cols, figsize=(21, 6), sharex=True, sharey=True)
    for i, model_name in enumerate(models):
        axes[i].set_title(f"Model: {model_name}")
        sns.scatterplot(
            data=df[df["model"] == model_name],
            x=objectives[0],
            y=objectives[1],
            hue="step",
            ax=axes[i],
            # palette=sns.cubehelix_palette(),
        )
    plt.close()

    return fig


def save_figure(*, filename: str, chapter: str, figure: Figure) -> None:
    """Utility to save figures in multiple format."""
    if chapter not in _CHAPTERS:
        raise ValueError(f"Unknown chapter provided: {chapter}, supported chapters: {_CHAPTERS}")
    for format in ['png', 'pdf', 'svg']:
        output = _DEFAULT_OUTPUT_LOCATION / chapter / "Figures" / f"{filename}.{format}"
        print("Saving: ", output)
        figure.savefig(str(output), bbox_inches="tight", format=f"{format}")
