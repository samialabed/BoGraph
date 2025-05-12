import re
import time
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd
from causalnex.structure.pytorch import from_pandas
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def structure_learning_timing(
    param_targets: pd.DataFrame,
    perfs: pd.DataFrame,
    main_targets: Set[str],
    param_names: Set[str],
    verbose: bool = False,
):
    if verbose:
        print("pre-processing input")
    preprocessed_df = whole_pipeline(param_targets, perfs)
    if verbose:
        print("Starting the structure learning process")
    start_time = time.time()
    sm = from_pandas(
        preprocessed_df,
        # w_threshold=0.8,
        tabu_parent_nodes=main_targets,
        tabu_child_nodes=param_names,
    )
    sm.remove_edges_below_threshold(0.25)
    sm.add_edges_from(
        [
            ("bench_stats.cycle", "EDP", {"weight": 3}),
            ("bench_stats.avg_power", "EDP", {"weight": 3}),
        ],
        origin="expert",
    )
    duration = time.time() - start_time
    # Find max indegree
    max_indegree = max(sm.degree, key=lambda x: x[1])[1]
    return {"time": duration, "max_dim": max_indegree}


def whole_pipeline(param_targets: pd.DataFrame, perfs: pd.DataFrame) -> pd.DataFrame:
    """
    perfs: extra statistics parsed
    param_targets: parameters and objectives.
    """
    # standardize the param_targets
    scaler = StandardScaler()
    all_cols_no_idx = set(param_targets.columns) - {
        "step",
    }
    scaled_param_target = scaler.fit_transform(param_targets[all_cols_no_idx].values)
    param_targets.loc[:, all_cols_no_idx] = scaled_param_target

    # prune the prefs
    prefs_low_variance_removed_df = prune_metrics(perfs)
    metric_decomposed = sub_group_decomposer(prefs_low_variance_removed_df)

    # fianlly merge with pruned perfs_metrics
    param_targets = param_targets.merge(metric_decomposed, on=["step"])
    param_targets = param_targets.drop(
        columns=[
            "step",
        ]
    )
    return param_targets


def prune_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Remove low variance features"""
    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
    sel.fit_transform(df)
    df_pruned = df.loc[:, sel.get_support()]
    return df_pruned


def sub_group_decomposer(df: pd.DataFrame) -> pd.DataFrame:
    # Capture groups
    # Ensure all within same ranges
    scaler = StandardScaler()
    sub_group_extractor = re.compile(r".*\.([^.]+)", re.RegexFlag.IGNORECASE)
    group_extractor = re.compile(r"([^.]+)", re.RegexFlag.IGNORECASE)
    main_groups = defaultdict(list)
    sub_groups = defaultdict(list)

    # All columns minus the none static one
    idx_cols = {"step"}
    values_cols = set(df.columns) - idx_cols

    for metric in values_cols:
        groups = group_extractor.findall(metric)
        if not groups or len(groups) < 2:
            continue
        sub_name = sub_group_extractor.findall(metric)

        main_groups[groups[1]].append(sub_name[0])
        sub_groups[sub_name[0]].append(metric)

    # import pandas as pd
    metric_pruned = df.copy()
    for group, sub_group_names in main_groups.items():
        # print(f"{group} has: {len(sub_metrics)} items")
        group_vals = []
        for sub_group_name in sub_group_names:
            sub_metrics = sub_groups[sub_group_name]
            scaler = StandardScaler()
            scaled_vals = scaler.fit_transform(metric_pruned[sub_metrics].values)
            if len(sub_metrics) > 1:
                # reduce it to 1.
                transformer = FactorAnalysis(n_components=1)
                decomposed_vals = transformer.fit_transform(scaled_vals)
                group_vals.append(decomposed_vals.squeeze())
            else:
                group_vals.append(scaled_vals.squeeze())

        group_vals_ = np.vstack(group_vals).T
        group_transformer = FactorAnalysis(n_components=1)
        group_pruned_val = group_transformer.fit_transform(group_vals_)
        metric_pruned[group] = group_pruned_val

    new_cols = set(list(main_groups.keys()) + ["step"])

    metric_pruned = metric_pruned[new_cols]
    return metric_pruned
