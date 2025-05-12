import logging
from pathlib import Path

from pandas import DataFrame

import autorocks.dir_struct as ds
from autorocks.data.loader.exp_dao import ModelsComparisonData
from autorocks.data.loader.model_result_aggregator import (
    combine_all_exp_into_model_summary,
)
from autorocks.data.loader.utils import ls_subdir

LOG = logging.getLogger()


def create_all_models_comparison_dataset(
    exp_dir: Path,
    force_recompute: bool = False,
    skip_models=None,
    save_results: bool = True,
) -> ModelsComparisonData:
    """Create a dataset comparing all models under one "experiment group".
    e.g. SingleTaskGP (repeated x5) vs StaticBoGraph (repeated x4)
    Experiment group = Env/Objective[s]/iters/<models>
    """
    if skip_models is None:
        skip_models = {}
    # TODO: just reuse the saved in memory result
    models_dir = ls_subdir(exp_dir)

    model_perf = DataFrame()
    sys_observations = DataFrame()
    param_df = DataFrame()

    for model_dir in models_dir:
        model_name = model_dir.name
        if model_name in skip_models:
            continue
        model_data = combine_all_exp_into_model_summary(
            path_to_model_exp=model_dir,
            force_recompute=force_recompute,
            save_results=save_results,
        )
        # Column "model" is inferred from the name of the directory stucture.
        model_data.set_col("model", model_name)

        param_df = param_df.append(model_data.sys_params, ignore_index=True)
        model_perf = model_perf.append(model_data.model_performance, ignore_index=True)
        sys_observations = sys_observations.append(
            model_data.sys_observations, ignore_index=True
        )

    model_comparison_data = ModelsComparisonData(
        model_performance=model_perf,
        sys_observations=sys_observations,
        sys_params=param_df,
    )
    if save_results:
        model_comparison_data.save(exp_dir)

    return model_comparison_data


if __name__ == "__main__":
    create_all_models_comparison_dataset(
        exp_dir=Path(ds.LocalResultDir / "branin/synthetic/2_params"),
        force_recompute=False,
    )
