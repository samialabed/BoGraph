import logging
from pathlib import Path
from typing import Set

import autorocks.data.loader.filenames_const as fn
from autorocks.data.loader.exp_dao import ModelExperimentsData
from autorocks.data.loader.exp_result_aggregator import get_all_iterations_as_one_exp
from autorocks.data.loader.utils import ls_subdir
from autorocks.dir_struct import LocalResultDir

LOG = logging.getLogger()


def combine_all_exp_into_model_summary(
    path_to_model_exp: Path, force_recompute: bool = False, save_results: bool = True
) -> ModelExperimentsData:
    """
    Combine all repeated experiment
     (e.g. 2022_05_13_18_32, 2022_05_18_19_18, 2022_05_18_19_36)
     into a single dataframe (model_all_exp_result)

    """
    # update if forced or if no checkpoint created (meaning first pass)
    should_create_model_summary_files = (
        force_recompute or not (path_to_model_exp / fn.PARSED_EXP).is_file()
    )
    set_of_experiments_done = {x.name for x in ls_subdir(path_to_model_exp)}

    if not should_create_model_summary_files:
        # check if there has been any change since last time we parsed
        parsed_exp = _read_parsed_file_stats(path_to_model_exp)
        if parsed_exp != set_of_experiments_done:
            LOG.info("Found new experiments %s", parsed_exp - set_of_experiments_done)
            should_create_model_summary_files = True

    if should_create_model_summary_files:
        LOG.info("Creating model summary in %s", path_to_model_exp)
        # create checkpoint file
        all_experiments_of_this_model = []
        for iteration, path_to_single_exp in enumerate(
            sorted(ls_subdir(path_to_model_exp))
        ):
            exp_data = get_all_iterations_as_one_exp(
                path_to_single_exp, force_recompute=force_recompute
            )
            exp_data.set_col(col_name="iteration", col_val=iteration)
            all_experiments_of_this_model.append(exp_data)

        model_dataset_all_exp = ModelExperimentsData.from_experiments_dataset(
            all_experiments_of_this_model
        )

        if save_results:
            model_dataset_all_exp.save(path_to_model_exp)
            _write_parsed_file_stats(path_to_model_exp, set_of_experiments_done)
        return model_dataset_all_exp
    else:  # reuse cached results
        return ModelExperimentsData.load(path_to_model_exp)


def _read_parsed_file_stats(path_to_model_exp: Path) -> Set[str]:
    parsed_stats_fp = path_to_model_exp / fn.PARSED_EXP
    assert parsed_stats_fp.is_file()

    with open(parsed_stats_fp, "r") as f:
        return set(f.readline().split(","))


def _write_parsed_file_stats(path_to_model_exp: Path, parsed_dirs: Set[str]) -> None:
    parsed_stats_fp = path_to_model_exp / fn.PARSED_EXP
    LOG.info("Creating checkpoint file %s", parsed_stats_fp)

    with open(parsed_stats_fp, "w") as f:
        f.writelines(",".join(parsed_dirs))


if __name__ == "__main__":
    combine_all_exp_into_model_summary(
        Path(LocalResultDir / "branin/synthetic/2_params/30_iter/BoTorch"),
        force_recompute=True,
    )
