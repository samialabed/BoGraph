import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import pandas as pd
from sysgym.envs.gem5.parsers import parse_statistics

from autorocks.data.loader.utils import ls_subdir

TOMBSTONE: str = "gem5_stats_tombstone.txt"


@dataclass
class Gem5AllStatisticsDF:
    """Hack to just maintain previous interface working"""

    system: pd.DataFrame
    performance: pd.DataFrame

    @staticmethod
    def from_csv(csv_path: Path, prefix: str = "main") -> "Gem5AllStatisticsDF":
        system_df = pd.read_csv(csv_path / f"{prefix}_gem5_sys_stats.csv", index_col=0)
        performance_df = pd.read_csv(
            csv_path / f"{prefix}_gem5_perf_stats.csv", index_col=0
        )
        return Gem5AllStatisticsDF(system=system_df, performance=performance_df)

    def set_col(self, col_name: str, value: any) -> None:
        self.system[col_name] = value
        self.performance[col_name] = value

    def save_csv(self, output: Path, prefix: str = "main"):
        self.system.to_csv(output / f"{prefix}_gem5_sys_stats.csv")
        self.performance.to_csv(output / f"{prefix}_gem5_perf_stats.csv")


def gem5_stats_combiner(
    gem5_stats: List[Gem5AllStatisticsDF],
) -> Gem5AllStatisticsDF:
    system = pd.concat([stat.system for stat in gem5_stats], axis=1)
    performance = pd.concat([stat.performance for stat in gem5_stats])
    return Gem5AllStatisticsDF(
        system=system,
        performance=performance,
    )


def per_iteration_parser(iteration_path: Path) -> Gem5AllStatisticsDF:
    # Extract the statistics and assign it a step number
    # specific_file can be replaced with exp_dir

    csv_prefix = "steps"
    tombstone = f"{csv_prefix}_{TOMBSTONE}"

    stats_files = glob.glob(str(iteration_path / "**" / "stats.txt"), recursive=True)
    if require_update(filepath=iteration_path, tombstone=tombstone):

        print(f"Steps parsing in {iteration_path}")

        extract_step_num = re.compile(r"(\d*)/env_output")

        parsed_steps = []
        all_stats = []
        for stat_file in stats_files:
            step_num = int(extract_step_num.findall(stat_file)[0])
            env_stats = parse_statistics(Path(stat_file))
            #  Hack to maintain old interface with new one
            env_stats = Gem5AllStatisticsDF(
                system=env_stats.system.as_df(),
                performance=env_stats.performance,
            )
            env_stats.set_col("step", step_num)
            all_stats.append(env_stats)
            parsed_steps.append(str(step_num))

        gem_all_statistics = gem5_stats_combiner(all_stats)
        gem_all_statistics.save_csv(iteration_path, prefix=csv_prefix)
        create_checkpoint(
            filepath=iteration_path, tombstone=tombstone, parsed_files=parsed_steps
        )
        return gem_all_statistics
    # No update required use cached results
    return Gem5AllStatisticsDF.from_csv(iteration_path, prefix=csv_prefix)


def per_model_parser(model_path: Path) -> Gem5AllStatisticsDF:
    csv_prefix = "iterations"
    tombstone = f"{csv_prefix}_{TOMBSTONE}"

    if require_update(filepath=model_path, tombstone=tombstone):
        iterations_dir = ls_subdir(model_path)
        all_iter = []
        parsed_iterations = []
        print(f"Iterations parsing in {model_path}")
        for iteration_num, iteration_dir in enumerate(iterations_dir):
            stats_in_iter = per_iteration_parser(iteration_dir)
            stats_in_iter.set_col("iteration", iteration_num)
            parsed_iterations.append(iteration_dir.name)
            all_iter.append(stats_in_iter)

        gem_all_statistics = gem5_stats_combiner(all_iter)
        gem_all_statistics.save_csv(model_path, prefix=csv_prefix)
        create_checkpoint(
            filepath=model_path, tombstone=tombstone, parsed_files=parsed_iterations
        )
        return gem_all_statistics
    # No update required use cached results
    return Gem5AllStatisticsDF.from_csv(model_path, prefix=csv_prefix)


def all_models_parser(exp_path: Path, use_cached: bool = False) -> Gem5AllStatisticsDF:
    csv_prefix = "models"
    tombstone = f"{csv_prefix}_{TOMBSTONE}"
    if use_cached:
        return Gem5AllStatisticsDF.from_csv(exp_path, prefix=csv_prefix)

    print(f"Exp parsing in {exp_path}")

    models_dir = ls_subdir(exp_path)
    all_models = []
    parsed_models = []
    for model_dir in models_dir:
        model_name = model_dir.name

        stats_in_model = per_model_parser(model_dir)
        stats_in_model.set_col("model", model_name)
        all_models.append(stats_in_model)
        parsed_models.append(model_name)

    gem_all_statistics = gem5_stats_combiner(all_models)
    gem_all_statistics.save_csv(exp_path, prefix=csv_prefix)
    create_checkpoint(
        filepath=exp_path, tombstone=tombstone, parsed_files=parsed_models
    )
    return gem_all_statistics


def require_update(filepath: Path, tombstone: str) -> bool:
    # This needs to be recursive

    # Check if it require update.
    # Either the tombstone not created
    if not (filepath / tombstone).is_file():
        print("No checkpoint file found, requiring update.")
        return True
    #  or the tombstone doesnt' contain all directories parsed before
    exp_dirs = filter(lambda x: x.name != "logs", ls_subdir(filepath))
    exp_dirs = {x.name for x in exp_dirs}
    prev_parsed_exp = _read_parsed_file_stats(filepath / tombstone)

    dir_diffs = exp_dirs - prev_parsed_exp
    if len(dir_diffs) > 0:
        print(f"Update required for {dir_diffs}")
        return True

    print(f"No update required to {tombstone}")
    return False


def _read_parsed_file_stats(path_to_tombstone: Path) -> Set[str]:
    assert path_to_tombstone.is_file()
    with open(path_to_tombstone, "r") as f:
        return set(f.readline().split(","))


def create_checkpoint(filepath: Path, tombstone: str, parsed_files: List[str]):
    with open(filepath / tombstone, "w") as f:
        f.writelines(",".join(parsed_files))
