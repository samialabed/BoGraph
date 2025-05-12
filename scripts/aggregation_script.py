#!/bin/env python


import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

from autorocks.data.loader.exp_result_aggregator import get_all_iterations_as_one_exp
from autorocks.data.loader.filenames_const import CONFIG_FILE
from autorocks.data.loader.utils import ls_subdir
from autorocks.dir_struct import LocalResultDir, PackageRootDir

LOG = logging.getLogger()


def parse_all_directory(in_dir: Path, out_dir: Path):
    assert in_dir.is_dir(), (
        "Expected in_path to be a path to a directory "
        "containing fragmented experiments results."
    )
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: maybe use glob to find all config file and then parse
    #  any experiment with config file, rather than this structure of ls_subdir
    for env_exp in ls_subdir(in_dir):
        if "Levy" in str(env_exp):
            for obj in ls_subdir(env_exp):
                # for bench in ls_subdir(obj):
                for params in ls_subdir(obj):
                    for iters in ls_subdir(params):
                        for optimizer in ls_subdir(iters):
                            for path_to_exp in ls_subdir(optimizer):
                                LOG.info("Aggregating %s", path_to_exp)
                                mirrored_outdir = out_dir / path_to_exp.relative_to(
                                    in_dir
                                )
                                if not mirrored_outdir.is_dir():
                                    mirrored_outdir.mkdir(parents=True, exist_ok=True)
                                    LOG.info("Created %s", mirrored_outdir)
                                    # preserve the config file
                                    shutil.copyfile(
                                        src=path_to_exp / CONFIG_FILE,
                                        dst=mirrored_outdir / CONFIG_FILE,
                                    )
                                get_all_iterations_as_one_exp(
                                    path_to_experiment=path_to_exp,
                                    out_path=mirrored_outdir,
                                    force_recompute=True,
                                    save_results=True,
                                )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""
        Script that purpose is to parse given location (default to local_execution dir)
        and produce a single CSV for each experiment rather than many folders and files
        """
    )
    parser.add_argument(
        "--in_path",
        help="Optional override path to where to scan for input data to aggregate.",
        type=Path,
        required=False,
        default=LocalResultDir,
    )

    parser.add_argument(
        "--debug", help="Run in debug mode and create more logs", action="store_true"
    )

    parser.add_argument(
        "--out_path",
        help="Path to where to store aggregated result.",
        type=Path,
        required=False,
        default=PackageRootDir / "ProcessedDataNew",
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
    parse_all_directory(in_dir=args.in_path, out_dir=args.out_path)
