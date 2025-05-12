import glob
import logging
import shutil
import subprocess
from pathlib import Path
from string import Template

from timeout_decorator import timeout_decorator

from autorocks.envs.postgres.benchmarks.benchbase.cfg import BenchbaseCFG
from autorocks.envs.postgres.benchmarks.benchbase.dao import BenchbaseResult
from autorocks.envs.postgres.launcher.launcher_abc import PostgresLauncher
from autorocks.logging_util import ENV_RUNNER_LOGGER
from autorocks.utils.converters import convert_to_seconds

LOG = logging.getLogger(ENV_RUNNER_LOGGER)


class Benchbase:
    # TODO make benchmark interface
    def __init__(self, artifacts_output_dir: Path, cfg: BenchbaseCFG):
        self.cfg = cfg
        self._artifacts_output_dir = artifacts_output_dir

        if cfg.benchbase_executable_dir is None:
            current_dir = Path(__file__).parent
            self.benchbase_executable_dir = current_dir / "compiled"
            assert self.benchbase_executable_dir.is_dir(), (
                f"Expected benchbase to be compiled at {self.benchbase_executable_dir},"
                f" run benchbase_installation.sh "
            )
        else:
            self.benchbase_executable_dir = cfg.benchbase_executable_dir

        with open(
            Path(__file__).parent / "configs" / f"{self.cfg.bench}_template.xml",
            encoding="UTF-8",
        ) as bench_template_f:
            self.benchmark_template = Template(bench_template_f.read())

    @timeout_decorator.timeout(seconds=convert_to_seconds("2h"))
    def evaluate(self, env: PostgresLauncher) -> BenchbaseResult:
        # benchbase puts all results in one directory and uses timestamp to
        # differentiate between different executions - what we do here is to force
        # different iterations to be in separate folder
        tmp_output_dir = self._artifacts_output_dir / "benchbase_builds/"
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_template_filled = self.benchmark_template.substitute(
            {
                "username": env.cfgs.env_var.POSTGRES_USER,
                "password": env.cfgs.env_var.POSTGRES_PASSWORD,
                "ip_addr": env.cfgs.ip_addr,
                "port": env.cfgs.port,
                "scale_factor": self.cfg.scale_factor,
            }
        )
        filled_benchmark_output_path = tmp_output_dir / f"{self.cfg.bench}.xml"
        with open(filled_benchmark_output_path, "w") as f:
            f.write(benchmark_template_filled)

        info_log_loc = self._artifacts_output_dir / "benchmark.info"
        err_log_loc = self._artifacts_output_dir / "benchmark.err"
        with open(
            info_log_loc,
            "w",
        ) as out, open(err_log_loc, "w") as err:
            with subprocess.Popen(
                f"java -jar benchbase.jar {self.cfg.as_cmd()}  "
                f"--config {filled_benchmark_output_path} -d {tmp_output_dir}",
                stdout=out,
                stderr=err,
                cwd=str(self.benchbase_executable_dir),
                shell=True,
            ) as process:
                process.communicate()
                if len(glob.glob(str(tmp_output_dir / "*.summary.json"))) < 1:
                    LOG.error(
                        "Failed to execute benchmark: check error file://%s",
                        err_log_loc,
                    )

                    LOG.error(
                        "Failed to execute benchmark: check info file://%s",
                        info_log_loc,
                    )
                    return BenchbaseResult.bad_system()

        benchbase_res = BenchbaseResult.from_result_dir(tmp_output_dir)
        LOG.debug("Cleaning benchmark temporary files in %s", str(tmp_output_dir))
        shutil.rmtree(tmp_output_dir)
        LOG.info("Benchmark results: %s", benchbase_res)
        return benchbase_res
