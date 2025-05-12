import logging
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_random_exponential

from autorocks.envs.postgres.benchmarks.benchbase.benchbase import Benchbase
from autorocks.envs.postgres.env_cfg import PostgresEnvConfig
from autorocks.envs.postgres.env_metrics import PostgresMetrics
from autorocks.envs.postgres.env_system_metrics import PostgresSystemMetrics
from autorocks.envs.postgres.launcher.launcher_abc import PostgresLauncher
from autorocks.envs.postgres.launcher.launcher_factory import get_launcher
from autorocks.logging_util import ENV_RUNNER_LOGGER
from sysgym import Environment, EnvParamsDict
from sysgym.wrappers.repeat_env_wrapper import repeat_env

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Failed to load psycopg2 module, if you are on mac install it")

LOG = logging.getLogger(ENV_RUNNER_LOGGER)


class Postgres(Environment):
    def __init__(self, env_cfg: PostgresEnvConfig, artifacts_output_dir: Path):
        super().__init__(env_cfg, artifacts_output_dir)
        self.benchmark = Benchbase(
            artifacts_output_dir=artifacts_output_dir, cfg=env_cfg.bench_cfg
        )
        self.launcher = get_launcher(launcher_cfgs=env_cfg.launcher_settings)
        self.run = repeat_env(num_times=env_cfg.repeat_evaluation)(self.run)

    # TODO: move retry to environment
    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60)
    )  # Retry here to clean the db if the benchmark fail
    def run(self, params: EnvParamsDict) -> PostgresMetrics:
        with self.launcher(params, self.env_cfg.launcher_settings) as env:
            LOG.info("Evaluating params: %s", params)
            bench_res = self.benchmark.evaluate(env)
            assert bench_res is not None, "Expected bench_res to contain results"
            metrics = _retrieve_metrics(env)
            assert metrics is not None, "Expected postgres to report additional metrics"
            postgres_measures = PostgresMetrics(
                bench_metrics=bench_res, system_metrics=metrics
            )
            return postgres_measures


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
def _retrieve_metrics(env: PostgresLauncher) -> PostgresSystemMetrics:
    """Query postgres for its metrics"""
    postgres_metrics_dict = {}
    conn = None
    try:
        db_name = env.cfgs.env_var.POSTGRES_DB
        with psycopg2.connect(
            database=db_name,
            user=env.cfgs.env_var.POSTGRES_USER,
            password=env.cfgs.env_var.POSTGRES_PASSWORD,
            host=env.cfgs.ip_addr,
            port=env.cfgs.port,
        ) as conn:
            name_to_metric_dict = PostgresSystemMetrics.metric_group_and_type()
            for (key, metric) in name_to_metric_dict.items():
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                sql_query = metric.sql_query(database_name=db_name)
                cursor.execute(sql_query)
                record = cursor.fetchone()
                postgres_metrics_dict[key] = metric(**record)
            return PostgresSystemMetrics(**postgres_metrics_dict)
    except Exception as e:
        LOG.error("Failed to retrieve additional metrics from postgres. Err: %s", e)
        return PostgresSystemMetrics.failed_metrics_retrieval()
    finally:
        if conn:
            conn.close()
