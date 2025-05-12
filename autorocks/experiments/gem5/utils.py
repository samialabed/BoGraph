from sysgym import EnvConfig
from sysgym.envs import Gem5EnvConfig
from sysgym.envs.gem5.benchmarks.benchmark_settings import (
    BenchmarkSuite,
    Gem5BenchmarkConfig,
    MemoryType,
    Simulators,
)
from sysgym.envs.gem5.benchmarks.benchmark_tasks import MachSuiteTask
from sysgym.envs.gem5.benchmarks.pre_defained_docker_settings import (
    aladdin_docker_settings,
)

docker_settings = aladdin_docker_settings()


def gem5_env(benchmark: Gem5BenchmarkConfig) -> EnvConfig:
    env_cfg = Gem5EnvConfig(
        bench_cfg=benchmark,
        container_settings=docker_settings,
        retry_attempt=3,
    )
    return env_cfg


def gem5_benchmark_plan(task: MachSuiteTask) -> Gem5BenchmarkConfig:
    return Gem5BenchmarkConfig(
        source_dir=docker_settings.gem_workspace_dir,
        bench_suite=BenchmarkSuite.MACHSUITE,
        task=task,
        simulator=Simulators.CPU,
        memory_type=MemoryType.CACHE,
    )
