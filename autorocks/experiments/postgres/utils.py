from autorocks.envs.postgres.benchmarks.benchbase.available_benchmarks import (
    BenchmarkClass,
)
from autorocks.envs.postgres.benchmarks.benchbase.cfg import BenchbaseCFG


def create_benchmark_cfg(
    benchmark_class: BenchmarkClass, scale_factor: int
) -> BenchbaseCFG:
    return BenchbaseCFG(
        bench=benchmark_class,
        execute=True,
        load=True,
        create=True,
        scale_factor=scale_factor,
    )
