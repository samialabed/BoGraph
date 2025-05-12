from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from sysgym.env_abc import BenchmarkConfig

from autorocks.envs.postgres.benchmarks.benchbase.available_benchmarks import (
    BenchmarkClass,
)


@dataclass
class BenchbaseCFG(BenchmarkConfig):
    bench: BenchmarkClass
    execute: bool
    load: bool
    create: bool
    scale_factor: int
    benchbase_executable_dir: Optional[Path] = None

    def as_cmd(self) -> str:
        d = asdict(self)
        d.pop("scale_factor")
        d.pop("benchbase_executable_dir")
        return " ".join(f"--{k} {v}" for (k, v) in d.items())

    @property
    def name(self) -> str:
        return str(self.bench)
