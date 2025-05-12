import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union


@dataclass
class BenchbaseResult:
    # TODO: make a benchmark result interface
    # goodput requests/seconds
    goodput: float
    latency_p99: float
    latency_p95: float
    latency_avg: float
    latency_min: float
    latency_median: float
    latency_max: float
    # requests/seconds
    throughput: float

    @staticmethod
    def from_json(result_json: Dict[str, Union[Dict, float]]) -> "BenchbaseResult":
        latency_dist: Dict[str, float] = result_json["Latency Distribution"]
        return BenchbaseResult(
            goodput=result_json["Goodput (requests/second)"],
            latency_p99=latency_dist["99th Percentile Latency (microseconds)"],
            latency_p95=latency_dist["95th Percentile Latency (microseconds)"],
            latency_avg=latency_dist["Average Latency (microseconds)"],
            latency_min=latency_dist["Minimum Latency (microseconds)"],
            latency_median=latency_dist["Median Latency (microseconds)"],
            latency_max=latency_dist["Maximum Latency (microseconds)"],
            throughput=result_json["Throughput (requests/second)"],
        )

    @staticmethod
    def from_json_file(filepath: Path) -> "BenchbaseResult":
        with open(filepath, "rb") as f:
            json_file = json.load(f)
        return BenchbaseResult.from_json(json_file)

    @staticmethod
    def from_result_dir(result_dir: Path) -> "BenchbaseResult":
        summary_file_path = Path(glob.glob(str(result_dir / "*.summary.json"))[-1])
        return BenchbaseResult.from_json_file(summary_file_path)

    @staticmethod
    def bad_system() -> "BenchbaseResult":
        # TODO: This should be something we can make reusable across envs
        return BenchbaseResult(
            goodput=-1.0,
            latency_p99=1e10,
            latency_p95=1e10,
            latency_avg=1e10,
            latency_min=1e10,
            latency_median=1e10,
            latency_max=1e10,
            # requests/seconds
            throughput=-1.0,
        )
