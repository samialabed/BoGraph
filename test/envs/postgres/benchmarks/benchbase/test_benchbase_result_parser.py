import unittest
from pathlib import Path

from autorocks.envs.postgres.benchmarks.benchbase import BenchbaseResult


class TestBenchbaseOutputParser(unittest.TestCase):
    def test_parsing_produce_benchbase_result(self):
        test_input = {
            "scalefactor": "1",
            "Current Timestamp (milliseconds)": 1646241382921,
            "Benchmark Type": "tpcc",
            "isolation": "TRANSACTION_SERIALIZABLE",
            "DBMS Version": "PostgreSQL 14.2 on aarch64-unknown-linux-musl, compiled by gcc (Alpine 10.3.1_git20211027) 10.3.1 20211027, 64-bit",
            "Goodput (requests/second)": 33.31620875370126,
            "terminals": "1",
            "DBMS Type": "POSTGRES",
            "Latency Distribution": {
                "95th Percentile Latency (microseconds)": 51237,
                "Maximum Latency (microseconds)": 76220,
                "Median Latency (microseconds)": 28860,
                "Minimum Latency (microseconds)": 1904,
                "25th Percentile Latency (microseconds)": 24138,
                "90th Percentile Latency (microseconds)": 42043,
                "99th Percentile Latency (microseconds)": 60269,
                "75th Percentile Latency (microseconds)": 37324,
                "Average Latency (microseconds)": 29874,
            },
            "Throughput (requests/second)": 33.4328738168708,
        }

        expected = BenchbaseResult(
            goodput=33.31620875370126,
            latency_p99=60269,
            latency_p95=51237,
            latency_avg=29874,
            latency_min=1904,
            latency_median=28860,
            latency_max=76220,
            throughput=33.4328738168708,
        )

        actual = BenchbaseResult.from_json(test_input)

        self.assertEqual(
            expected, actual, f"failed test: expected {expected}, actual {actual}"
        )

    def test_opening_file_and_parsing(self):
        file_location = (
            Path(__file__).parent
            / "example_output/tpcc/tpcc_2022-03-02_17-16-22.summary.json"
        )

        expected = BenchbaseResult(
            goodput=33.31620875370126,
            latency_p99=60269,
            latency_p95=51237,
            latency_avg=29874,
            latency_min=1904,
            latency_median=28860,
            latency_max=76220,
            throughput=33.4328738168708,
        )

        actual = BenchbaseResult.from_json_file(file_location)
        self.assertEqual(
            expected, actual, f"failed test: expected {expected}, actual {actual}"
        )

    def test_opening_file_and_parsing_result_dir(self):
        result_dir = Path(__file__).parent / "example_output/tpcc/"

        expected = BenchbaseResult(
            goodput=33.31620875370126,
            latency_p99=60269,
            latency_p95=51237,
            latency_avg=29874,
            latency_min=1904,
            latency_median=28860,
            latency_max=76220,
            throughput=33.4328738168708,
        )

        actual = BenchbaseResult.from_result_dir(result_dir)
        self.assertEqual(
            expected, actual, f"failed test: expected {expected}, actual {actual}"
        )
