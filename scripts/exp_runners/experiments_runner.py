import dataclasses
from dataclasses import dataclass
from time import sleep
from typing import Callable, Dict, Set

from sysgym.env_abc import BenchmarkConfig
from sysgym.utils.enum import BenchmarkTask

from autorocks.execution.manager import ExecutionPlan, runner_manager
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.optimizer.nni_opt.nni_opt_cfg import NNIOptConfig
from autorocks.optimizer.opt_configs import OptimizerConfig
from sysgym import EnvConfig


@dataclass(frozen=True)
class ExpUID:
    bench_task: BenchmarkTask
    optimizer_name: str

    def __str__(self):
        return (
            f"bench_name: {str(self.bench_task)}, "
            f"optimizer_name: {self.optimizer_name}"
        )


@dataclass(frozen=True)
class ExperimentsSkip:
    skip_exp: Set[ExpUID] = dataclasses.field(default_factory=set)
    skip_optimizer: Set[str] = dataclasses.field(default_factory=set)
    skip_bench: Set[BenchmarkTask] = dataclasses.field(default_factory=set)

    def __post_init__(self):
        if self.skip_exp:
            print(f"Skipping bench_name x optimizer tasks: {self.skip_exp}")
        if self.skip_bench:
            print(f"Skipping the following bench_name: {self.skip_bench}")
        if self.skip_optimizer:
            print(f"Skipping the following optimizers: {self.skip_optimizer}")

    def eval(self, task: ExpUID) -> bool:
        return (
            task in self.skip_exp
            or task.optimizer_name in self.skip_optimizer
            or task.bench_task in self.skip_bench
        )


class ExperimentsRunner:
    def __init__(
        self,
        iterations: int,
        optimizer_to_evaluate: Dict[str, OptimizerConfig],
        benchmarks_to_evaluate: Dict[
            BenchmarkTask, Callable[[BenchmarkTask], BenchmarkConfig]
        ],
        env_callable: Callable[[BenchmarkConfig], EnvConfig],
        exp_skip: ExperimentsSkip,
    ):
        self.iterations = iterations
        self.exp_skip = exp_skip
        self.benchmarks_to_evaluate = benchmarks_to_evaluate
        self.optimizer_to_evaluate = optimizer_to_evaluate
        self.env_callable = env_callable

    def execute(self):
        for (bench_task, bench_cfg_callable) in self.benchmarks_to_evaluate.items():
            for (opt_name, optimizer_cfg) in self.optimizer_to_evaluate.items():
                exp_id = ExpUID(bench_task=bench_task, optimizer_name=opt_name)
                if self.exp_skip.eval(exp_id):
                    print(f"Skipping: {exp_id}")
                    continue

                bench = bench_cfg_callable(bench_task)
                env_cfg = self.env_callable(bench)

                print(f"###### Evaluating experiment: {exp_id}!")

                if isinstance(optimizer_cfg, NNIOptConfig):
                    execution_plan = ExecutionPlan.NNI
                else:
                    execution_plan = ExecutionPlan.LocalExecution

                config = ExperimentConfigs(
                    iterations=self.iterations,
                    opt_cfg=optimizer_cfg,
                    env_cfg=env_cfg,
                    debug=False,
                )
                try:
                    runner_manager(config=config, execution_plan=execution_plan)
                    print(f"Finished optimizing {bench_task}. using exp: {opt_name}")
                except Exception as e:
                    print(f"Failed experiment {exp_id} with error: {e}")
                print("###############################################")
                # wait for teardown
                sleep(15)
        print("FINISHED A FULL RUN")
