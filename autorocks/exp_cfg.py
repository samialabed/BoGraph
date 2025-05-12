import time
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json
from sysgym import EnvConfig

from autorocks.optimizer.opt_configs import OptimizerConfig


@dataclass_json
@dataclass
class ExperimentConfigs:
    iterations: int
    opt_cfg: OptimizerConfig
    env_cfg: EnvConfig
    debug: bool = False  # enable debug logger output into console
    exp_dir: Optional[str] = None  # override the output directory for the experiment
    run_once: bool = False  # exist as a hack to get around NNI design
    experiment_time: str = time.strftime("%Y_%m_%d_%H_%M")
    repeat: int = 1  # Number of times to repeat the experiment evaluation
