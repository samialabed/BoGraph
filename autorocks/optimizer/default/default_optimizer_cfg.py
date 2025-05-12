from dataclasses import dataclass

from dataclasses_json import dataclass_json

from autorocks.optimizer.opt_configs import OptimizerConfig


@dataclass_json
@dataclass
class DefaultConfig(OptimizerConfig):
    name: str = "Default"
