from dataclasses import dataclass
from typing import Dict

from dataclasses_json import dataclass_json

from autorocks.optimizer.opt_configs import OptimizerConfig


@dataclass_json
@dataclass
class StaticOptCfg(OptimizerConfig):
    static_params: Dict[str, any]
