from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union

from dataclasses_json import dataclass_json
from sysgym import EnvMetrics


class OptMode(Enum):
    maximize = "MAXIMIZE"
    minimize = "MINIMIZE"


@dataclass_json
@dataclass(frozen=True)
class OptimizationObjective(ABC):
    name: str
    opt_mode: OptMode

    def extract_opt_target_with_sign(self, state: EnvMetrics) -> Union[float, int]:
        """Handles the minimization and maximization.
        Optimizer always assume maximization, so minimization is -1 * obj"""
        sign = 1
        if self.opt_mode == OptMode.minimize:
            sign = -1

        return self.extract_obj(state) * sign

    @abstractmethod
    def extract_obj(self, env_metric: EnvMetrics) -> Union[float, int]:
        """The real extraction method"""
