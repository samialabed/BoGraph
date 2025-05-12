from dataclasses import dataclass
from typing import Dict

from sysgym import EnvMetrics, EnvParamsDict


# TODO: remove this - doesn't make sense to create it like this then separate it in
#  the optimizer
@dataclass(frozen=True)
class EnvState:
    params: EnvParamsDict
    measurements: EnvMetrics

    def as_dict(self) -> Dict[str, any]:
        params_dict = dict(self.params.items())
        measurements_dict = self.measurements.as_flat_dict()

        return {**params_dict, **measurements_dict}
