from dataclasses import dataclass
from typing import Dict, Union

from dataclasses_json import dataclass_json
from sysgym import EnvMetrics


@dataclass_json
@dataclass(frozen=True)
class TestFunctionMeasurements(EnvMetrics):
    target: Union[float, int]
    structure: Dict[str, any]

    def as_flat_dict(self) -> Dict[str, any]:
        return {"target": self.target, **self.structure}
