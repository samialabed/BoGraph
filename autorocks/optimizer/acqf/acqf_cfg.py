from dataclasses import dataclass
from typing import Optional

from botorch.acquisition import MCAcquisitionObjective


@dataclass
class AcqfOptimizerCfg:
    dim: int
    num_restarts: int = 13
    raw_samples: int = 512
    batch_size: int = 1
    objective: Optional[MCAcquisitionObjective] = None
