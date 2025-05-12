from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import config, dataclass_json

from autorocks.optimizer.acqf.acqf_abc import AcquisitionFunctionWrapperABC
from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel
from autorocks.optimizer.opt_configs import OptimizerConfig


@dataclass_json
@dataclass
class BoTorchConfig(OptimizerConfig):
    surrogate_model: BoTorchModel = field(metadata=config(encoder=repr))
    acquisition_function: AcquisitionFunctionWrapperABC = field(
        metadata=config(encoder=repr)
    )
    random_iter: int  # number of random iterations before starting the optimizer
    retry: int  # number of retries if get params fail
    seed: Optional[int] = None  # TODO: Move seed into a _FLAG context
    restore_from_checkpoint: bool = False

    def __post_init__(self):
        # validate configs
        if self.retry < 0:
            raise ValueError(f"Retry should be >= 0, instead got {self.retry}.")
