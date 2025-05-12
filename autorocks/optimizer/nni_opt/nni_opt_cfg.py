from dataclasses import dataclass

from dataclasses_json import dataclass_json

from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.optimizer.opt_configs import OptimizerConfig


@dataclass_json
@dataclass
class NNIOptConfig(OptimizerConfig):
    tuner_name: NNITuner
