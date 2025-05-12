from abc import abstractmethod

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from sysgym import EnvConfig, Environment, EnvParamsDict


class TestFunction(Environment):
    def __init__(self, env_cfg: EnvConfig, artifacts_output_dir):
        super().__init__(env_cfg, artifacts_output_dir)

    @abstractmethod
    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        pass
