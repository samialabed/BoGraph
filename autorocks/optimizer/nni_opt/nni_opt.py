from typing import Dict

import nni

from autorocks.envs.env_state import EnvState
from autorocks.optimizer.optimizer_abc import Optimizer


class NNIOptimizer(Optimizer):
    def optimize_space(self) -> Dict[str, any]:
        # get params from tuner
        params = nni.get_next_parameter()
        # ensure we are getting params in right order
        params = self.param_space.dict_to_numpy(params)
        params = self.param_space.numpy_to_dict(params)
        # for (k, v) in params.items():
        #     params[k] = self._param_space[v].numpy_to_dict(v)
        self.ctx.logger.debug(f"Next parameter to test: {params}")
        self.logger.info(f"Next suggested configurations: {params}.")
        return params

    def observe_state(self, state: EnvState):
        super().observe_state(state)
        nni.report_final_result(
            {"default": self.observed_states.optimization_target[-1].item()}
        )
