from autorocks.optimizer.optimizer_abc import PARAMS_RETURN_TYPE, Optimizer
from autorocks.optimizer.static.static_cfg import StaticOptCfg


class StaticOptimizer(Optimizer):
    def __init__(self, cfg: StaticOptCfg):
        super().__init__(cfg)
        # TO ensure pycharm correctly identfiy the type
        self.cfg = cfg

    def optimize_space(self) -> PARAMS_RETURN_TYPE:
        return self.cfg.static_params
