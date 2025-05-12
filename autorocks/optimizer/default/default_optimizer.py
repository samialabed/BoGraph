from typing import Dict

from autorocks.optimizer.optimizer_abc import Optimizer


class Default(Optimizer):
    """Optimizer that returns rocksdb's default parameters, used as a baseline."""

    def optimize_space(self) -> Dict[str, any]:
        res = {}
        for param in self.param_space.parameters():
            res[param.name] = param.box.default

        return res
