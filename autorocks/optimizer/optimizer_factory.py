from autorocks.optimizer.bograph.dag_options import BoBnConfig, BoGraphConfig
from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig
from autorocks.optimizer.default.default_optimizer_cfg import DefaultConfig
from autorocks.optimizer.nni_opt.nni_opt_cfg import NNIOptConfig
from autorocks.optimizer.opt_configs import OptimizerConfig
from autorocks.optimizer.optimizer_abc import Optimizer
from autorocks.optimizer.static.static_cfg import StaticOptCfg


def optimizer_from_cfg(opt_cfg: OptimizerConfig) -> Optimizer:
    if isinstance(opt_cfg, NNIOptConfig):

        from autorocks.optimizer.nni_opt.nni_opt import NNIOptimizer

        optimizer = NNIOptimizer
    elif isinstance(opt_cfg, DefaultConfig):
        from autorocks.optimizer.default.default_optimizer import Default

        optimizer = Default
    elif isinstance(opt_cfg, BoTorchConfig):

        from autorocks.optimizer.botorch_opt.botorch_optimizer import BoTorchOptimizer

        optimizer = BoTorchOptimizer

    elif isinstance(opt_cfg, BoBnConfig):
        from autorocks.optimizer.bograph.dag_optimizer import BoBnOptimizer

        optimizer = BoBnOptimizer

    elif isinstance(opt_cfg, BoGraphConfig):

        from autorocks.optimizer.bograph.dag_optimizer import DAGOptimizer

        optimizer = DAGOptimizer
    elif isinstance(opt_cfg, StaticOptCfg):
        from autorocks.optimizer.static.static_opt import StaticOptimizer

        optimizer = StaticOptimizer
    else:
        raise ValueError(f"{opt_cfg} model is not recognised. ")

    return optimizer(cfg=opt_cfg)
