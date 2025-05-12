from typing import Type

from autorocks.envs.synthetic.func_abc import TestFunction
from autorocks.envs.synthetic.funcs.ackley import AkcleyCfg
from autorocks.envs.synthetic.funcs.branin import BraninCfg
from autorocks.envs.synthetic.funcs.branin_currin import BraninCurCfg
from autorocks.envs.synthetic.funcs.bukin import BuckinCfg
from autorocks.envs.synthetic.funcs.forrester import ForresterCfg
from autorocks.envs.synthetic.funcs.levy import LevyCfg
from autorocks.envs.synthetic.funcs.schwefel import SchwefelCfg
from sysgym import EnvConfig


def env_from_cfg(env_cfg: EnvConfig) -> Type[TestFunction]:
    if isinstance(env_cfg, BraninCurCfg):
        from autorocks.envs.synthetic.funcs.branin_currin import BraninCur

        env_cls = BraninCur

    elif isinstance(env_cfg, BraninCfg):
        from autorocks.envs.synthetic.funcs.branin import Branin

        env_cls = Branin
    elif isinstance(env_cfg, SchwefelCfg):
        from autorocks.envs.synthetic.funcs.schwefel import Schwefel6D

        env_cls = Schwefel6D

    elif isinstance(env_cfg, BuckinCfg):
        from autorocks.envs.synthetic.funcs.bukin import Buckin

        env_cls = Buckin
    elif isinstance(env_cfg, ForresterCfg):
        from autorocks.envs.synthetic.funcs.forrester import ForresterFunction

        env_cls = ForresterFunction

    elif isinstance(env_cfg, AkcleyCfg):
        from autorocks.envs.synthetic.funcs.ackley import Akcley6D

        env_cls = Akcley6D

    elif isinstance(env_cfg, LevyCfg):
        from autorocks.envs.synthetic.funcs.levy import LevyND

        env_cls = LevyND
    else:
        raise ValueError(f"Unrecognised environment type {type(env_cfg)}")

    return env_cls
