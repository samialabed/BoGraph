import math
from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction


@dataclass_json
@dataclass(init=False, frozen=True)
class BraninConstants:
    a: float = 1
    b: float = 5.1 / (4 * math.pi**2)
    c: float = 5 / math.pi
    r: float = 6
    s: float = 10
    t: float = 1 / (8 * math.pi)


@dataclass(init=False, frozen=True)
class BraninParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(lower_bound=-5, upper_bound=10)
    x2: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=15)


@dataclass_json
@dataclass(frozen=True)
class BraninCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "Branin"

    constants_values: BraninConstants = BraninConstants()  # default constants


class Branin(TestFunction):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):
        f(x) = a(x_2 - bx^2 + cx_1 - r)^2 + s(1-t)cost(x_1)+s

    Ref: https://www.sfu.ca/~ssurjano/branin.html
    """

    _optimal_value: float = 0.397887
    _optimal_parameters = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        x1 = params["x1"]
        x2 = params["x2"]
        constants = self.env_cfg.constants_values

        t1 = constants.a * (
            x2 - (constants.b * (x1**2)) + (constants.c * x1) - constants.r
        )
        t2 = constants.s * (1 - constants.t) * np.cos(x1)
        y = t1**2 + t2 + constants.s

        return TestFunctionMeasurements(
            target=y, structure={"t1": t1, "t1_pow2": t1**2, "t2": t2}
        )
