from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction


@dataclass(init=False, frozen=True)
class BuckinParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(lower_bound=-15, upper_bound=-5)
    x2: ContinuousBox = ContinuousBox(lower_bound=-3, upper_bound=3)


@dataclass_json
@dataclass(frozen=True)
class BuckinCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "Buckin"


class Buckin(TestFunction):
    r"""Buckin test function.

    Two-dimensional function
    The sixth Bukin function has many local minima, all of which lie in a ridge.


    Ref: https://www.sfu.ca/~ssurjano/bukin6.html
    """

    _optimal_value: float = 0
    _optimal_parameters = [-10, 1]

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        x1 = params["x1"]
        x2 = params["x2"]
        t1 = 100 * np.sqrt(np.abs(x2 - 0.01 * (x1**2)))
        t2 = 0.01 * np.abs(x1 + 10)

        y = t1 + t2
        return TestFunctionMeasurements(target=y, structure={"t1": t1, "t2": t2})
