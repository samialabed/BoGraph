from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction


@dataclass(init=False, frozen=True)
class ForresterParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=1)
    x2: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=1)


@dataclass_json
@dataclass(frozen=True)
class ForresterCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "Forrester"


class ForresterFunction(TestFunction):
    r"""Forrester 2D test function multi-objectives..

    2-dimensional function created from two forrester formulation

    .. math::
        f_1(x) = (6x-2)^2 sin(12x-4)
        f_2(x) = 0.5 * (f_1(x)) + 10(x-0.5) + 5

    Ref:
        https://www.sfu.ca/~ssurjano/forretal08.html
    """

    @staticmethod
    def original(x):
        return np.power(6 * x - 2, 2) * np.sin(12 * x - 4)

    @staticmethod
    def alternative_form(x):
        return (0.5 * ForresterFunction.original(x)) + (10 * (x - 0.5)) + 5

    _optimal_value: float = -5.355644513470416
    _optimal_parameters = [0.7575757575757577, 0.11111111111111112]

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""

        f1 = ForresterFunction.original(params["x1"])
        f2 = ForresterFunction.alternative_form(params["x2"])
        y = f1 + f2

        return TestFunctionMeasurements(target=y, structure={"fx1": f1, "fx2": f2})
