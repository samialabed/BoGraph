from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction


@dataclass(init=False, frozen=True)
class BNHParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=5)
    x2: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=3)


@dataclass_json
@dataclass(frozen=True)
class BNHCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "BNH"


class BNHFunction(TestFunction):
    r"""Bnh 2D test function multi-objectives..

    2-dimensional function

    .. math::
        f_1(x) = 4x_{1}^{2} + 4x_2^2
        f_2(x) = (x_1 - 5)^2 + (x_2 -5)^2

    Ref:
        https://www.scirp.org/(S(i43dyn45teexjx455qlt3d2q))/reference/ReferencesPapers.aspx?ReferenceID=1233853
    """

    _optimal_value: float = 0
    _optimal_parameters = [420.9687] * 6

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""

        constant = 418.0829
        dimensions = len(params)

        structure = {}

        x_values = []
        for i in range(dimensions):
            xi = params[f"x{i + 1}"]
            decomposed_xi = np.sin(np.abs(xi)) * xi
            structure[f"f(x{i + 1})"] = decomposed_xi
            x_values.append(decomposed_xi)

        y = constant * dimensions - np.sum(x_values)

        return TestFunctionMeasurements(target=y, structure=structure)
