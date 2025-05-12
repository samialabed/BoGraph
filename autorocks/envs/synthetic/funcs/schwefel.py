from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction


@dataclass(init=False, frozen=True)
class Schwefel6DParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)
    x2: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)
    x3: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)
    x4: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)
    x5: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)
    x6: ContinuousBox = ContinuousBox(lower_bound=-500, upper_bound=500)


@dataclass_json
@dataclass(frozen=True)
class SchwefelCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "Schwefel"


class Schwefel6D(TestFunction):
    r"""Schwefel 6D test function.

    6-dimensional function
    The Schwefel function is complex, with many local minima
    Ref: https://www.sfu.ca/~ssurjano/schwef.html
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
            xi = params[f"x{i+1}"]
            decomposed_xi = np.sin(np.abs(xi)) * xi
            structure[f"f(x{i + 1})"] = decomposed_xi
            x_values.append(decomposed_xi)

        y = constant * dimensions - np.sum(x_values)

        return TestFunctionMeasurements(target=y, structure=structure)
