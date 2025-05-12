import math
from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
import torch
from dataclasses_json import dataclass_json
from sysgym import EnvConfig, EnvParamsDict
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox
from torch import Tensor

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction
from autorocks.envs.synthetic.funcs.branin import BraninConstants


class BraninCurConstants(NamedTuple):
    branin: BraninConstants = BraninConstants()


@dataclass(init=False, frozen=True)
class BraninCur2DParametersSpace(ParamsSpace):

    x1: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=1)
    x2: ContinuousBox = ContinuousBox(lower_bound=0, upper_bound=1)


@dataclass_json
@dataclass(frozen=True)
class BraninCurCfg(EnvConfig):
    constants_values: BraninCurConstants = BraninCurConstants()  # default constants

    @property
    def name(self) -> str:
        return "BraninCurrin"


class BraninCur(TestFunction):
    r"""
        Two objective problem composed of the Branin and Currin functions.
        Taken from BoTorch

            Branin (rescaled):

                f(x) = (
                15*x_1 - 5.1 * (15 * x_0 - 5) ** 2 / (4 * pi ** 2) + 5 * (15 * x_0 - 5)
                / pi - 5
                ) ** 2 + (10 - 10 / (8 * pi)) * cos(15 * x_0 - 5))

            Currin:

                f(x) = (1 - exp(-1 / (2 * x_1))) * (
                2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
                ) / 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20\
    """

    _optimal_value: float = 59.36011874867746
    _optimal_parameters = [0, 0]

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""

        structure = {}
        constants = self.env_cfg.constants_values

        def _branin(x1: float, x2: float) -> float:
            # scale the params
            x1 = 15 * x1 - 5
            x2 = 15 * x2

            t1 = constants.branin.a * (
                x2
                - (constants.branin.b * (x1**2))
                + (constants.branin.c * x1)
                - constants.branin.r
            )
            t2 = constants.branin.s * (1 - constants.branin.t) * np.cos(x1)
            y = t1**2 + t2 + constants.branin.s

            structure["t1"] = t1
            structure["t1_pow2"] = t1**2
            structure["t2"] = t2
            return y

        x1_param = params["x1"]
        x2_param = params["x2"]
        branin = _branin(x1=x1_param, x2=x2_param)

        def _currin(x1: float, x2: float) -> float:
            if x2 == 0:
                factor1 = 0
            else:
                factor1 = 1 - np.exp(-1 / (2 * x2))
            numer = 2300 * (x1**3) + 1900 * (x1**2) + 2092 * x1 + 60
            denom = 100 * (x1**3) + 500 * (x1**2) + 4 * x1 + 20
            numer_denom = numer / denom
            structure["currin_factor"] = factor1
            structure["currin_denom"] = numer_denom
            return factor1 * numer_denom

        currin = _currin(x1=x1_param, x2=x2_param)

        structure["branin"] = branin
        structure["currin"] = currin

        # Linear combination of the two objective, don't use this for optimization
        y = branin * currin

        return TestFunctionMeasurements(target=y, structure=structure)

    def forward(self, x: Tensor) -> List[Tensor]:
        # branin
        constants = self.env_cfg.constants_values

        a, b, c = (
            constants.branin.a,
            constants.branin.b,
            constants.branin.c,
        )

        branin_lhd = a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=-1)))
        branin_rhd = torch.exp(torch.mean(torch.cos(c * x), dim=-1))
        branin = -branin_lhd - branin_rhd + a + math.e

        # cur
        x0 = x[..., 0]
        x1 = x[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x1))
        numer = 2300 * (x0.pow(3)) + 1900 * (x0.pow(2)) + 2092 * x0 + 60
        denom = 100 * (x0.pow(3)) + 500 * (x0.pow(2)) + 4 * x0 + 20
        numer_denom = numer / denom
        currin = factor1 * numer_denom
        y = branin * currin

        return [y, branin_lhd, branin_rhd, currin, numer_denom, factor1]
