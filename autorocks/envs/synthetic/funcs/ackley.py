import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dataclasses_json import dataclass_json
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ContinuousBox
from torch import Tensor

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction
from sysgym import EnvConfig, EnvParamsDict


@dataclass_json
@dataclass(init=False, frozen=True)
class AkcleyConstants:
    a: float = 20
    b: float = 0.2
    c: float = 2 * math.pi


@dataclass(init=False, frozen=True)
class Ackley6DParametersSpace(ParamsSpace):
    x1: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )
    x2: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )
    x3: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )
    x4: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )
    x5: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )
    x6: ContinuousBox = ContinuousBox(
        lower_bound=-32.768, upper_bound=32.768, default=0
    )


@dataclass_json
@dataclass(frozen=True)
class AkcleyCfg(EnvConfig):
    @property
    def name(self) -> str:
        return "Akcley"

    constants_values: AkcleyConstants = AkcleyConstants()  # default constants


class Akcley6D(TestFunction):
    r"""Akcley 6D test function.

    6-dimensional function
     The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768],
     for all i = 1, …, d, although it may also be restricted to a smaller domain.

    Ref: https://www.sfu.ca/~ssurjano/ackley.html
    """

    _optimal_value: float = 0
    _optimal_parameters = [0, 0, 0, 0, 0, 0]

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        constants = self.env_cfg.constants_values
        structure = {}
        x_power = []
        x_cos = []
        for i in range(1, len(params) + 1):
            xi = params[f"x{i}"]
            xi_power = xi**2
            xi_cos = np.cos(constants.c * xi)

            x_power.append(xi_power)
            x_cos.append(xi_cos)
            #
            # structure[f"cos(x{i})"] = xi_cos
            # structure[f"pow(x{i})"] = xi_power

        lhd = constants.a * np.exp(-constants.b * np.sqrt(np.mean(x_power)))

        rhd = np.exp(np.mean(x_cos))

        y = -lhd - rhd + constants.a + math.e

        structure["lhd"] = lhd
        structure["rhd"] = rhd

        return TestFunctionMeasurements(target=y, structure=structure)

    def forward(self, x: Tensor) -> List[Tensor]:
        constants = self.env_cfg.constants_values

        a, b, c = constants.a, constants.b, constants.c

        lhd = a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=-1)))
        rhd = torch.exp(torch.mean(torch.cos(c * x), dim=-1))
        y = -lhd - rhd + a + math.e
        return [y, lhd, rhd]
