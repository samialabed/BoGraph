import dataclasses
import math
from pathlib import Path

import networkx as nx
import torch
from dataclasses_json import dataclass_json
from sysgym.params import ParamsSpace, boxes

from autorocks.envs.synthetic.env_measure import TestFunctionMeasurements
from autorocks.envs.synthetic.func_abc import TestFunction
from sysgym import EnvConfig, EnvParamsDict


def make_levy_space(dim: int) -> ParamsSpace:
    fields = [
        (f"x{i}", boxes.ContinuousBox, boxes.ContinuousBox(-10, 10, default=0))
        for i in range(dim)
    ]
    return dataclasses.make_dataclass(
        "LevySpace", fields=fields, bases=(ParamsSpace,), init=False, frozen=True
    )()


def make_struct(dim: int) -> nx.DiGraph:

    levy_struct = nx.DiGraph()
    levy_struct.add_edges_from([("x0", "part1"), (f"x{dim - 1}", "part3")])
    for i in range(dim - 1):
        levy_struct.add_edge(f"x{i}", f"z{i}")
        levy_struct.add_edge(f"z{i}", "part2")

    levy_struct.add_edges_from(
        [("part1", "target"), ("part2", "target"), ("part3", "target")]
    )
    return levy_struct


@dataclass_json
@dataclasses.dataclass(frozen=True)
class LevyCfg(EnvConfig):

    dim: int
    noise_std: float = 0

    @property
    def name(self) -> str:
        return f"Levy{self.dim}D"


class LevyND(TestFunction):
    r"""Levy D test function.

    D-dimensional function
     The function is usually evaluated on the hypercube xi ∈ [-10, 10],
      for all i = 1, …, d.

    Ref: https://www.sfu.ca/~ssurjano/levy.html
    """

    def __init__(self, env_cfg: LevyCfg, artifacts_output_dir: Path):
        super().__init__(env_cfg, artifacts_output_dir)
        self._noise_std = env_cfg.noise_std

    def run(self, params: EnvParamsDict) -> TestFunctionMeasurements:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        with torch.no_grad():
            noise_factor = 0
            if self._noise_std > 0:
                noise_factor = self._noise_std * torch.rand(1)

            X = torch.tensor(params.as_numpy())
            w = 1.0 + (X - 1.0) / 4.0
            part1 = (
                torch.sin(math.pi * w[..., 0]) ** 2 + noise_factor
            )  # This is the main struct x0->part1 +
            # x0 -> i0, x1-> i1, ... x_n-1 ->i_n-1
            # all goes into part2
            intermediate_part2 = (w[..., :-1] - 1.0) ** 2 * (
                1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2
            ) + noise_factor
            part2 = torch.sum(
                intermediate_part2,
                dim=-1,
            )
            # this is just the last dim so xd -> part3
            part3 = (w[..., -1] - 1.0) ** 2 * (
                1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
            ) + noise_factor
            # part1, part2, part3 -> y

            struct = {
                "part1": float(part1),
                "part2": float(part2),
                "part3": float(part3),
                "target": float(part1 + part2 + part3),
            }

            for i in range(intermediate_part2.size()[-1]):
                struct[f"z{i}"] = float(intermediate_part2[..., i])

            return TestFunctionMeasurements(target=struct["target"], structure=struct)
