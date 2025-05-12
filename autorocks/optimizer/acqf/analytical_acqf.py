from typing import List

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch import Tensor

from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.acqf.acqf_abc import (
    AcquisitionFunctionWrapperABC,
    MOBOAcquisitionFunctionABC,
)


class ExpectedImprovementWrapper(AcquisitionFunctionWrapperABC):
    # Useful blog
    # https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
    def __init__(self, optimizer_cfg: AcqfOptimizerCfg):
        self.acqf = ExpectedImprovement
        super().__init__(acqf=self.acqf, optimizer_cfg=optimizer_cfg)

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        return self.acqf(model=model, best_f=observed_y.max())

    def __repr__(self) -> str:
        return f"Acqf:{str(self.acqf)}"


class UpperConfidentBoundWrapper(AcquisitionFunctionWrapperABC):
    def __init__(self, beta: float, optimizer_cfg: AcqfOptimizerCfg):
        self.acqf = UpperConfidenceBound
        super().__init__(acqf=self.acqf, optimizer_cfg=optimizer_cfg)
        self._beta = beta

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        return self.acqf(model=model, beta=self._beta)

    def __repr__(self) -> str:
        return f"Acqf:{str(self.acqf)}, Beta: {self._beta}"


class ExpectedHypervolumeImprovementWrapper(MOBOAcquisitionFunctionABC):
    def __init__(
        self,
        ref_point: List[float],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        self.acqf = ExpectedHypervolumeImprovement
        super().__init__(
            ref_point=ref_point, acqf=self.acqf, optimizer_cfg=optimizer_cfg
        )
        self.acqf = ExpectedHypervolumeImprovement

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        ref_point = torch.tensor(self.ref_point, device=observed_x.device)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=observed_y)

        return self.acqf(
            model=model,
            partitioning=partitioning,
            ref_point=self.ref_point,
        )
