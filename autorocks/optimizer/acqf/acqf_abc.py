from abc import ABC, abstractmethod
from typing import List, Tuple, Type

import torch
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling import MCSampler
from torch import Tensor

from autorocks.global_flags import DEVICE
from autorocks.optimizer.acqf.acqf_cfg import AcqfOptimizerCfg


class AcquisitionFunctionWrapperABC(ABC):
    """Wrapper that allows specifying the ACQF at config leve
    l and only build the acquisition function when needed.

    TODO: Add ACQF related only debugger.
    """

    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        self.acqf = acqf
        self.opt_cfg = optimizer_cfg

    @abstractmethod
    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        """Build the ACQF at optimization time given the training dataset."""

    def __repr__(self) -> str:
        return f"Acqf:{str(self.acqf)}, optimizer_cfg:{self.opt_cfg}"

    def optimize(
        self, acqf: AcquisitionFunction, bounds: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        candidates, acqf_val = optimize_acqf(
            acq_function=acqf,
            # bounds=bounds,
            bounds=torch.stack(
                [
                    torch.zeros(self.opt_cfg.dim, dtype=torch.double, device=DEVICE),
                    torch.ones(self.opt_cfg.dim, dtype=torch.double, device=DEVICE),
                ]
            ),
            q=self.opt_cfg.batch_size,
            num_restarts=self.opt_cfg.num_restarts,
            raw_samples=self.opt_cfg.raw_samples,
        )
        return candidates, acqf_val


class MCAcquisitionFunctionWrapperABC(AcquisitionFunctionWrapperABC, ABC):
    """Wrapper that allows specifying the monte carlo based ACQF at config leve"""

    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        sampler: MCSampler,
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(acqf=acqf, optimizer_cfg=optimizer_cfg)
        self.sampler = sampler
        self.acqf = acqf

    @abstractmethod
    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> MCAcquisitionFunction:
        """Build the ACQF at optimization time given the training dataset."""

    def __repr__(self) -> str:
        return (
            f"{repr(super())}, sampler: {self.sampler}, "
            f"drawn_samples: {self.sampler.sample_shape}"
        )


class qMOBOAcquisitionFunctionABC(MCAcquisitionFunctionWrapperABC, ABC):
    """Wrapper for qMultiObjective ACQF"""

    # Point where to draw the hyper-volume from.
    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        sampler: MCSampler,
        ref_point: List[float],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(acqf=acqf, optimizer_cfg=optimizer_cfg, sampler=sampler)
        self.ref_point = ref_point

    def __repr__(self):
        return f"{repr(super())}, ref_point: {self.ref_point}, "


class MOBOAcquisitionFunctionABC(AcquisitionFunctionWrapperABC, ABC):
    """Wrapper for MultiObjective ACQF"""

    # Point where to draw the hyper-volume from.
    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        ref_point: List[float],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(acqf=acqf, optimizer_cfg=optimizer_cfg)
        self.ref_point = ref_point

    def __repr__(self):
        return f"{repr(super())}, ref_point: {self.ref_point}, "


class DiscreteMaxValueBaseWrapperABC(AcquisitionFunctionWrapperABC, ABC):
    """Wrapper for Entropy based ACQF"""

    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        candidate_set_size: int,
        bounds: Tensor,
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(acqf=acqf, optimizer_cfg=optimizer_cfg)
        self.candidate_set = torch.rand(candidate_set_size, bounds.size(1))
