import math
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
from botorch.acquisition import AcquisitionFunction, qExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling import MCSampler
from torch import Tensor

from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.acqf.acqf_abc import AcquisitionFunctionWrapperABC


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


class TurboAcquisitionFunctionWrapperABC(AcquisitionFunctionWrapperABC, ABC):
    def __init__(
        self, acqf: Type[AcquisitionFunction], optimizer_cfg: AcqfOptimizerCfg
    ):
        super().__init__(acqf=acqf, optimizer_cfg=optimizer_cfg)
        self._turbo_state = TurboState(
            dim=optimizer_cfg.dim, batch_size=optimizer_cfg.batch_size
        )
        self._tr_lb = None
        self._tr_ub = None

    def update_turbo_state(
        self,
        lengthscale: torch.Tensor,
        observed_x: torch.Tensor,
        observed_y: torch.Tensor,
    ):
        x_center = observed_x[observed_y.argmax(), :].clone()
        weights = lengthscale.squeeze(0).detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        self._tr_lb = torch.clamp(
            x_center - weights * self._turbo_state.length / 2.0, 0.0, 1.0
        )
        self._tr_ub = torch.clamp(
            x_center + weights * self._turbo_state.length / 2.0, 0.0, 1.0
        )


class qTurboExpectedImprovementWrapper(TurboAcquisitionFunctionWrapperABC):
    def __init__(self, sampler: MCSampler, optimizer_cfg: AcqfOptimizerCfg):
        self.acqf = qExpectedImprovement
        super().__init__(acqf=self.acqf, optimizer_cfg=optimizer_cfg)
        self.sampler = sampler
        self._tr_lb = torch.zeros(optimizer_cfg.dim)
        self._tr_ub = torch.ones(optimizer_cfg.dim)

    def build(
        self,
        model: Model,
        observed_x: Tensor,
        observed_y: Tensor,
        lengthscale: Optional[torch.Tensor] = None,
    ) -> AcquisitionFunction:
        if lengthscale is None:
            lengthscale = model.covar_module.base_kernel.lengthscale
        # Scale the TR to be proportional to the lengthscales
        self.update_turbo_state(
            lengthscale=lengthscale, observed_x=observed_x, observed_y=observed_y
        )

        return self.acqf(model=model, best_f=observed_y.max(), sampler=self.sampler)

    def optimize(
        self, acqf: AcquisitionFunction, bounds: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        candidates, acqf_val = optimize_acqf(
            acq_function=acqf,
            bounds=torch.stack([self._tr_lb, self._tr_ub]),
            q=self.opt_cfg.batch_size,
            num_restarts=self.opt_cfg.num_restarts,
            raw_samples=self.opt_cfg.raw_samples,
        )
        return candidates, acqf_val

    def __repr__(self) -> str:
        return f"TurboAcqf:{repr(super())}"
