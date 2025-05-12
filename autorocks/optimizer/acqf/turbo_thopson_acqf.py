from typing import Optional, Tuple, Type

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation import MaxPosteriorSampling
from botorch.generation.sampling import SamplingStrategy
from botorch.models.model import Model
from torch import Tensor
from torch.quasirandom import SobolEngine

from autorocks.global_flags import DEVICE
from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.acqf.turbo_acqf import TurboAcquisitionFunctionWrapperABC


class ThompsonTurboWrapper(TurboAcquisitionFunctionWrapperABC):
    def __init__(
        self,
        acqf: Type[AcquisitionFunction],
        optimizer_cfg: AcqfOptimizerCfg,
        n_candidates: Optional[int] = None,
    ):
        super().__init__(acqf, optimizer_cfg)
        self._n_candidates = n_candidates

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        if self._n_candidates is None:
            self._n_candidates = min(5000, max(2000, 200 * observed_x.shape[-1]))

        sobol = SobolEngine(self.opt_cfg.dim, scramble=True)
        pert = sobol.draw(self._n_candidates).to(dtype=torch.double, device=DEVICE)
        pert = self._tr_lb + (self._tr_ub - self._tr_lb) * pert
        # Create a perturbation mask
        prob_perturb = min(20.0 / self.opt_cfg.dim, 1.0)
        mask = (
            torch.rand(
                n_candidates, self.opt_cfg.dim, dtype=torch.double, device=DEVICE
            )
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[
            ind, torch.randint(0, self.opt_cfg.dim - 1, size=(len(ind),), device=DEVICE)
        ] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, self.opt_cfg.dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        return thompson_sampling

    def optimize(self, acqf: SamplingStrategy) -> Tuple[Tensor, Tensor]:
        assert isinstance(
            acqf, SamplingStrategy
        ), "ThompsonSampling expects a sampling strategy"
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = acqf(X_cand, num_samples=batch_size)

        return X_next, torch.zeros(1)
