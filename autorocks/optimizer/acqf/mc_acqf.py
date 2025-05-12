from typing import List

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    qExpectedImprovement,
    qLowerBoundMaxValueEntropy,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch import Tensor

from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.acqf.acqf_abc import (
    DiscreteMaxValueBaseWrapperABC,
    MCAcquisitionFunctionWrapperABC,
    qMOBOAcquisitionFunctionABC,
)


class qExpectedHypervolumeImprovementWrapper(qMOBOAcquisitionFunctionABC):
    def __init__(
        self,
        sampler: MCSampler,
        ref_point: List[float],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(
            acqf=qExpectedHypervolumeImprovement,
            sampler=sampler,
            ref_point=ref_point,
            optimizer_cfg=optimizer_cfg,
        )
        self.acqf = qExpectedHypervolumeImprovement

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        ref_point = torch.tensor(self.ref_point, device=observed_x.device)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=observed_y)

        return self.acqf(
            model=model,
            sampler=self.sampler,
            partitioning=partitioning,
            ref_point=self.ref_point,
        )


class qNoisyExpectedHypervolumeImprovementWrapper(qMOBOAcquisitionFunctionABC):
    def __init__(
        self,
        sampler: MCSampler,
        ref_point: List[float],
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        self.acqf = qNoisyExpectedHypervolumeImprovement

        super().__init__(
            acqf=self.acqf,
            sampler=sampler,
            ref_point=ref_point,
            optimizer_cfg=optimizer_cfg,
        )

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        """Select reference point that is slightly worse than best observe,
        the best observe is a pareto frontier, then we take the average of them,
        then we choose the highest average.
        """
        pareto_frontier = (
            is_non_dominated(observed_y).unsqueeze(1).expand(observed_y.size())
        )
        # mask dominated points as 0
        best_points = observed_y * pareto_frontier
        best_points_avg_idx = torch.argmax(torch.mean(best_points, dim=-1))
        ref_point = best_points[best_points_avg_idx] - 0.1
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=observed_y)

        return self.acqf(
            model=model,
            X_baseline=observed_x,
            sampler=self.sampler,
            partitioning=partitioning,
            ref_point=self.ref_point,
            prune_baseline=True,
        )


class qExpectedImprovementWrapper(MCAcquisitionFunctionWrapperABC):
    def __init__(
        self,
        sampler: MCSampler,
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(
            acqf=qExpectedImprovement, sampler=sampler, optimizer_cfg=optimizer_cfg
        )
        self.acqf = qExpectedImprovement

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        return self.acqf(model=model, sampler=self.sampler, best_f=observed_y.max())


class qNoisyExpectedImprovementWrapper(MCAcquisitionFunctionWrapperABC):
    def __init__(
        self,
        sampler: MCSampler,
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        self.acqf = qNoisyExpectedImprovement

        super().__init__(acqf=self.acqf, sampler=sampler, optimizer_cfg=optimizer_cfg)

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        return self.acqf(
            model=model,
            sampler=self.sampler,
            X_baseline=observed_x,
            prune_baseline=True,
            objective=self.opt_cfg.objective,
        )


class qLowerBoundMaxValueEntropyWrapper(DiscreteMaxValueBaseWrapperABC):
    def __init__(
        self, candidate_set_size: int, bounds: Tensor, optimizer_cfg: AcqfOptimizerCfg
    ):
        acqf = qLowerBoundMaxValueEntropy
        super().__init__(
            acqf=acqf,
            candidate_set_size=candidate_set_size,
            bounds=bounds,
            optimizer_cfg=optimizer_cfg,
        )
        self.acqf = acqf

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        self.candidate_set.to(observed_x)
        return self.acqf(model=model, candidate_set=self.candidate_set)


class qUpperConfidenceBoundWrapper(MCAcquisitionFunctionWrapperABC):
    def __init__(
        self,
        sampler: MCSampler,
        beta: float,
        optimizer_cfg: AcqfOptimizerCfg,
    ):
        super().__init__(
            acqf=qUpperConfidenceBound, sampler=sampler, optimizer_cfg=optimizer_cfg
        )
        self.acqf = qUpperConfidenceBound
        self.beta = beta

    def build(
        self, model: Model, observed_x: Tensor, observed_y: Tensor
    ) -> AcquisitionFunction:
        return self.acqf(model=model, sampler=self.sampler, beta=self.beta)

    def __repr__(self) -> str:
        return f"{repr(super())}, beta:{self.beta}, "
