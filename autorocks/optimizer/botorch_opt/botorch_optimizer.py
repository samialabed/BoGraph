import logging
from typing import Dict, Tuple

import numpy as np
import torch

# from botorch.models import transforms
from botorch.models.model import Model
from botorch.utils import transforms
from tenacity import before_sleep_log, retry, stop_after_attempt
from torch import Tensor

from autorocks.global_flags import DEVICE
from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig
from autorocks.optimizer.optimizer_abc import Optimizer


class BoTorchOptimizer(Optimizer):
    """Optimizer that provide a BoTorch wrapper."""

    def __init__(self, cfg: BoTorchConfig):
        super().__init__(cfg=cfg)
        self._batch_size = 1
        # _burn_in_iterations: num of iterations to do randomly
        self._burn_in_iterations = cfg.random_iter
        self.surrogate_model = cfg.surrogate_model
        self._acqf_wrapper = cfg.acquisition_function

        self.bo_loop = retry(
            stop=stop_after_attempt(self.cfg.retry),
            reraise=True,
            before_sleep=before_sleep_log(self.logger, logging.DEBUG),
        )(self._bo_loop)
        self._bounds = torch.from_numpy(self.param_space.bounds()).T.to(
            device=DEVICE, dtype=torch.double
        )
        # self._input_normalizer = transforms.Normalize(
        #     d=self._bounds.shape[1], bounds=self._bounds
        # )
        # self._out_standardizer = transforms.Standardize(
        #     m=len(self.obj_manager.objectives)
        # )

    def optimize_space(self) -> Dict[str, any]:
        if self._burn_in_iterations > 0:
            # draw samples for parameter from random as starting point
            next_configurations = self.param_space.sample(num=1, seed=self.cfg.seed)
            self._burn_in_iterations -= 1
        else:
            # get next best configuration from the acquisition function
            next_configurations = self.bo_loop()
        suggested_params = self.param_space.numpy_to_dict(next_configurations)
        self.logger.info("Next suggested configurations: %s.", suggested_params)
        return suggested_params

    def _bo_loop(self) -> Tensor:
        train_x, train_y = self.normalize_standardize_obs()
        surrogate_model = self.surrogate_model.model(
            train_x=train_x,
            train_y=train_y,
            # input_transform=self._input_normalizer,
            # outcome_transform=self._out_standardizer,
        )
        optimized_candidates = self.search_model(surrogate_model, train_x, train_y)
        next_configurations = transforms.unnormalize(optimized_candidates, self._bounds)
        return next_configurations

    def search_model(
        self,
        model: Model,
        observed_xs: Tensor,
        observed_y: Tensor,
    ) -> torch.Tensor:
        acqf = self._acqf_wrapper.build(
            model=model, observed_x=observed_xs, observed_y=observed_y
        )
        candidates, _ = self._acqf_wrapper.optimize(acqf, self._bounds)
        return candidates.detach()

    def normalize_standardize_obs(self) -> Tuple[Tensor, Tensor]:
        """Get previous normalized parameters and standardized observation"""
        previous_x = np.asarray(self.observed_states.parameters)
        previous_y = np.asarray(self.observed_states.optimization_target)
        train_x = torch.tensor(previous_x, device=DEVICE, dtype=torch.double)
        train_y = torch.tensor(previous_y, device=DEVICE, dtype=torch.double)
        train_x = transforms.normalize(train_x, self._bounds)
        train_y = transforms.standardize(train_y)
        return train_x, train_y
