from typing import List

import torch
from sysgym.params import ParamsSpace

from autorocks.envs.objective_dao import OptimizationObjective
from autorocks.execution.nni_launcher.nni_tuners import NNITuner
from autorocks.optimizer import acqf
from autorocks.optimizer.botorch_opt.opt_config import BoTorchModel
from autorocks.optimizer.opt_configs import OptimizerConfig


def default(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
):

    from autorocks.optimizer.default.default_optimizer_cfg import DefaultConfig

    return DefaultConfig(param_space=param_schema, opt_objectives=opt_obj)


def nni_opt(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
    tuner_name: NNITuner,
) -> OptimizerConfig:

    from autorocks.optimizer.nni_opt.nni_opt_cfg import NNIOptConfig

    return NNIOptConfig(
        param_space=param_schema,
        tuner_name=tuner_name,
        opt_objectives=opt_obj,
        name=str(tuner_name),
    )


def turbo(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
    surrogate_model: BoTorchModel,
) -> OptimizerConfig:
    from botorch.sampling import SobolQMCNormalSampler

    from autorocks.optimizer.acqf import qTurboExpectedImprovementWrapper
    from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig

    return BoTorchConfig(
        name=f"BoTorch_{surrogate_model.name}",
        param_space=param_schema,
        surrogate_model=surrogate_model,
        acquisition_function=qTurboExpectedImprovementWrapper(
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
            optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_schema.dimensions),
        ),
        opt_objectives=opt_obj,
        random_iter=10,
        retry=3,
    )


def botorch(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
    surrogate_model: BoTorchModel,
) -> OptimizerConfig:

    from botorch.sampling import SobolQMCNormalSampler

    from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig

    return BoTorchConfig(
        name=f"BoTorch_{surrogate_model.name}",
        param_space=param_schema,
        surrogate_model=surrogate_model,
        acquisition_function=acqf.qExpectedImprovementWrapper(
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
            optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_schema.dimensions),
        ),
        opt_objectives=opt_obj,
        random_iter=10,
        retry=1,
    )


def botorch_mt_all(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
    surrogate_model: BoTorchModel,
    weighting: torch.Tensor,
) -> OptimizerConfig:

    from botorch import acquisition, sampling

    from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig

    return BoTorchConfig(
        name=f"BoTorch_{surrogate_model.name}",
        param_space=param_schema,
        surrogate_model=surrogate_model,
        acquisition_function=acqf.qNoisyExpectedImprovementWrapper(
            sampler=sampling.SobolQMCNormalSampler(
                # https://github.com/pytorch/botorch/pull/1037
                # batch_range = (0, -1) so collapse_batch_dims works correctly
                num_samples=1024,
                batch_range=(0, -1),
            ),
            optimizer_cfg=acqf.AcqfOptimizerCfg(
                dim=param_schema.dimensions,
                objective=acquisition.LinearMCObjective(weighting),
            ),
        ),
        opt_objectives=opt_obj,
        random_iter=2,
        retry=1,
    )


def mobo_botorch(
    param_schema: ParamsSpace,
    opt_obj: List[OptimizationObjective],
    surrogate_model: BoTorchModel,
) -> OptimizerConfig:

    from botorch.sampling import SobolQMCNormalSampler

    from autorocks.optimizer.botorch_opt.opt_config import BoTorchConfig

    return BoTorchConfig(
        name=f"MOBO_{surrogate_model.name}",
        param_space=param_schema,
        surrogate_model=surrogate_model,
        acquisition_function=acqf.qNoisyExpectedHypervolumeImprovementWrapper(
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
            ref_point=[0.0, 0.0],
            optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_schema.dimensions),
        ),
        opt_objectives=opt_obj,
        random_iter=10,
        retry=3,
    )
