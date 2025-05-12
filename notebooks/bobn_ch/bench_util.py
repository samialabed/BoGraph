import dataclasses
import gc
import warnings
from typing import Any, Mapping, NamedTuple

import botorch
import torch
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, standardize
from gpytorch import ExactMarginalLogLikelihood
from sysgym.params import ParamsSpace, boxes

# warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op", category=UserWarning
)


class ExperimentResult(NamedTuple):
    model: str
    step: int
    restart: int
    candidate: Mapping[str, float]
    result: float
    best_f: float
    steps_to_best_f: int
    peak_memory: int
    runtime_ms: float


def reset_cuda() -> bool:
    from numba import cuda

    device = cuda.get_current_device()
    device.reset()
    cuda.close()
    return torch.cuda.is_available()


def clear_memory():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if "cuda" in str(obj.device):
                obj.detach().to("cpu")
                del obj
    # torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats("cuda")


def botorch_optimize(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    problem: botorch.test_functions.SyntheticTestFunction,
    tkwargs: Mapping[str, Any],
) -> torch.Tensor:
    train_x = normalize(train_x, problem.bounds.to(**tkwargs)).to(**tkwargs)
    train_y = standardize(train_y).to(**tkwargs).unsqueeze(1)
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
    ).to(**tkwargs)
    # Fit and train the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit_gpytorch_mll(mll)
    model.eval()
    acf = botorch.acquisition.qExpectedImprovement(
        model, best_f=train_y.max(), sampler=SobolQMCNormalSampler(torch.Size([512]))
    )
    new_candidate, _ = botorch.optim.optimize_acqf(
        acf,
        bounds=torch.stack(
            [
                torch.zeros(problem.dim),
                torch.ones(problem.dim),
            ]
        ).to(**tkwargs),
        q=1,
        num_restarts=16,
        raw_samples=256,
    )

    return botorch.utils.transforms.unnormalize(
        new_candidate, problem.bounds.to(**tkwargs)
    ).detach()


@dataclasses.dataclass(init=False, frozen=True)
class BraninSpace(ParamsSpace):
    x1: boxes.ContinuousBox = boxes.ContinuousBox(-5, 10)
    x2: boxes.ContinuousBox = boxes.ContinuousBox(0, 15)


@dataclasses.dataclass(init=False, frozen=True)
class AckleySpace(ParamsSpace):
    x1: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
    x2: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
    x3: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
    x4: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
    x5: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
    x6: boxes.ContinuousBox = boxes.ContinuousBox(-32.768, 32.768)
