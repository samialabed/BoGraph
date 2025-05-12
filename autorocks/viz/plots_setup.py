from typing import NamedTuple, Union

import botorch.test_functions.base
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from botorch import models, posteriors, utils

plt.style.use("ggplot")
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
sns.set_context("paper")  # , font_scale=1.5, rc={"lines.linewidth": 1.5})
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"
plt.rc("text", usetex=False)
plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc("pdf", use14corefonts=True)


def generate_data(
    n: int,
    eval_problem: botorch.test_functions.base.BaseTestProblem,
):
    r"""
    Generates the initial data for the experiments.
    Args:
        n: Number of training points.
        eval_problem: The callable used to evaluate the objective function.
    Returns:
        The train_X and train_Y. `n x d` and `n x m`.
    """
    train_x = utils.draw_sobol_samples(bounds=eval_problem.bounds, n=n, q=1).squeeze(1)
    train_obj = eval_problem(train_x)
    return train_x, train_obj


class ConfidenceRegion(NamedTuple):
    lower: torch.Tensor
    upper: torch.Tensor
    mean: torch.Tensor


def confidence_region(
    posterior: posteriors.Posterior,
) -> ConfidenceRegion:
    """
    Returns 2 standard deviations above and below the mean.

    :rtype: ConfidenceRegion
    :return: a pair of tensors of size (b x d) or (d), where
        b is the batch size and d is the dimensionality of the random
        variable. The first (second) Tensor is the lower (upper) end of
        the confidence region.
    """
    stddev = posterior.variance.sqrt()
    std2 = stddev.mul_(2)
    mean = posterior.mean

    return ConfidenceRegion(
        lower=mean.sub(std2).cpu(), upper=mean.add(std2).cpu(), mean=mean.cpu()
    )


def predict(
    model: models.model.Model, test_x: torch.Tensor, observation_noise: bool = True
) -> ConfidenceRegion:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.eval()
        model.likelihood.eval()
        # posterior = model.posterior(torch.cat([test_x, main_task],-1))
        # posterior = model.posterior(torch.tensor(data_x.values))
        posterior = model.posterior(test_x, observation_noise=observation_noise)
        return confidence_region(posterior)


def plot_test_prediction_plot(cr: ConfidenceRegion, test_y: torch.Tensor) -> plt.Figure:
    """Plot showing the mean and 95% CI for each test point."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.plot([0, 80], [0, 80], "b--", lw=2)

    lower, upper, mean = cr.lower, cr.upper, cr.mean
    yerr = torch.cat((lower.unsqueeze(0), upper.unsqueeze(0)), dim=0).squeeze(-1)
    markers, caps, bars = ax.errorbar(
        test_y.squeeze(-1).cpu(),
        mean.squeeze(-1).cpu(),
        yerr=yerr,
        fmt=".",
        capsize=4,
        elinewidth=2.0,
        ms=14,
        c="k",
        ecolor="gray",
    )
    [bar.set_alpha(0.8) for bar in bars]
    [cap.set_alpha(0.8) for cap in caps]
    ax.set_xlabel("True value", fontsize=20)
    ax.set_ylabel("Predicted value", fontsize=20)
    ax.set_aspect("equal")
    ax.grid(True)
    return fig


def plot_prediction_against_test(
    cr: ConfidenceRegion,
    test_y: torch.Tensor,
    train_y: Union[torch.Tensor, None] = None,
) -> plt.Figure:
    """
    Args:
        cr: The confidence region of the posterior.
        test_y: is the test prediction.
        train_y: is also the observed prediction.
    """
    lower, upper, mean = cr.lower, cr.upper, cr.mean
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    arg_sorted = np.argsort(test_y.squeeze().cpu(), 0)
    prediction_mean = mean[arg_sorted].squeeze().cpu()

    x_axis = np.arange(0, prediction_mean.shape[0])
    if train_y is not None:
        ax.scatter(x_axis, train_y.detach().numpy(), c="black", label="observed")

    ax.scatter(x_axis, prediction_mean.detach().numpy(), c="b", label="prediction")
    ax.fill_between(
        x_axis,
        lower[arg_sorted].squeeze(),
        upper[arg_sorted].squeeze(),
        color="skyblue",
        label="95% confidence interval",
        alpha=0.3,
    )
    ax.plot(x_axis, test_y.cpu()[arg_sorted], c="r", label="truth")
    fig.legend()
    return fig
