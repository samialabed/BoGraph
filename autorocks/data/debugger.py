import logging
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import torch
import torch.utils.data
from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader

from autorocks.data.dataset import BOSystemDataset
from autorocks.optimizer.acqf.acqf_abc import AcquisitionFunctionWrapperABC
from autorocks.optimizer.bograph.dag_dao.model_nodes.model_node_abc import (
    ModelNode,
    PredictionResult,
)

LOG = logging.getLogger()


@dataclass
class AnalysisResult:
    res: PredictionResult
    fig: Optional[Figure] = None


class ModelScore(NamedTuple):
    RMSE: Tensor
    MSE: Tensor


class OfflineDebugger:
    def __init__(
        self,
        observed_dataset: BOSystemDataset,
        model_callable: Callable[[Tensor, Tensor], ModelNode],
        shuffle_dataset: bool = False,
        separate_dataset: Optional[BOSystemDataset] = None,
    ):
        self._whole_dataset = observed_dataset
        self.model_callable = model_callable

        self.shuffle_training = shuffle_dataset
        if not separate_dataset:
            train_size = int(0.8 * len(self._whole_dataset))
            test_size = len(self._whole_dataset) - train_size
            self.train_set, self.test_set = torch.utils.data.random_split(
                self._whole_dataset, [train_size, test_size]
            )
        else:
            # Use everything in the dataset to test
            # While the training set will be limited by a call to set(model, iter)
            self.train_set, self.test_set = self._whole_dataset, separate_dataset[:]

    def observe_x_step(self, step: int) -> ModelNode:
        """return a model fitted after observing that number of steps"""
        train_dataloader = DataLoader(
            self.train_set, batch_size=step, shuffle=self.shuffle_training
        )
        train_x, train_y = next(iter(train_dataloader))
        return self.model_callable(train_x, train_y)

    def score_per_step(self, max_step: int, plot: bool = False) -> List[ModelScore]:
        """returns the MSE/RSE per step"""
        scores = []
        for step in range(max_step):
            score = self.score_at_step(step)
            scores.append(score)
        if plot:
            print("TODO: not implemented plotting")

        return scores

    def score_at_step(self, step: int) -> ModelScore:
        """Return the score of the model from observing `step` number of observations"""
        model = self.observe_x_step(step=step)
        test_x, test_y = self.test_set[:]

        with torch.no_grad():
            predicted_mean = model.posterior(X=test_x).mean.squeeze()
            mse = torch.mean(torch.abs(predicted_mean - test_y))
            rmse = torch.sqrt(torch.mean(torch.pow(predicted_mean - test_y, 2)))

        rmse = rmse.detach().cpu()
        mse = mse.detach().cpu()
        print(f"RMSE = {rmse}, MSE = {mse}")
        return ModelScore(RMSE=rmse, MSE=mse)

    def regression_at_step(
        self,
        step: int,
        acqf_wrapper: Optional[AcquisitionFunctionWrapperABC] = None,
    ) -> Figure:
        """Perform regression using the model at this step.

        if `acqf_wrapper` is provided, it will show the acqf_line and log the results.
        """
        model = self.observe_x_step(step=step)
        train_x, train_y = model.train_x, model.train_y
        pred = model.predict(X=train_x).as_numpy()

        # Visualize what has been observed
        pca = PCA(n_components=1)
        decomposed_x = pca.fit_transform(train_x, train_y)
        sorted_x = np.argsort(decomposed_x.squeeze())

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(
            decomposed_x[sorted_x],
            train_y[sorted_x],
            "k*",
            label="Observed data points",
        )
        ax.plot(
            decomposed_x[sorted_x],
            pred.mean[sorted_x],
            color="r",
            label="Model fit",
        )
        ax.fill_between(
            decomposed_x[sorted_x].squeeze(),
            y1=pred.lower[sorted_x],
            y2=pred.upper[sorted_x],
            alpha=0.5,
            label="CI",
        )
        if acqf_wrapper:
            acqf = acqf_wrapper.build(
                model=model,
                observed_x=train_x,
                observed_y=train_y,
            )

            candidates, acqf_val = optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack(
                    [
                        torch.zeros(train_x.shape[-1], dtype=torch.double),
                        torch.ones(train_x.shape[-1], dtype=torch.double),
                    ]
                ),
                q=1,
                num_restarts=12,
                raw_samples=1024,
            )

            print(f"Candidate: {candidates}, ACQF_Value: {acqf_val}")
            candidates = pca.transform(candidates)
            ax.axvline(x=candidates, label="ACQF Chosen point", color="g")

        ax.set(xlabel="Compressed X", ylabel="System objective", title="Observed data")
        ax.legend()
        plt.close()

        return fig

    def regress_against_all_dataset(
        self, step: int, plot: bool = True
    ) -> AnalysisResult:
        """Compare the prediction of the model fitted using `step` number of
        observations against all the dataset"""
        model = self.observe_x_step(step=step)
        test_x, test_y = self.test_set[:]
        pred = model.predict(X=test_x).as_numpy()
        outcome_diff = test_y - pred.mean
        LOG.info("Outcome difference: %s", outcome_diff)
        fig = None
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            mean_sorted_args = torch.argsort(test_y.squeeze())
            ax.plot(test_y, test_y, color="r", label="Ideal outcome")
            ax.plot(
                test_y[mean_sorted_args],
                pred.mean[mean_sorted_args],
                label="Prediction to outcome",
            )
            # Shade between the lower and upper confidence bounds
            ax.fill_between(
                test_y[mean_sorted_args].squeeze(),
                y1=pred.lower[mean_sorted_args],
                y2=pred.upper[mean_sorted_args],
                alpha=0.5,
                label="CI",
            )
            ax.set(xlabel="Actual outcome", ylabel="Predicted outcome")
            ax.legend()
            plt.close()

        return AnalysisResult(fig=fig, res=pred)

    def regress_on_top_of_test(self, step: int) -> Figure:
        """Visualize regression after observing `step` observations,
        compare it to real data"""
        # Visualize what has been observed
        pca = PCA(n_components=1)
        test_x, test_y = self.test_set[:]

        model = self.observe_x_step(step=step)

        decomposed_x = pca.fit_transform(test_x, test_y)
        sorted_x = np.argsort(decomposed_x.squeeze())

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(
            decomposed_x[sorted_x], test_y[sorted_x], color="g", label="Actual data"
        )

        posterior = model.predict(X=test_x).as_numpy()

        ax.plot(
            decomposed_x[sorted_x],
            posterior.mean[sorted_x],
            color="r",
            label="Model prediction",
        )
        ax.fill_between(
            decomposed_x[sorted_x].squeeze(),
            y1=posterior.lower[sorted_x],
            y2=posterior.upper[sorted_x],
            alpha=0.5,
            label="CI",
        )

        ax.set(
            xlabel="Compressed X",
            ylabel="System objective",
            title="Model prediction vs test data",
        )
        ax.legend()
        plt.close()
        return fig
