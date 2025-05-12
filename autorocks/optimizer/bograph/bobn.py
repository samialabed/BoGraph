import contextlib
import logging
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import botorch
import gpytorch.means
import networkx as nx
import pandas as pd
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import standardize, unnormalize
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from autorocks.optimizer.acqf import AcqfOptimizerCfg, qTurboExpectedImprovementWrapper
from autorocks.optimizer.bograph import bobn_utils
from sysgym.params import ParamsSpace
from sysgym.params.boxes import ParamBox

TWARGS = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

_NodeToTensor = MutableMapping[str, Tensor]


def _get_context(conservative_mode: bool) -> Callable[[], contextlib.contextmanager]:
    if conservative_mode:
        return torch.autograd.graph.save_on_cpu
    return contextlib.nullcontext


class ParametersDispenser:
    def __init__(self, params: Mapping[str, ParamBox]):
        """Simplify moving from samples to parameter and vice-versa"""
        self._param_to_idx = {}
        self._params_in_order = []

        bounds = []
        for idx, param in enumerate(params):
            bounds.append(torch.tensor(params[param].bounds).unsqueeze(1))
            self._param_to_idx[param] = torch.tensor(idx, device=TWARGS["device"])
            self._params_in_order.append(param)
        self.bounds = torch.concat(bounds, dim=-1).to(**TWARGS)

    def __call__(self, samples: Tensor, *args, **kwargs) -> _NodeToTensor:
        # knows which parameter to forward the samples to
        res = {}
        for p, idx in self._param_to_idx.items():
            res[p] = torch.index_select(samples, dim=-1, index=idx)
        return res

    @property
    def params_in_order(self) -> List[str]:
        return self._params_in_order

    @property
    def params_to_idx(self) -> Mapping[str, Tensor]:
        return self._param_to_idx


def graph_inference(
    dag: nx.DiGraph,
    subgraph: Set[str],
    X: torch.Tensor,
    param_dispenser: ParametersDispenser,
    models: Mapping[str, GPyTorchModel],
    conditioning_values: _NodeToTensor,
    conservative_mode: bool = False,
) -> Tuple[botorch.posteriors.GPyTorchPosterior, _NodeToTensor]:
    posterior_cache: _NodeToTensor = {}
    posterior_cache.update(param_dispenser(X))

    for node in nx.topological_sort(dag.subgraph(subgraph)):
        if node in posterior_cache:
            # Skip already computed results, which automatically skips the params.
            continue
        if len(dag.in_edges(node)) == 0:
            logging.warning(
                "Node %s has no parents, will be ignored in the inference.", node
            )
            continue
        # use dictionary to allow easier access to parameters at the model level
        parents_vals = []
        for parent in dag.predecessors(node):
            if parent in posterior_cache:
                # Collect the parents cached results
                parent_samples = posterior_cache[parent]
            else:
                # Use the fixed value to condition and block that path of the graph.
                # Condition based on shape of X [batch, num_samples, restarts, output]
                # But ignore the output, output size should be 1.
                parent_samples = (
                    conditioning_values[parent].broadcast_to(X.shape[:-1]).unsqueeze(-1)
                )
            parents_vals.append(parent_samples)
        parents_vals = torch.concat(parents_vals, -1)
        node_model = models[node]
        if conservative_mode:
            node_model.to(device=torch.device("cuda"))
        node_posterior = node_model.posterior(parents_vals, observation_noise=True)
        if conservative_mode:
            node_model.to(device=torch.device("cpu"))
        if dag.nodes()[node].get("is_sink"):
            return node_posterior, posterior_cache
        else:
            posterior_cache[node] = node_posterior.mean

    raise ValueError("No sink node detected")


class BoBnGraph(GPyTorchModel):
    def __init__(
        self,
        graph: nx.DiGraph,
        subgraph: Set[str],
        params: Mapping[str, ParamBox],
        objectives: Set[str],
        conservative_mode: bool = False,
        use_turbo: bool = False,
    ):
        super().__init__()
        self._graph = graph
        self._subgraph = subgraph
        self._params = params
        self._objectives = objectives
        self._param_dispenser = ParametersDispenser(self._params)
        self._conservative_mode = conservative_mode

        self.models = None
        self.conditioning_values = None
        self._num_outputs = len(objectives)
        self._use_turbo = use_turbo

        name = []
        for node in nx.topological_sort(graph.subgraph(subgraph)):
            name.append(node)
        self._name = ",".join(name)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    @property
    def name(self) -> str:
        return self._name

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> botorch.posteriors.GPyTorchPosterior:
        assert self.models, "Models have not been initialized"
        assert self.conditioning_values, "Conditioning values have not been initialized"

        posterior, _ = graph_inference(
            dag=self._graph,
            subgraph=self._subgraph,
            X=X,
            param_dispenser=self._param_dispenser,
            models=self.models,
            conditioning_values=self.conditioning_values,
        )
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def _load_relevant_models_to_device(self, models: Mapping[str, Model], device: str):
        for node in self._subgraph:
            if node not in self._params:
                models[node].to(device=torch.device(device))

    def generate_initial_points(self, n: int) -> Mapping[str, Tensor]:
        """Generate the initial points to explore teh decomposed space."""
        return self._param_dispenser(
            botorch.utils.draw_sobol_samples(
                bounds=torch.stack(
                    [
                        torch.zeros(len(self._params)),
                        torch.ones(len(self._params)),
                    ]
                ).to(**TWARGS),
                n=n,
                q=1,
            )
        )

    def optimize(
        self,
        observation: Mapping[str, float],
        models: Mapping[str, Model],
        conditioning_values: _NodeToTensor,
    ) -> Tuple[Mapping[str, Any], torch.Tensor]:
        self.conditioning_values = conditioning_values
        self.models = models

        x_baseline = [observation[param] for param in self._params]
        x_baseline = torch.tensor(x_baseline).T.to(**TWARGS)
        # TODO: retest normalization here.
        x_baseline = botorch.utils.transforms.normalize(
            x_baseline, self._param_dispenser.bounds
        )

        if self._conservative_mode:
            self._load_relevant_models_to_device(self.models, "cuda")
        # TODO: handle mobo and custom acquisition function
        sampler = SobolQMCNormalSampler(torch.Size([1024 * 4]))
        optimization_dim = len(self._params)
        # TODO: BEGIN REFACTOR INTO CONFIGe
        bounds = torch.stack(
            [
                torch.zeros(optimization_dim),
                torch.ones(optimization_dim),
            ]
        ).to(**TWARGS)
        if self._use_turbo:
            y_baseline = []
            lengthscale = self.lengthscale()

            for obj in self._objectives:
                y_baseline.append(observation[obj])

            y_baseline = torch.tensor(y_baseline).T.to(**TWARGS)
            acf_wrapper = qTurboExpectedImprovementWrapper(
                sampler,
                AcqfOptimizerCfg(
                    dim=len(self._params), num_restarts=16, raw_samples=1024 * 2
                ),
            )
            acf = acf_wrapper.build(
                model=self,
                observed_x=x_baseline,
                observed_y=y_baseline,
                lengthscale=lengthscale,
            )
            candidate, acqf_values = acf_wrapper.optimize(acf, bounds)
        else:
            acf = botorch.acquisition.qNoisyExpectedImprovement(
                model=self,
                X_baseline=x_baseline,
                sampler=sampler,
            )
            with _get_context(self._conservative_mode)():
                # TODO: handle categorical again.
                candidate, acqf_values = botorch.optim.optimize_acqf(
                    acf,
                    bounds=bounds,
                    q=1,
                    raw_samples=1024 * 2,
                    num_restarts=16,
                )
            # TODO: END REFACTOR
            candidate = candidate.detach()
            acqf_values = acqf_values.detach().cpu()
        if self._conservative_mode:
            self._load_relevant_models_to_device(self.models, "cpu")
        return self._param_dispenser(candidate), acqf_values

    def lengthscale(self) -> torch.Tensor:
        """Returns the weights of the parameters."""
        param_to_weight = {}
        graph = self._graph.subgraph(self._subgraph)
        for param in self._params:
            model_child = list(graph.successors(param))
            if len(model_child) != 1:
                # TODO: Figure out how to handle union of nodes. Possibly sum?
                raise NotImplementedError(f"{model_child} appears in multiple models.")
            model_child = model_child[0]
            idx = list(graph.predecessors(model_child)).index(param)
            len_scale = self.models[model_child].covar_module.base_kernel.lengthscale
            param_to_weight[param] = len_scale[:, idx].unsqueeze(0)
        weights = []
        for param in self._param_dispenser.params_in_order:
            weights.append(param_to_weight[param])

        return torch.concat(weights, -1)

    def __str__(self) -> str:
        return f"{self._subgraph}. Nodes: {self.nodes}"

    def subset_output(self, _: List[int]) -> Model:
        pass

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        pass


class BoBn:
    def __init__(
        self,
        dag: nx.DiGraph,
        params: ParamsSpace,
        objectives: Set[str],
        conservative_mode: bool = False,
        use_turbo: bool = False,
    ):
        self._max_dim = 0
        for obj in objectives:
            obj_degree = dag.in_degree(obj)
            obj_degree = obj_degree if obj_degree else 0
            assert obj_degree > 0, f"{obj} is unreachable in the DAG."
        for param in params:
            param_degree = dag.out_degree(param)
            param_degree = param_degree if param_degree else 0
            assert param_degree > 0, f"{param} has no outgoing edges."

        self._params = params
        self._objectives = objectives
        self._dag = dag
        if conservative_mode and not torch.cuda.is_available():
            logging.warning(
                "Conservative mode requested but running on CPU, ignoring the "
                "conservative mode. "
            )
            conservative_mode = False
        self._conservative_mode = conservative_mode
        self._use_turbo = use_turbo

        for objective in self._objectives:
            self._dag.nodes()[objective]["is_sink"] = True
        self._subgraphs = self._build_independent_subgraphs()
        self._param_dispenser = ParametersDispenser(params)

    @property
    def max_dim(self) -> int:
        return self._max_dim

    def _params_in_subgraph(self, subgraph: Set[str]) -> Mapping[str, ParamBox]:
        """Extract the parameters used in this subgraph"""
        params = {}
        for node in subgraph:
            if node in self._params:
                params[node] = self._params[node]
        return params

    def _build_independent_subgraphs(self) -> List[BoBnGraph]:
        """Build independent subgraphs"""
        independent_subgraphs = []
        max_dim = 0
        for subgraph in bobn_utils.create_d_separable_subgraphs(
            self._dag, parameter_nodes=set(self._params), objectives=self._objectives
        ):
            subgraph_max_dim = max(
                self._dag.subgraph(subgraph).in_degree, key=lambda x: x[1]
            )
            logging.info(
                "Independent subgraph %s, with a max dimension and node: %s",
                subgraph,
                subgraph_max_dim,
            )
            max_dim = max(max_dim, subgraph_max_dim[1])

            independent_subgraphs.append(
                BoBnGraph(
                    graph=self._dag,
                    subgraph=subgraph,
                    params=self._params_in_subgraph(subgraph),
                    objectives=self._objectives,
                    conservative_mode=self._conservative_mode,
                    use_turbo=self._use_turbo,
                )
            )
        logging.info(
            "Built independent subgraphs, with the largest dimension: %d", max_dim
        )

        self._max_dim = max_dim
        return independent_subgraphs

    def generate_initial_points(self, n: int) -> List[Mapping[str, float]]:
        """Generates `n` Random exploration points while making use of decomposition."""
        all_candidate = []
        for i in range(n):
            all_candidate.append({})
        with torch.no_grad():

            for subgraph in self._subgraphs:
                sub_candidate = subgraph.generate_initial_points(n)
                for k, v in sub_candidate.items():
                    assert (
                        k not in all_candidate
                    ), f"Key {k} detected in all candidates, subgraph: {subgraph}"

                    list_of_suggestions_for_param = unnormalize(
                        v, torch.tensor(self._params[k].bounds)
                    ).squeeze()
                    if list_of_suggestions_for_param.dim() == 0:
                        list_of_suggestions_for_param = (
                            list_of_suggestions_for_param.unsqueeze(0)
                        )
                    for idx in range(n):
                        all_candidate[idx][k] = self._params[k].from_numpy(
                            list_of_suggestions_for_param[idx]
                        )

        return all_candidate

    def build_graph_models(
        self, observations: Mapping[str, Any], preprocess_observations: bool = False
    ) -> Mapping[str, GPyTorchModel]:
        models = {}
        for node in self._dag.nodes():
            if node in self._params:
                continue

            # Build the training data from parents.
            train_y = torch.tensor(observations[node]).unsqueeze(-1).to(**TWARGS)
            train_x = []
            indices_to_normalise = []
            input_transform = None
            outcome_transform = None
            mean_module = None

            for idx, parent in enumerate(self._dag.predecessors(node)):
                parent_train_x = torch.tensor(observations[parent])
                if preprocess_observations:
                    if parent in self._params:
                        # Normalize the observations of the parameters directly.
                        parent_train_x = botorch.utils.transforms.normalize(
                            parent_train_x,
                            torch.tensor(self._params[parent].bounds),
                        )
                    else:
                        # Learn normalization of non parameters,
                        # parameters have known normalization.
                        indices_to_normalise.append(idx)
                train_x.append(parent_train_x)
            train_x = torch.vstack(train_x).T.to(**TWARGS)

            model_dims = len(list(self._dag.predecessors(node)))
            # If there are mixed observation and parameters, learn the bounds for
            # observation while using the normalized variables directly
            if preprocess_observations:
                if len(indices_to_normalise) > 0:
                    input_transform = Normalize(
                        d=model_dims,
                        indices=indices_to_normalise,
                        learn_bounds=True,
                    )
                outcome_transform = Standardize(train_y.shape[-1])
            if node in self._objectives:
                # this is a sink node, we should not use outcome_transform,
                # but rather directly standardize the prediction.
                # Outcome transform de-transforms the results.
                train_y = standardize(train_y)
                outcome_transform = None
                mean_module = gpytorch.means.LinearMean(model_dims)
            # TODO: Handle multi-task node and custom nodes.
            model = botorch.models.SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                input_transform=input_transform,
                outcome_transform=outcome_transform,
                mean_module=mean_module,
            )
            model.to(**TWARGS)
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            botorch.fit_gpytorch_mll(mll)
            model.eval()
            if input_transform is not None:
                input_transform.eval()
            if outcome_transform is not None:
                outcome_transform.eval()
            if self._conservative_mode:
                model.to(device=torch.device("cpu"))
            models[node] = model

        assert len(models) > 0, "Expected more than one model to be built, got 0."
        return models

    def build_conditioning_values(
        self, models: Mapping[str, GPyTorchModel]
    ) -> MutableMapping[str, Any]:
        """Generates conditioning values through sampling the trained models."""
        with torch.no_grad():
            samples = botorch.utils.draw_sobol_samples(
                bounds=torch.stack(
                    [
                        torch.zeros(len(self._params)),
                        torch.ones(len(self._params)),
                    ]
                ).to(**TWARGS),
                n=1024,
                q=1,
            )
            _, condition_values = graph_inference(
                dag=self._dag,
                subgraph=set(self._dag.nodes),
                X=samples,
                param_dispenser=self._param_dispenser,
                models=models,
                conditioning_values={},
                conservative_mode=self._conservative_mode,
            )
            for k, v in condition_values.items():
                # Condition on the means of all the samples
                condition_values[k] = v.mean()
        return condition_values

    def opt(
        self,
        observations: Mapping[str, Any],
        preprocess_observations: bool = True,
        return_acqf_values: bool = False,
        conditioning_values_override: Optional[Mapping[str, float]] = None,
        build_conditioning_values_by_sampling: bool = False,
    ) -> Union[Tuple[Mapping[str, Any], Mapping[str, torch.Tensor]], Mapping[str, Any]]:
        """Run the BoBn optimization loop.


        Args:
            observations: The previously observed observations dictionary,
                should match the entry in the `self._dag`.
            preprocess_observations: If enabled, add preprocessing to input and output
                data as part of the inference. Should be disabled if
                 the data comes preprocessed.
            return_acqf_values: If enabled, returns all acquisition function evaluation
            stored in a dictionary: [Subgraph: Tensor[num_restarts, value]].
            conditioning_values_override: Optional value to override the conditioning
                values instead of using sampling based approach using a fixed approach.
            build_conditioning_values_by_sampling: If true, use qusi-random sampling to
                approximate the conditioning values, otherwise use the best found
                 results.

        Return:
            The next candidate to try on the system: Mapping[str, Any],
            and optionally a Mapping[str, torch] if `return_acqf_values`, that stores
            each subgraph optimized acquisition function value.
        """
        with _get_context(self._conservative_mode)():
            models = self.build_graph_models(observations, preprocess_observations)

        conditioning_values = {}
        if build_conditioning_values_by_sampling:
            conditioning_values = self.build_conditioning_values(models)
        else:
            if isinstance(observations, pd.DataFrame):
                observation_df = observations
            else:
                observation_df = pd.DataFrame(observations)
            assert len(self._objectives) == 1
            for obj in self._objectives:
                observation_df = observation_df.loc[observation_df[obj].argmax()]
            for node in self._dag.nodes:
                conditioning_values[node] = torch.tensor(observation_df[node]).to(
                    **TWARGS
                )

        if conditioning_values_override is not None:
            for key, value in conditioning_values_override.items():
                value_to_override = conditioning_values[key]
                conditioning_values[key] = torch.tensor(
                    value,
                    dtype=value_to_override.dtype,
                    device=value_to_override.device,
                ).reshape_as(conditioning_values[key])

        all_candidate = {}
        subgraph_acqf_values = {}

        for subgraph in self._subgraphs:
            sub_candidate, acqf_values = subgraph.optimize(
                observations, models=models, conditioning_values=conditioning_values
            )
            subgraph_acqf_values[subgraph.name] = acqf_values
            for k, v in sub_candidate.items():
                assert (
                    k not in all_candidate
                ), f"Key {k} detected in all candidates, subgraph: {subgraph}"
                all_candidate[k] = self._params[k].from_numpy(
                    unnormalize(
                        v.detach().cpu(), torch.tensor(self._params[k].bounds)
                    ).squeeze()
                )

            del subgraph
        del conditioning_values
        del models
        if return_acqf_values:
            return all_candidate, subgraph_acqf_values
        return all_candidate

    def posterior(
        self,
        testing_x: torch.Tensor,
        preprocess_observations: bool,
        training_observations: Mapping[str, Any],
    ) -> botorch.posteriors.GPyTorchPosterior:
        with _get_context(self._conservative_mode)():
            models = self.build_graph_models(
                training_observations, preprocess_observations
            )

        with torch.no_grad():
            posterior, _ = graph_inference(
                dag=self._dag,
                subgraph=set(self._dag.nodes),
                X=testing_x,
                param_dispenser=self._param_dispenser,
                models=models,
                conditioning_values={},
                conservative_mode=self._conservative_mode,
            )
        return posterior

    def plot_hypothesis(
        self,
        observations: Mapping[str, Any],
        conditioning_values_hypothesis: Mapping[str, float],
    ) -> botorch.posteriors.GPyTorchPosterior:
        """Overrides the sampling results based and show posterior at that point."""
        with _get_context(self._conservative_mode)():
            models = self.build_graph_models(observations)

        conditioning_values_hypothesis_tensors: _NodeToTensor = {}
        # Completely detach edges from the graph that have fixed value
        slimmed_dag = self._dag.copy()
        for node, value in conditioning_values_hypothesis.items():
            print("Edges to remove: ", list(slimmed_dag.in_edges(node)))
            slimmed_dag.remove_edges_from(list(slimmed_dag.in_edges(node)))
            conditioning_values_hypothesis_tensors[node] = (
                torch.tensor(value).reshape(1, 1).to(**TWARGS)
            )
        slimmed_dag.remove_nodes_from(list(nx.isolates(slimmed_dag)))

        with torch.no_grad():
            samples = botorch.utils.draw_sobol_samples(
                bounds=torch.stack(
                    [
                        torch.zeros(len(self._params)),
                        torch.ones(len(self._params)),
                    ]
                ).to(**TWARGS),
                n=1024,
                q=1,
            )
            posterior, _ = graph_inference(
                dag=slimmed_dag,
                subgraph=set(slimmed_dag.nodes),
                X=samples,
                param_dispenser=self._param_dispenser,
                models=models,
                conditioning_values=conditioning_values_hypothesis_tensors,
                conservative_mode=self._conservative_mode,
            )
        return posterior
