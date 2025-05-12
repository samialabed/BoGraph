import logging
from typing import Set

import networkx as nx
import pandas as pd
import torch

from autorocks.optimizer.bograph import bobn
from sysgym.params import ParamsSpace

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    """Evaluates a proposed DAG against GOLEM scoring function."""

    def __init__(
        self,
        dag: nx.DiGraph,
        params: ParamsSpace,
        objectives: Set[str],
        *,
        l1_penalty_cof: float = 2e-2,
        dag_penalty: float = 5.0,
    ):
        """Create a DAG evaluator."""

        # self._num_nodes = dag.number_of_nodes()
        dag = dag.copy()
        assert dag.is_directed(), "Expecting a directed graph"
        self.connected_nodes = [node for node in dag.nodes() if dag.degree(node) > 0]
        self._num_nodes = len(self.connected_nodes)
        nodes_to_remove = set(dag.nodes) - set(self.connected_nodes)
        for node in nodes_to_remove:
            dag.remove_node(node)

        adj_matrix = torch.tensor(nx.to_pandas_adjacency(dag).values)
        self._adj_matrix = self._zero_diag(adj_matrix)

        self.h = (
            torch.trace(torch.matrix_exp(self._adj_matrix * self._adj_matrix))
            - self._num_nodes
        )
        self.dag_penalty = dag_penalty * self.h
        self.l1_penalty = l1_penalty_cof * torch.norm(self._adj_matrix, p=1)

        self.likelihood = torch.tensor([0])

        bobn_dag = bobn.BoBn(dag, params, objectives)
        self.max_dim = bobn_dag.max_dim
        logging.info(f"Max dimension: {self.max_dim}")

    def score(self, X: pd.DataFrame) -> torch.Tensor:
        """Evaluate a DAG against GOLEM scoring function."""
        """Build tensorflow graph."""
        # Placeholders and variables
        # Likelihood, penalty terms and score
        X = torch.tensor(X[self.connected_nodes].values)
        self.likelihood = self._compute_likelihood(X)
        score = self.likelihood + self.l1_penalty + self.dag_penalty
        logging.log(
            logging.INFO,
            "Score: {:.4f}, Likelihood: {:.4f}".format(score, self.likelihood),
        )
        return score

    def _zero_diag(self, matrix: torch.Tensor) -> torch.Tensor:
        """Sets the diagonals of B to zero.

        Returns:
            * [d, d] weighted matrix of zeros.
        """
        return (torch.ones(self._num_nodes) - torch.eye(self._num_nodes)) * matrix

    def _compute_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Computes (negative log) likelihood in the linear Gaussian case.

        Returns: torch.Tensor: Likelihood term (scalar-valued).
        Note: Assumes equal noise variances
        """
        return (
            0.5
            * self._num_nodes
            * torch.log(torch.square(torch.linalg.norm(X - X @ self._adj_matrix)))
            - torch.linalg.slogdet(torch.eye(self._num_nodes) - self._adj_matrix)[1]
        )
