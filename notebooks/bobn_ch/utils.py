from collections import defaultdict
from typing import Mapping, Sequence, Set

import networkx as nx
from IPython.core.display import SVG, display


def draw(graph, path=None):
    svg = nx.nx_agraph.to_agraph(graph).draw(path=path, prog="dot", format="svg")
    display(SVG(svg))


def _generate_all_pairs(parameter_nodes: Set[str], include_self: bool = False):
    """Generates all combination of paris in the set."""
    all_params_pair = set()
    for p1 in parameter_nodes:
        for p2 in parameter_nodes:
            if not include_self and p1 == p2:
                continue
            all_params_pair.add(tuple(sorted((p1, p2))))

    return all_params_pair


def _find_dconnected_subgraphs(
    dag: nx.DiGraph, parameter_nodes: Set[str], objectives: Set[str]
) -> Mapping[str, Set]:
    all_d_connected = defaultdict(set)
    # Capture all combinations of the parameters
    param_pairs = _generate_all_pairs(parameter_nodes)
    param_to_skip = set()
    unseen_params = parameter_nodes.copy()

    # Directly connect parameters that are parents of the objective with the objective.
    for obj in objectives:
        for n in dag.predecessors(obj):
            if n in parameter_nodes:
                all_d_connected[obj].add(n)
                param_to_skip.add(n)
                unseen_params.remove(n)
    for p1, p2 in param_pairs:
        if p1 in param_to_skip or p2 in param_to_skip:
            continue
        # Find all children of the parameters
        union_of_children = set(dag.successors(p1)).union(set(dag.successors(p2)))
        if not nx.d_separated(dag, {p1}, {p2}, union_of_children):
            all_d_connected[str(union_of_children)].add(p1)
            all_d_connected[str(union_of_children)].add(p2)
            all_d_connected[str(union_of_children)].update(union_of_children)
            if p1 in unseen_params:
                unseen_params.remove(p1)
            if p2 in unseen_params:
                unseen_params.remove(p2)

    for unseen_param in unseen_params:
        unseen_param_children = set(dag.successors(unseen_param))
        all_d_connected[str(unseen_param_children)].add(unseen_param)
        all_d_connected[str(unseen_param_children)].update(unseen_param_children)
    return all_d_connected


def _connect_subgraph_from_source_to_target(
    dag: nx.DiGraph, sources: Set[str], targets: Set[str]
) -> Set[str]:
    """Finds the subgraph connecting source nodes to target nodes."""
    connecting_nodes = set()
    for source in sources:
        for target in targets:
            if nx.has_path(dag, source, target):
                for node in nx.shortest_path(dag, source, target):
                    connecting_nodes.add(node)

    return connecting_nodes


def create_d_separable_subgraphs(
    dag: nx.DiGraph, *, parameter_nodes: Set[str], objectives: Set[str]
) -> Sequence[Set[str]]:
    """Helper that generates the d-separable subgraphs."""

    d_connected_subgraphs = _find_dconnected_subgraphs(dag, parameter_nodes, objectives)
    all_subgraphs = []
    for group in d_connected_subgraphs.values():
        print(f"{group=}")
        all_subgraphs.append(
            _connect_subgraph_from_source_to_target(dag, group, objectives)
        )
    return all_subgraphs
