import logging
from collections import defaultdict
from typing import Mapping, Sequence, Set

import networkx as nx
from IPython.core.display import SVG, display
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def draw(graph, path=None, better_label: bool = True):

    if better_label:
        for node in graph.nodes():
            graph.nodes[node]["label"] = (
                str(node)
                .replace("_", " ")
                .title()
                .replace(" ", "")
                .replace("Macros", "")
                .replace("Micros", "")
                .replace("Iops", "IOPS")
                .replace(".", "")
                .replace("Size", "Sz")
                .replace("Number", "Num")
                .replace("Level0", "L0")
            )

    fmt = "svg"
    if path is not None:
        if path.endswith(".svg"):
            fmt = "svg"
        elif path.endswith(".pdf"):
            fmt = "pdf"
        else:
            print("Unrecognized file extension", path.split(".")[-1])
    svg = nx.nx_agraph.to_agraph(graph).draw(
        path=path, prog="dot", format=fmt, args="-Gsize=10"
    )
    if path is None:
        display(SVG(svg))


def _is_sink_node(G: nx.DiGraph, node: str) -> bool:
    """Checks if a node is a sink node in a directed graph."""
    return G.out_degree(node) == 0


def find_all_sinks(G: nx.DiGraph) -> Set[int]:
    """Finds all sink nodes in a directed graph."""
    all_sinks = set()
    for node in G.nodes():
        if _is_sink_node(G, node):
            all_sinks.add(node)
    return all_sinks


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
        print(f"{p1=}: {p2=}: {union_of_children=}")
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
        logging.info("Creating a subgraph for the group: %s", group)
        all_subgraphs.append(
            _connect_subgraph_from_source_to_target(dag, group, objectives)
        )
    logging.info("Creating a %d number of d-separable subgraphs.", len(all_subgraphs))
    return all_subgraphs
