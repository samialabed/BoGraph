import logging
from typing import List, Optional, Set, Tuple

import networkx as nx

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas


def ensure_parameter_node_causality(G: nx.DiGraph, param_nodes: Set[str]) -> nx.DiGraph:
    """Ensures that parameters node have no incoming edges except from parameters."""
    new_g = nx.DiGraph()
    for (from_node, to_node) in G.edges:
        if to_node in param_nodes:
            # Swap the intermediate -> param so it is param -> intermediate.
            # Correlation goes both ways, we need to ensure it is going one way only
            # from parameter to intermediate
            if from_node not in param_nodes:
                # Do not reverse if the edge is between parameters.
                new_g.add_edge(to_node, from_node)
            else:
                logging.warning(
                    "Hierarchical parameter detected: %s, %s. Removing it.",
                    from_node,
                    to_node,
                )
        else:
            new_g.add_edge(from_node, to_node)

    return new_g


def only_nodes_coming_out_of_source(G: nx.DiGraph, param_nodes: Set[str]) -> nx.DiGraph:
    """Returns a subgraph that only contains paths coming out of source nodes."""
    reachable_nodes = set()
    for param_node in param_nodes:
        # Use set union to accumulate reachable nodes from each source
        reachable_nodes |= set(nx.dfs_successors(G, param_node))

    return G.subgraph(reachable_nodes)


def postprocess_param_intermediate(G: nx.DiGraph, param_nodes: Set[str]) -> nx.DiGraph:
    """[1]Postprocessing pipeline for parameter nodes."""

    G = ensure_parameter_node_causality(G, param_nodes)

    graph_nodes = set(G.nodes)
    G.add_nodes_from(param_nodes.difference(graph_nodes))

    return only_nodes_coming_out_of_source(G, param_nodes)


def ensure_objective_causality(
    G: nx.DiGraph, target_objectives: Set[str]
) -> nx.DiGraph:
    """Ensures causality that nodes causes change to target objective."""
    # new_g = copy.deepcopy(G)
    # edges_to_remove = set()
    # for target_objective in target_objectives:
    #     # Correlation goes both ways, we need to ensure it is going one way only.
    #     out_edges = new_g.out_edges(target_objective)
    #     edges_to_remove.update(set(out_edges))
    #
    #     outgoing_subgraph = nx.DiGraph(out_edges).reverse()
    #     new_g.update(outgoing_subgraph)
    # new_g.remove_edges_from(edges_to_remove)
    # return new_g
    new_g = nx.DiGraph()
    for (from_node, to_node) in G.edges:
        if from_node in target_objectives:
            logging.info(
                "Flipping the order of: %s, to %s",
                str((from_node, to_node)),
                str((to_node, from_node)),
            )
            new_g.add_edge(to_node, from_node)
        else:
            new_g.add_edge(from_node, to_node)

    assert nx.is_directed_acyclic_graph(new_g), "Expected a DAG, found a cycle"
    for target in target_objectives:
        assert (
            len(new_g.out_edges(target)) == 0
        ), f"Expected no children of target objective, got: {list(new_g.out_edges(target))}"
        assert (
            len(new_g.in_edges(target)) > 0
        ), f"Expected at least a parent of target objective"

    return new_g


def only_nodes_connected_to_objectives(
    G: nx.DiGraph, target_objectives: Set[str]
) -> nx.DiGraph:
    """Removes all nodes in the graph G that are not connected to the sink_node.

    Args:
      G: A NetworkX graph.
      target_objectives: The optimization objectives.

    Returns:
      A graph with only nodes that are ancestors of the target objective.
    """
    reachable_nodes = set()

    for target_objective in target_objectives:
        # Remove any node not reachable to current sink node
        reachable_nodes |= set(nx.dfs_predecessors(G.reverse(), target_objective))
        reachable_nodes.add(target_objective)
    return G.subgraph(reachable_nodes)


def postprocess_intermediate_objectives(
    G: nx.DiGraph, target_nodes: Set[str]
) -> nx.DiGraph:
    """[2]Postprocessing pipeline for intermediate-objective nodes."""
    assert G.number_of_nodes() > 0, "Expected a DAG with some nodes, got 0."
    G = ensure_objective_causality(G, target_nodes)
    return only_nodes_connected_to_objectives(G, target_nodes)


def paths_from_sources_to_sinks(G: nx.DiGraph, sources: Set[str], sinks: Set[str]):
    new_g = nx.DiGraph()
    for source in sources:
        if source not in G:
            logging.warning("Source %s not found!", source)
            continue
        for sink in sinks:
            paths_to_sink = list(nx.all_simple_edge_paths(G, source, sink, cutoff=None))
            for path in paths_to_sink:
                new_g.add_edges_from(path)
    return new_g


def paths_from_sources_to_sinks_vprune(
    G: nx.DiGraph, sources: Set[str], sinks: Set[str]
):
    g = G.copy()
    only_target_left = False
    while not only_target_left:
        nodes_to_remove = {node for node, out_degree in g.out_degree if out_degree == 0}
        for sink in sinks:
            nodes_to_remove.remove(sink)
        only_target_left = len(nodes_to_remove) == 0
        g.remove_nodes_from(nodes_to_remove)
    return g


def postprocess_structure(
    G: nx.DiGraph,
    sources: Set[str],
    sinks: Set[str],
    known_edges: Optional[List[Tuple[str, str]]] = None,
) -> nx.DiGraph:
    """

    Logic:
        * 1. Ensure the causality of parameters.
        * 2. Ensure the causality of objectives
        * 3. Add any missing parameters
        * 4. Add expert defined edges

    Returns: Graph connecting all parameters to objectives.
    """
    params_to_obj_graph = ensure_parameter_node_causality(G, sources)

    # 2. Ensure the causality of the sink nodes.
    params_to_obj_graph = ensure_objective_causality(params_to_obj_graph, sinks)

    params_to_obj_graph = paths_from_sources_to_sinks_vprune(
        params_to_obj_graph, sources, sinks
    )

    # 2. Connect parameters to main objective if they aren't connected
    graph_nodes = set(params_to_obj_graph.nodes)
    missing_params = sources.difference(graph_nodes)
    if missing_params:
        print("Manually connecting param: ", missing_params)
        # If there is no path found through intermediate results (metrics)
        # add direct connection manually
        for missing_param in missing_params:
            params_to_obj_graph.add_edges_from(
                [(missing_param, sink) for sink in sinks]
            )

    # 5. Finally, add expert defined edges
    if known_edges:
        params_to_obj_graph.add_edges_from(known_edges)

    missing_params = sources.difference(set(params_to_obj_graph.nodes))
    assert (
        len(missing_params) == 0
    ), f"Expected all parameters to be connected to the graph, missing {missing_params}"

    return params_to_obj_graph
