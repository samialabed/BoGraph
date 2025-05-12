import functools
import operator
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
from causalnex.plots import plot_structure
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from pandas import DataFrame
from pygraphviz import AGraph

from autorocks.viz import causal_style_sheet as style
from autorocks.viz.causal_style_sheet import NodeType

warnings.filterwarnings("ignore")

name_extractor = re.compile("([^.]+)")


def post_process_graph(G: nx.DiGraph, sources: List[str], sinks: List[str]):
    params_to_obj_graph = nx.DiGraph()
    # 1. Keep only nodes that are connected to objectives from parameters.
    for source in sources:
        if source not in G:
            print(f"Param: {source} not found in the graph")
            continue
        # remove all in-edges going to source
        for sink in sinks:
            paths_to_sink = list(nx.all_simple_edge_paths(G, source, sink, cutoff=None))
            for path in paths_to_sink:
                params_to_obj_graph.add_edges_from(path)

    edges_to_remove = [list(params_to_obj_graph.in_edges(source)) for source in sources]
    edges_to_remove += [list(params_to_obj_graph.out_edges(sink)) for sink in sinks]
    params_to_obj_graph.remove_edges_from(
        functools.reduce(operator.iconcat, edges_to_remove, [])
    )

    return params_to_obj_graph


def sub_structure_search(
    df: DataFrame,
    main_targets: List[str],
    sub_metrics_group: Dict[str, List[str]],
    params_name: List[str],
    output_dir: Path,
):
    for key in sub_metrics_group.keys():
        # if key in   ["cpu","mem_ctrls"]:
        if key not in ["mem_ctrls"]:  # ignore these for now
            continue

        print(f"Processing graph for {key}")
        structural_search_df = structural_search_sub_targets(
            df=df,
            main_targets=main_targets,
            sub_targets=sub_metrics_group[key],
            params_name=params_name,
        )

        sm = from_pandas(
            structural_search_df,
            w_threshold=0.9,
            tabu_parent_nodes=main_targets,
            tabu_child_nodes=params_name,
        )

        with open(f"{output_dir}/{key}_gem5_structure_extra_graph.p", "wb") as f:
            pickle.dump(sm, f)

        f = f"{key}_full_graph_gem5_structure_extra.png"
        viz = plot_struct_customized(
            sm,
            graph_name=f"Structure against {key}",
            param_nodes=set(params_name),
            sink_nodes=set(main_targets),
        )
        viz.draw(str(output_dir / f), format="png")

        sm = sm.get_largest_subgraph()
        f = f"{key}_gem5_structure_extra.png"
        viz = plot_struct_customized(
            sm,
            graph_name=f"Structure against {key}",
            param_nodes=set(params_name),
            sink_nodes=set(main_targets),
        )
        viz.draw(str(output_dir / f), format="png")


def structural_search_sub_targets(
    df: DataFrame,
    main_targets: List[str],
    sub_targets: List[str],
    params_name: List[str],
) -> DataFrame:
    structure_targets = []
    structure_targets += main_targets
    structure_targets += sub_targets
    structure_targets += params_name

    structural_search_df = df[structure_targets]

    return structural_search_df


def plot_struct_customized(
    g: StructureModel, graph_name: str, param_nodes: Set[str], sink_nodes: Set[str]
) -> AGraph:
    # Making all model_nodes hexagonal with black coloring
    node_attributes = {}
    for node in g.nodes:
        if node in sink_nodes:
            node_type = NodeType.OUTPUT_NODE
        elif node in param_nodes:
            node_type = NodeType.INPUT_NODE
        else:
            node_type = NodeType.OBS_NODE
        node_attributes[node] = style.node_style(
            label=clean_node_name(node), node_type=node_type
        )

    # Customising edges
    edge_attributes = {
        (u, v): style.edge_style(
            w=2 if (u in param_nodes) or (v in sink_nodes) else 0.2,
            expert=d.get("expert", False),
        )
        for u, v, d in g.edges(data=True)
    }
    graph_attributes = style.graph_style(graph_name=graph_name)
    return plot_structure(
        g,
        # plot_options={"prog": "dot"},
        # graph_attributes=graph_attributes,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )


def clean_node_name(node) -> str:
    if node == "EDP":
        return "EDP"
    node = (
        node.replace("fft_transpose", "")
        .replace("aes_aes", "")
        .replace("spmv_ellpack", "")
    )
    name = name_extractor.findall(node)
    if len(name) > 2:
        first_part = name[-2].replace("_", " ").title().replace(" ", "")
        second_part = name[-1].replace("_", " ").title().replace(" ", "")
        name = " ".join([first_part, second_part])
        up_idx = [i for i, c in enumerate(name) if c.isspace()][-1]
        new_name = name[:up_idx] + "\n" + name[up_idx:]
    else:
        new_name = name[-1]
    cleaned_name = new_name.replace("_", " ").title()
    return cleaned_name
