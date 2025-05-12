from typing import Dict, Optional

from autorocks.utils.enum import ExtendedEnum


def graph_style(graph_name: str) -> Dict[str, str]:
    return {
        "splines": "spline",  # no overlap
        "ordering": "out",
        "ratio": "fill",  # This is necessary to control the size of the image
        "size": "8,3",
        # Set the size of the final image. (this is a typical presentation size)
        "fontcolor": "#FFFFFFD9",
        "fontname": "Arial",
        "fonttype": "none",
        "fontsize": 10,
        # "labeljust": "l",
        # "labelloc": "t",
        "pad": "0",
        "dpi": 300,
        "nodesep": 0.2,
        "ranksep": "0.8 equally",
        # "label": graph_name,
        "bgcolor": "transparent",
        "rankdir": "LR",
    }


class NodeType(ExtendedEnum):
    INPUT_NODE = "input_node"
    OUTPUT_NODE = "output_node"
    OBS_NODE = "obs_node"


def node_style(
    label: str,
    node_type: NodeType,
    override_color: Optional[str] = None,
) -> Dict[str, str]:
    if node_type.value == NodeType.INPUT_NODE.value:
        color = "white"
        shape = "oval"
        fontcolor = "black"
        height = 2
        width = 2.5
    elif node_type.value == NodeType.OUTPUT_NODE.value:
        color = "white"
        shape = "tripleoctagon"
        fontcolor = "black"
        height = 2
        width = 2
    else:
        shape = "oval"
        color = "grey"  # "#000000"
        height = 1.5
        width = 2
        fontcolor = "black"  # "#FFFFFFD9"
    if override_color:
        color = override_color
    return {
        "shape": shape,
        "width": width,  # 3.2,  # 2
        "fixedsize": "true",
        "height": height,
        "fillcolor": color,
        "fontcolor": fontcolor,
        "penwidth": "7",
        "color": "#4a90e2d9",  # border color
        "fontsize": 30,
        "fontname": "Arial",
        "labelloc": "c",
        "label": label,
    }


def edge_style(
    u: Optional[str] = None,
    v: Optional[str] = None,
    w: Optional[int] = 1,
    expert: bool = False,
) -> Dict[str, str]:
    return {
        "penwidth": int(w) + 1,  # Setting edge thickness
        "weight": int(w) + 4,  # Higher "weight"s mean shorter edges
        "arrowsize": 2,  # - 1.5 * w,  # Avoid too large arrows
        "arrowtail": "dot",
        "color": "black" if not expert else "red",
        "arrowhead": "normal",
        # "constraint": False,
    }
