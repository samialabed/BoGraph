import enum
from typing import NamedTuple

import networkx as nx
import pandas as pd
from castle import algorithms

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas


class DAGType(enum.Enum):
    """DAG types to learn.
    Attributes:
        INTERMEDIATE_OBJ: DAG that connects intermediate metrics to objectives.
        PARAM_INTERMEDIATE: DAG that connects parameters to intermediates.
        FULL: DAG that uses all the available params, intermediate, objectives.
    """

    INTERMEDIATE_OBJ = "intermediate_objective_dag"
    PARAM_INTERMEDIATE = "parameter_intermediate_dag"
    FULL = "full_dag"
    RAW = "raw"


class LearnedDAG(NamedTuple):
    dag: nx.DiGraph
    score: float
    likelihood: float


def learn_dag(
    data: BoGraphDataPandas, dag_type: DAGType, save: bool = False
) -> LearnedDAG:
    model = algorithms.GOLEM(device_type="gpu")

    if dag_type.value == DAGType.INTERMEDIATE_OBJ.value:
        int_obj_df = data.intermediates_objs_df()
        data_values = int_obj_df.values
        columns = int_obj_df.columns
    elif dag_type.value == DAGType.PARAM_INTERMEDIATE.value:
        param_int_df = data.param_intermediates_df()
        data_values = param_int_df.values
        columns = param_int_df.columns
    elif dag_type.value == DAGType.FULL.value:
        full_df = data.to_combi_pandas()
        data_values = full_df.values
        columns = full_df.columns
    elif dag_type.value == DAGType.RAW.value:
        data_values = data
        columns = data.columns
    else:
        raise ValueError(f"Unknown DAGType: {dag_type}")

    model.learn(
        data_values,
        columns=columns,
    )
    dag = nx.from_pandas_adjacency(
        pd.DataFrame(
            model.causal_matrix,
            index=model.causal_matrix.columns,
            columns=model.causal_matrix.columns,
        ),
        create_using=nx.DiGraph,
    )
    if save:
        nx.write_gpickle(dag, f"{str(dag_type)}.gpickle")
    print("Done")
    return LearnedDAG(dag, likelihood=float(model.likelihood), score=float(model.loss))


def merge_dags(*dags: nx.DiGraph) -> nx.DiGraph:
    """Combines several DAGs into one."""
    combined_dag = nx.DiGraph()
    for dag in dags:
        combined_dag.update(dag)
    return combined_dag
