import enum
import time
from typing import List, NamedTuple, Set

from autorocks.optimizer.bograph import dag_discovery, dag_postprocessor, preprocessor
from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor
from notebooks.bobn_ch import utils


class Strategy(enum.Enum):
    NO_COMPRESS = "NoCompress"
    RANKER5 = "Ranker5"
    RANKER10 = "Ranker10"
    RANKER15 = "Ranker15"
    PCA = "PCA"

    def __eq__(self, other: "Strategy") -> bool:
        return self.value == other.value


class ExperimentResult(NamedTuple):
    runtime: float
    max_dim: int
    likelihood: float
    score: float


def get_preprocessors(strategy: Strategy) -> List[DataPreprocessor]:
    common_preprocessors = [
        # Add average for count
        preprocessor.GrouperProcessor(-2, preprocessor.Compressor.COMBINER),
        # If there are any with useful statistics get it.
        preprocessor.FilterProcessor(-2),
        preprocessor.VarianceThresholdPreprocessor(),
        preprocessor.MetricsStandardizerProcessor(standardize_params=True),
        # preprocessor.ParamNormalizerProcessor(param_space.bounds().T),
    ]
    if strategy == Strategy.NO_COMPRESS:
        return common_preprocessors
    if strategy == Strategy.RANKER5:
        common_preprocessors.append(preprocessor.RankerProcessor(top_k=5))
        return common_preprocessors
    if strategy == Strategy.RANKER10:
        common_preprocessors.append(preprocessor.RankerProcessor(top_k=10))
        return common_preprocessors
    if strategy == Strategy.RANKER15:
        common_preprocessors.append(preprocessor.RankerProcessor(top_k=15))
        return common_preprocessors
    if strategy == Strategy.PCA:
        common_preprocessors.append(
            preprocessor.GrouperProcessor(-2, preprocessor.Compressor.PCA)
        )
        return common_preprocessors
    raise ValueError("Unknown strategy: ", strategy)


def perform_dag_compression_exp(
    strategy: Strategy, data: BoGraphDataPandas, params: Set[str], objectives: Set[str]
) -> ExperimentResult:
    dp = PreprocessingPipeline(preprocessors=get_preprocessors(strategy))

    start_time = time.time()
    processed_data = dp.fit_transform(data)
    learned_g_info = dag_discovery.learn_dag(
        processed_data, dag_type=dag_discovery.DAGType.FULL
    )
    full_dag_pro = dag_postprocessor.postprocess_structure(
        learned_g_info.dag, params, objectives
    )
    end_time = time.time() - start_time

    max_dim = 0
    for subgraph in utils.create_d_separable_subgraphs(
        full_dag_pro, parameter_nodes=params, objectives=objectives
    ):
        subgraph_max_dim = max(
            full_dag_pro.subgraph(subgraph).in_degree, key=lambda x: x[1]
        )
        max_dim = max(subgraph_max_dim[1], max_dim)

    return ExperimentResult(
        runtime=end_time,
        max_dim=max_dim,
        likelihood=learned_g_info.likelihood,
        score=learned_g_info.score,
    )
