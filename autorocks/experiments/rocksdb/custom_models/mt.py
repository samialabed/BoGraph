from pathlib import Path

import networkx as nx
from botorch.sampling import SobolQMCNormalSampler

from autorocks.envs.postgres.schema import PostgresParametersCollection10
from autorocks.experiments.postgres.objs.latency import LatencyP99
from autorocks.optimizer import acqf
from autorocks.optimizer.bograph.dag_options import BoGraphConfig, DAGPrior
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.preprocessor.normalizer import ParamNormalizerProcessor
from autorocks.optimizer.bograph.preprocessor.standardizer import (
    MetricsStandardizerProcessor,
)
from autorocks.optimizer.bograph.structure_learn.scheduler import StaticUpdateStrategy
from autorocks.optimizer.bograph.structure_learn.static_dag import StaticDAG
from autorocks.optimizer.opt_configs import OptimizerConfig


def bograph_static_postgres(
    param_space: PostgresParametersCollection10,
) -> OptimizerConfig:
    known_dag = nx.read_edgelist(
        Path(__file__).parent / "postgres.edgelist.gcastle", create_using=nx.DiGraph
    )
    dag_prior = DAGPrior(
        initial_dag=known_dag,
    )

    params_to_keep = set()

    for p in param_space.keys():
        if p in known_dag:
            params_to_keep.add(p)

    param_space = param_space.subset(params_to_keep)
    return BoGraphConfig(
        name="BoGraphStatic",
        param_space=param_space,
        opt_objectives=[LatencyP99()],
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=100),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
        acquisition_function=acqf.qNoisyExpectedImprovementWrapper(
            sampler=SobolQMCNormalSampler(num_samples=8026),
            optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_space.dimensions),
        ),
    )
