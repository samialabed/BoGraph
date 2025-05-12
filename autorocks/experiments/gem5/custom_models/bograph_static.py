import networkx as nx
from botorch.sampling import SobolQMCNormalSampler
from sysgym.params import ParamsSpace

from autorocks.experiments.gem5.custom_models.constants import AVG_POWER, EDP, SIM_SEC
from autorocks.experiments.gem5.custom_models.edp_formula import EDPFormula
from autorocks.experiments.gem5.objs import (
    EDPObjective,
    LatencyObjective,
    PowerObjective,
)
from autorocks.optimizer import acqf
from autorocks.optimizer.bograph.dag_options import BoGraphConfig, DAGPrior
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.preprocessor.normalizer import ParamNormalizerProcessor
from autorocks.optimizer.bograph.preprocessor.standardizer import (
    MetricsStandardizerProcessor,
)
from autorocks.optimizer.bograph.preprocessor.variance_threshold import (
    VarianceThresholdPreprocessor,
)
from autorocks.optimizer.bograph.structure_learn.scheduler import StaticUpdateStrategy
from autorocks.optimizer.bograph.structure_learn.static_dag import StaticDAG
from autorocks.optimizer.opt_configs import OptimizerConfig


def bograph_static_edp(param_space: ParamsSpace) -> OptimizerConfig:
    known_dag = nx.DiGraph()
    known_dag.add_edges_from([(p, SIM_SEC) for p in param_space.keys()])
    known_dag.add_edges_from([(p, AVG_POWER) for p in param_space.keys()])
    known_dag.add_edges_from([(SIM_SEC, EDP), [AVG_POWER, EDP]])

    dag_prior = DAGPrior(
        known_models={EDP: EDPFormula},
        initial_dag=known_dag,
        known_edges=[(SIM_SEC, EDP), [AVG_POWER, EDP]],
    )

    return BoGraphConfig(
        name="BoGraphStaticUCB",
        param_space=param_space,
        opt_objectives=[EDPObjective()],
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=20),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                VarianceThresholdPreprocessor(known_nodes={SIM_SEC, AVG_POWER, EDP}),
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
        acquisition_function=acqf.UpperConfidentBoundWrapper(
            beta=0.2, optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_space.dimensions)
        ),
    )


def mobograph_static_pwr_lat(param_space: ParamsSpace) -> OptimizerConfig:
    known_dag = nx.DiGraph()
    known_dag.add_edges_from([("cycle_time", SIM_SEC), ("pipelining", SIM_SEC)])
    known_dag.add_edges_from(
        [
            ("cache_assoc", AVG_POWER),
            ("cache_size", AVG_POWER),
            ("cache_line_sz", AVG_POWER),
            ("pipelined_dma", AVG_POWER),
            ("cycle_time", AVG_POWER),
        ]
    )

    dag_prior = DAGPrior(
        initial_dag=known_dag,
    )

    param_space = param_space.subset(
        {
            "cycle_time",
            "pipelining",
            "cache_assoc",
            "cache_size",
            "pipelined_dma",
            "cache_line_sz",
        }
    )
    return BoGraphConfig(
        name="MOBoGraphStatic",
        param_space=param_space,
        opt_objectives=[
            LatencyObjective(),
            PowerObjective(),
        ],
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
        acquisition_function=acqf.qNoisyExpectedHypervolumeImprovementWrapper(
            sampler=SobolQMCNormalSampler(num_samples=8026),
            ref_point=[0.5, 0.5],
            optimizer_cfg=acqf.AcqfOptimizerCfg(dim=param_space.dimensions),
        ),
    )
