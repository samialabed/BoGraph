import networkx as nx
from botorch.sampling import SobolQMCNormalSampler

from autorocks.envs.synthetic.funcs.branin_currin import BraninCur2DParametersSpace
from autorocks.envs.synthetic.synth_objective_dao import (
    BraninMOBOFuncObjective,
    CurrinMOBOFuncObjective,
)
from autorocks.optimizer import acqf
from autorocks.optimizer.acqf import AcqfOptimizerCfg
from autorocks.optimizer.bograph.dag_options import BoGraphConfig, DAGPrior
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.preprocessor.normalizer import ParamNormalizerProcessor
from autorocks.optimizer.bograph.preprocessor.standardizer import (
    MetricsStandardizerProcessor,
)
from autorocks.optimizer.bograph.structure_learn.scheduler import StaticUpdateStrategy
from autorocks.optimizer.bograph.structure_learn.static_dag import StaticDAG
from autorocks.optimizer.opt_configs import OptimizerConfig

__OBJECTIVE = [BraninMOBOFuncObjective(), CurrinMOBOFuncObjective()]


def static_bograph(param_space: BraninCur2DParametersSpace) -> OptimizerConfig:
    known_dag = nx.DiGraph()
    # Branin structure
    known_dag.add_edges_from([("x1", "t2"), ("x1", "t1_pow2"), ("x2", "t1_pow2")])
    known_dag.add_edges_from([("t1_pow2", "branin"), ("t2", "branin")])
    # Currin structure
    known_dag.add_edges_from([("x2", "currin_factor"), ("x1", "currin_denom")])
    known_dag.add_edges_from([("currin_denom", "currin"), ("currin_factor", "currin")])
    dag_prior = DAGPrior(
        initial_dag=known_dag,
    )

    return BoGraphConfig(
        name="StaticBoGraphqEI",
        param_space=param_space,
        opt_objectives=__OBJECTIVE,
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=90),
        structure_discovery_strategy=StaticDAG(known_dag=known_dag),
        # acquisition_function=ExpectedHypervolumeImprovementWrapper(
        #     ref_point=[0.5, 0.2]
        # ),
        acquisition_function=acqf.qExpectedHypervolumeImprovementWrapper(
            sampler=SobolQMCNormalSampler(num_samples=8026),
            ref_point=[0.3, 0.3],
            optimizer_cfg=AcqfOptimizerCfg(dim=param_space.dimensions),
        ),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(
            [
                ParamNormalizerProcessor(param_space.bounds(True).T),
                MetricsStandardizerProcessor(),
            ]
        ),
    )
