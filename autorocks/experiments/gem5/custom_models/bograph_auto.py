import networkx as nx
from sysgym.params import ParamsSpace

from autorocks.experiments.gem5.custom_models.constants import AVG_POWER, EDP, SIM_SEC
from autorocks.experiments.gem5.custom_models.edp_formula import EDPFormula
from autorocks.experiments.gem5.objs import EDPObjective
from autorocks.optimizer.acqf import AcqfOptimizerCfg, UpperConfidentBoundWrapper
from autorocks.optimizer.bograph.dag_options import BoGraphConfig, DAGPrior
from autorocks.optimizer.bograph.dag_preprocessor import PreprocessingPipeline
from autorocks.optimizer.bograph.structure_learn.notears.notears import NoTears
from autorocks.optimizer.bograph.structure_learn.scheduler import StaticUpdateStrategy
from autorocks.optimizer.opt_configs import OptimizerConfig


def bograph_causal_edp(param_space: ParamsSpace) -> OptimizerConfig:
    initial_dag = nx.DiGraph()
    initial_dag.add_edges_from([(p, SIM_SEC) for p in param_space.keys()])
    initial_dag.add_edges_from([(p, AVG_POWER) for p in param_space.keys()])
    initial_dag.add_edges_from([(SIM_SEC, EDP), [AVG_POWER, EDP]])

    dag_prior = DAGPrior(
        known_edges=[(SIM_SEC, EDP), [AVG_POWER, EDP]],
        known_models={EDP: EDPFormula},
        initial_dag=initial_dag,
    )

    return BoGraphConfig(
        name="BoGraphNoTearsUCB",
        param_space=param_space,
        opt_objectives=[EDPObjective()],
        random_iter=10,
        update_strategy=StaticUpdateStrategy(update_freq_iter=20),
        structure_discovery_strategy=NoTears(
            parents_free_nodes=set(param_space.keys()),
            childfree_nodes={SIM_SEC, AVG_POWER, EDP},
        ),
        prior=dag_prior,
        retry=3,
        preprocessing_pipeline=PreprocessingPipeline(),
        acquisition_function=UpperConfidentBoundWrapper(
            beta=0.2, optimizer_cfg=AcqfOptimizerCfg(dim=param_space.dimensions)
        ),
    )
