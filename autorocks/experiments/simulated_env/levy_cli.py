from autorocks.envs.synthetic.funcs import levy
from autorocks.envs.synthetic.synth_objective_dao import TargetFuncObjective
from autorocks.execution.manager import runner_manager
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.experiments.simulated_env.cfgs import SIM_COMMON_OPTIMIZER
from autorocks.experiments.simulated_env.utils import sim_env_cli
from autorocks.optimizer.bograph import dag_options

_DIM = 6

available_optimizer = SIM_COMMON_OPTIMIZER
available_optimizer["bobn"] = lambda param_schema: dag_options.BoBnConfig(
    name="BoBn",
    param_space=param_schema,
    opt_objectives=[TargetFuncObjective()],
    random_iter=3,
    retry=3,
    dag=levy.make_struct(_DIM),
)

if __name__ == "__main__":
    cli_options = sim_env_cli(
        available_optimizer=list(available_optimizer.keys()) + ["bobn_paper"]
    )
    env_cfg = levy.LevyCfg(dim=_DIM, noise_std=0.01)
    param_schema = levy.make_levy_space(_DIM)
    opt_cfg = available_optimizer[cli_options.opt](param_schema)

    config = ExperimentConfigs(
        iterations=cli_options.iter,
        opt_cfg=opt_cfg,
        env_cfg=env_cfg,
        debug=cli_options.debug,
        exp_dir=cli_options.exp_dir,
    )
    runner_manager(config=config, execution_plan=cli_options.execution_plan)
