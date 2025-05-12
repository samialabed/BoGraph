import autorocks.experiments.simulated_env.custom_models.ackley_bograph_models as bograph_models
from autorocks.envs.synthetic.funcs.ackley import Ackley6DParametersSpace, AkcleyCfg
from autorocks.execution.manager import runner_manager
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.experiments.simulated_env.cfgs import SIM_COMMON_OPTIMIZER
from autorocks.experiments.simulated_env.utils import sim_env_cli

available_optimizer = SIM_COMMON_OPTIMIZER
available_optimizer["bograph_high_static"] = bograph_models.static_high_level
available_optimizer["bograph_mid_static"] = bograph_models.static_mid_level
available_optimizer["bograph_low_static"] = bograph_models.static_low_level
available_optimizer["bograph_auto"] = bograph_models.auto

if __name__ == "__main__":
    cli_options = sim_env_cli(available_optimizer=list(available_optimizer.keys()))
    env_cfg = AkcleyCfg()
    param_schema = Ackley6DParametersSpace()
    opt_cfg = available_optimizer[cli_options.opt](param_schema)

    config = ExperimentConfigs(
        iterations=cli_options.iter,
        opt_cfg=opt_cfg,
        env_cfg=env_cfg,
        debug=cli_options.debug,
        exp_dir=cli_options.exp_dir,
    )
    runner_manager(config=config, execution_plan=cli_options.execution_plan)
