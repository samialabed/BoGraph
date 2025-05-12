import autorocks.experiments.simulated_env.custom_models.branin_cur_models as custom_models
import autorocks.optimizer.botorch_opt.models as botorch_model
from autorocks.envs.synthetic.funcs.branin_currin import (
    BraninCur2DParametersSpace,
    BraninCurCfg,
)
from autorocks.envs.synthetic.synth_objective_dao import (
    BraninMOBOFuncObjective,
    CurrinMOBOFuncObjective,
)
from autorocks.execution.manager import runner_manager
from autorocks.exp_cfg import ExperimentConfigs
from autorocks.experiments import baseline_opt_cfg
from autorocks.experiments.simulated_env.utils import sim_env_cli

OBJECTIVE = [BraninMOBOFuncObjective(), CurrinMOBOFuncObjective()]

available_optimizer = {
    "BoTorchMOBO": lambda param_schema: baseline_opt_cfg.mobo_botorch(
        param_schema=param_schema,
        opt_obj=OBJECTIVE,
        surrogate_model=botorch_model.SingleTaskModel(),
    ),
    "Static": custom_models.static_bograph,
}

if __name__ == "__main__":
    cli_options = sim_env_cli(available_optimizer=list(available_optimizer.keys()))
    env_cfg = BraninCurCfg()
    param_schema = BraninCur2DParametersSpace()
    opt_cfg = available_optimizer[cli_options.opt](param_schema)

    config = ExperimentConfigs(
        iterations=cli_options.iter,
        opt_cfg=opt_cfg,
        env_cfg=env_cfg,
        debug=cli_options.debug,
        exp_dir=cli_options.exp_dir,
    )
    runner_manager(config=config, execution_plan=cli_options.execution_plan)
