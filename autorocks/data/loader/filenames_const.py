# TODO: Better naming!
# Experiments are structured as:
# Env_Name/Objective_Name/Benchmark_name/Num_params/NumIterations/OptimizerName/ExpTime

# every iteration in an experiment: The model observe the previous steps.
# These are directory under /ExpTime/<1>....<NumIterations>
# Each representing a single full system evaluation.
ITERATION_ENV_EXECUTION_TIME = "env_execution_time.txt"
ITERATION_MODEL_PERFS = "training_time.txt"
ITERATION_SYS_PARAMS = "params.json"
ITERATION_SYS_OBSERVATIONS = "measurement.json"
# TODO: Above need to be renamed

# all iterations collected in an experiment. The model cannot observe other experiment
# This combines all directories that full under the directory <ExpTime> into one CSV
EXPERIMENT_MODEL_PERFS = "exp_trainings_time.csv"
EXPERIMENT_SYS_PARAMS = "exp_chosen_params.csv"
EXPERIMENT_SYS_OBSERVATIONS = "exp_sys_measurements.csv"

# Combine all the experiments conducted for a specific model.
MODEL_ALL_EXP_MODEL_PERFS = "model_all_exp_training_time.csv"
MODEL_ALL_EXP_SYS_OBSERVATIONS = "model_all_exp_sys_measurements.csv"
MODEL_ALL_EXP_SYS_PARAMS = "model_all_exp_params.csv"

# Combine all models into one single file.
ALL_MODELS_MODEL_PERFS = "all_models_training_time.csv"
ALL_MODELS_SYS_PARAMS = "all_models_params.csv"
ALL_MODELS_SYS_OBSERVATIONS = "all_models_sys_measurements.csv"

# checkpointing file
PARSED_EXP = "parsed_exp.csv"

# Reproducibility files
CONFIG_FILE = "config_file.json"
