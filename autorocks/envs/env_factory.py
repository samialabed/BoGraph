from typing import Type

from sysgym.envs import Gem5EnvConfig, RocksDBEnvConfig

from sysgym import EnvConfig, Environment


def env_from_cfg(env_cfg: EnvConfig) -> Type[Environment]:
    # TODO: temporarily here until we migrate all environments to sysgym
    if isinstance(env_cfg, RocksDBEnvConfig):
        from sysgym.envs.rocksdb.env import RocksDBEnv

        env_cls = RocksDBEnv
    elif isinstance(env_cfg, Gem5EnvConfig):
        from sysgym.envs.gem5.env import Gem5

        env_cls = Gem5

    elif env_cfg.name == "postgres":
        from autorocks.envs.postgres.env import Postgres

        env_cls = Postgres
    else:
        from autorocks.envs.synthetic.funcs.syncth_factory import env_from_cfg

        env_cls = env_from_cfg(env_cfg)

    return env_cls
