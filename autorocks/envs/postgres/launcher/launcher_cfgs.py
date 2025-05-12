from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PostgresEnvVar(object):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class PostgresLauncherSettings(object):
    env_var: PostgresEnvVar
    ip_addr: str
    port: int
