from abc import ABC
from dataclasses import dataclass

from sysgym.params import ParamsSpace


@dataclass(init=False, frozen=True)
class PostgresParamsSpace(ParamsSpace, ABC):
    """Interface to allow versioning of Postgres params."""

    pass
