from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import List


@dataclass
class Metric(ABC):
    @classmethod
    @abstractmethod
    def _view_name(cls) -> str:
        """Name of the view that holds the metric"""

    @classmethod
    def failed_metric(cls) -> "Metric":
        failed_param_set = {}
        for f in fields(cls):
            failed_param_set[f.name] = -1
        return cls(**failed_param_set)

    @classmethod
    def sql_query(cls, database_name: str) -> str:
        return " ".join(
            [
                cls.select_query(),
                cls.from_query(),
                cls.filter_query(database_name=database_name),
            ]
        )

    @classmethod
    def select_query(cls) -> str:
        """Helper method that returns a selection query"""
        return f"SELECT {cls._select_fields()}"

    @classmethod
    def filter_query(cls, database_name: str) -> str:
        """Helper method that returns a filtering query"""
        return ""

    @classmethod
    def from_query(cls) -> str:
        """Helper method that builds a table query"""
        return f"FROM {cls._view_name()}"

    @classmethod
    def _select_fields(cls) -> str:
        """Return the _fields to select over"""
        return ", ".join(cls._fields())

    @classmethod
    def _fields(cls) -> List[str]:
        """Returns a list of _fields the metric cares about"""
        return [f.name for f in fields(cls)]
