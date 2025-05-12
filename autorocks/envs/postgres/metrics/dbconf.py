from dataclasses import dataclass

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class DatabaseConflicts(Metric):
    """The pg_stat_database_conflicts view will contain one row per database,
    showing database-wide statistics about query cancels occurring due to conflicts
    with recovery on standby servers. This view will only contain information
    on standby servers, since conflicts do not occur on primary servers."""

    @classmethod
    def filter_query(cls, database_name: str) -> str:
        return f"WHERE datname = '{database_name}'"

    @classmethod
    def _view_name(cls) -> str:
        return "pg_stat_database_conflicts"

    # Number of queries in this database that have been canceled due to dropped tablespaces
    confl_tablespace: int
    confl_lock: int  # Number of queries in this database that have been canceled due to lock timeouts
    confl_snapshot: int  # Number of queries in this database that have been canceled due to old snapshots
    # Number of queries in this database that have been canceled due to pinned buffers
    confl_bufferpin: int
    confl_deadlock: int  # Number of queries in this database that have been canceled due to deadlocks
