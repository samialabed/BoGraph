from dataclasses import dataclass
from decimal import Decimal

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class StatIOIndex(Metric):
    """The pg_statio_user_indexes view will contain one row for each index in the
    current database, showing statistics about I/O on that specific index. The
    pg_statio_user_indexes and pg_statio_sys_indexes views contain the same
    information, but filtered to only show user and system indexes respectively."""

    @classmethod
    def _view_name(cls) -> str:
        return "pg_statio_user_indexes"

    @classmethod
    def _select_fields(cls) -> str:
        """Sum all tables IO"""
        return ", ".join([f"SUM({f}) as {f}" for f in cls._fields()])

    idx_blks_read: float  # Number of disk blocks read from this index
    idx_blks_hit: float  # Number of buffer hits in this index

    def __post_init__(self):
        # The type it retrieved from the database is actually Decimal
        # hence the need to cast it to float
        if isinstance(self.idx_blks_read, Decimal):
            self.idx_blks_read = float(self.idx_blks_read)
        if isinstance(self.idx_blks_hit, Decimal):
            self.idx_blks_hit = float(self.idx_blks_hit)
