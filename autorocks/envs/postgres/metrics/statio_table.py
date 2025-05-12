from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class StatIOTable(Metric):
    """
    The pg_statio_user_tables view will contain one row for each table in the current
    database (including TOAST tables), showing statistics about I/O on that specific
    table. The pg_statio_user_tables and pg_statio_sys_tables views contain the same
    information, but filtered to only show user and system tables respectively.
    """

    @classmethod
    def _select_fields(cls) -> str:
        """Sum all tables IO"""
        return ", ".join([f"SUM({f}) as {f}" for f in cls._fields()])

    @classmethod
    def _view_name(cls) -> str:
        return "pg_statio_user_tables"

    heap_blks_read: float  # Number of disk blocks read from this table
    heap_blks_hit: float  # Number of buffer hits in this table
    idx_blks_read: float  # Number of disk blocks read from all indexes on this table
    idx_blks_hit: float  # Number of buffer hits in all indexes on this table
    # Number of disk blocks read from this table's TOAST table (if any)
    toast_blks_read: Optional[int]
    # Number of buffer hits in this table's TOAST table (if any)
    toast_blks_hit: Optional[int]
    # Number of disk blocks read from this table's TOAST table indexes (if any)
    tidx_blks_read: Optional[int]
    # Number of buffer hits in this table's TOAST table indexes (if any)
    tidx_blks_hit: Optional[int]

    def __post_init__(self):
        if isinstance(self.heap_blks_read, Decimal):
            self.heap_blks_read = float(self.heap_blks_read)
        if isinstance(self.heap_blks_hit, Decimal):
            self.heap_blks_hit = float(self.heap_blks_hit)
        if isinstance(self.idx_blks_read, Decimal):
            self.idx_blks_read = float(self.idx_blks_read)
        if isinstance(self.idx_blks_hit, Decimal):
            self.idx_blks_hit = float(self.idx_blks_hit)
