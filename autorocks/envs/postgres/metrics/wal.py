from dataclasses import dataclass
from decimal import Decimal

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class PGStatWAL(Metric):
    """The pg_stat_wal view will always have a single row, containing data about WAL
    activity of the cluster."""

    @classmethod
    def _view_name(cls) -> str:
        return "pg_stat_wal"

    # Total number of WAL records generated
    wal_records: int
    # Total number of WAL full page images generated
    wal_fpi: int
    # Total amount of WAL generated in bytes
    wal_bytes: float
    # Number of times WAL data was written to disk because WAL buffers became full
    wal_buffers_full: int
    # Number of times WAL buffers were written out to disk via XLogWrite request. See
    # Section 30.5 for more information about the internal WAL function XLogWrite.
    wal_write: int
    # Number of times WAL files were synced to disk via issue_xlog_fsync request (if
    # fsync is on and wal_sync_method is either fdatasync, fsync or
    # fsync_writethrough, otherwise zero). See Section 30.5 for more information
    # about the internal WAL function issue_xlog_fsync.
    wal_sync: int
    # Total amount of time spent writing WAL buffers to disk via XLogWrite request,
    # in milliseconds (if track_wal_io_timing is enabled, otherwise zero). This
    # includes the sync time when wal_sync_method is either open_datasync or open_sync.
    wal_write_time: float
    # Total amount of time spent syncing WAL files to disk via issue_xlog_fsync
    # request, in milliseconds (if track_wal_io_timing is enabled, fsync is on,
    # and wal_sync_method is either fdatasync, fsync or fsync_writethrough, otherwise
    # zero).
    wal_sync_time: float

    def __post_init__(self):
        if isinstance(self.wal_bytes, Decimal):
            self.wal_bytes = float(self.wal_bytes)
