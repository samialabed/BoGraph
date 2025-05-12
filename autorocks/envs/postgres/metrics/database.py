from dataclasses import dataclass
from typing import Optional

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class PGStatDatabase(Metric):
    """pg_stat_database: showing database-wide statistics.

    pg_stat_user_tables shows per table.
     For our purposes the higher grunality works fine.
    """

    @classmethod
    def filter_query(cls, database_name: str) -> str:
        return f"WHERE datname = '{database_name}'"

    @classmethod
    def _view_name(cls) -> str:
        return "pg_stat_database"

    # numbackends: Number of backends currently connected to this database, or NULL
    # for shared objects. This is the only column in this view that returns a value
    # reflecting current state; all other columns return the accumulated values since
    # the last reset.
    numbackends: Optional[int]
    # Number of transactions in this database that have been committed
    xact_commit: int
    # Number of transactions in this database that have been rolled back
    xact_rollback: int
    # Number of disk blocks read in this database
    blks_read: int
    # Number of times disk blocks were found already in the buffer cache, so that a
    # read was not necessary (this only includes hits in the PostgreSQL buffer cache,
    # not the operating system's file system cache)
    blks_hit: int
    # Number of rows returned by queries in this database
    tup_returned: int
    # Number of rows fetched by queries in this database
    tup_fetched: int
    # Number of rows inserted by queries in this database
    tup_inserted: int
    # Number of rows updated by queries in this database
    tup_updated: int
    # Number of rows deleted by queries in this database
    tup_deleted: int
    # Number of queries canceled due to conflicts with recovery in this database. (
    # Conflicts occur only on standby servers; see pg_stat_database_conflicts for
    # details.)
    conflicts: int
    # Number of temporary files created by queries in this database. All temporary
    # files are counted, regardless of why the temporary file was created (e.g.,
    # sorting or hashing), and regardless of the log_temp_files setting.
    temp_files: int
    # Total amount of data written to temporary files by queries in this database.
    # All temporary files are counted, regardless of why the temporary file was
    # created, and regardless of the log_temp_files setting.
    temp_bytes: int
    # Number of deadlocks detected in this database
    deadlocks: int
    # Time spent reading data file blocks by backends in this database, in milliseconds
    blk_read_time: float
    # Time spent writing data file blocks by backends in this database, in milliseconds
    blk_write_time: float
    # new metrics (post 9.4) Number of data page checksum failures detected in this
    # database (or on a shared object), or NULL if data checksums are not enabled.
    checksum_failures: Optional[int]
    # Time spent by database sessions in this database, in milliseconds (note that
    # statistics are only updated when the state of a session changes, so if sessions
    # have been idle for a long time, this idle time won't be included)
    session_time: float
    # Time spent executing SQL statements in this database, in milliseconds (this
    # corresponds to the states active and fastpath function call in pg_stat_activity)
    active_time: float
    # Time spent idling while in a transaction in this database, in milliseconds (
    # this corresponds to the states idle in transaction and idle in transaction (
    # aborted) in pg_stat_activity)
    idle_in_transaction_time: float
    # Total number of sessions established to this database
    sessions: int
    # Number of database sessions to this database that were terminated because
    # connection to the client was lost
    sessions_abandoned: int
    # Number of database sessions to this database that were terminated by fatal errors
    sessions_fatal: int
    # Number of database sessions to this database that were terminated by operator
    # intervention
    sessions_killed: int
