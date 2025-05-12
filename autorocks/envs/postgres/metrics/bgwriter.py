from dataclasses import dataclass

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class PGStatBGWriter(Metric):
    """pg_stat_bgwriter: statistics about the background writer process's activity."""

    @classmethod
    def _view_name(cls) -> str:
        return "pg_stat_bgwriter"

    # Number of scheduled checkpoints that have been performed
    checkpoints_timed: int
    # Number of requested checkpoints that have been performed
    checkpoints_req: int
    # Total amount of time that has been spent in the portion of checkpoint
    # processing where files are written to disk, in milliseconds
    checkpoint_write_time: float
    # Total amount of time that has been spent in the portion of checkpoint
    # processing where files are synchronized to disk, in milliseconds
    checkpoint_sync_time: float
    # Number of buffers written during checkpoints
    buffers_checkpoint: int
    # Number of buffers written by the background writer
    buffers_clean: int
    # Number of times the background writer stopped a cleaning scan because it had
    # written too many buffers
    maxwritten_clean: int
    # Number of buffers written directly by a backend
    buffers_backend: int
    # Number of times a backend had to execute its own fsync call (normally the
    # background writer handles those even when the backend does its own write)
    buffers_backend_fsync: int
    # Number of buffers allocated
    buffers_alloc: int
