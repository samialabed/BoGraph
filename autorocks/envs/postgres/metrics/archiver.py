from dataclasses import dataclass

from autorocks.envs.postgres.metrics.abc_metric import Metric


@dataclass
class PGStatArchiver(Metric):
    """The pg_stat_archiver view will always have a single row, containing data about
    the archiver process of the cluster."""

    @classmethod
    def _view_name(cls) -> str:
        return "pg_stat_archiver"

    archived_count: int  # Number of WAL files that have been successfully archived
    failed_count: int  # Number of failed attempts for archiving WAL files
