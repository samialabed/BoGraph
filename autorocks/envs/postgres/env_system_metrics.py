from dataclasses import asdict, dataclass
from typing import Dict, Type

from autorocks.envs.postgres.metrics.abc_metric import Metric
from autorocks.envs.postgres.metrics.archiver import PGStatArchiver
from autorocks.envs.postgres.metrics.bgwriter import PGStatBGWriter
from autorocks.envs.postgres.metrics.database import PGStatDatabase
from autorocks.envs.postgres.metrics.dbconf import DatabaseConflicts
from autorocks.envs.postgres.metrics.statio_index import StatIOIndex
from autorocks.envs.postgres.metrics.statio_table import StatIOTable
from autorocks.envs.postgres.metrics.wal import PGStatWAL


@dataclass
class PostgresSystemMetrics:
    archiver: PGStatArchiver
    wal: PGStatWAL
    bg_writer: PGStatBGWriter
    stat_database: PGStatDatabase
    db_conflicts: DatabaseConflicts
    io_index: StatIOIndex
    io_table: StatIOTable

    @classmethod
    def metric_group_and_type(cls) -> Dict[str, Type[Metric]]:
        return {
            "archiver": PGStatArchiver,
            "wal": PGStatWAL,
            "bg_writer": PGStatBGWriter,
            "stat_database": PGStatDatabase,
            "db_conflicts": DatabaseConflicts,
            "io_index": StatIOIndex,
            "io_table": StatIOTable,
        }

    @classmethod
    def failed_metrics_retrieval(cls):
        return PostgresSystemMetrics(
            archiver=PGStatArchiver.failed_metric(),
            wal=PGStatWAL.failed_metric(),
            bg_writer=PGStatBGWriter.failed_metric(),
            stat_database=PGStatDatabase.failed_metric(),
            db_conflicts=DatabaseConflicts.failed_metric(),
            io_index=StatIOIndex.failed_metric(),
            io_table=StatIOTable.failed_metric(),
        )

    def as_dict(self):
        archiver_dict = asdict(self.archiver)
        wal_dict = asdict(self.wal)
        bg_writer_dict = asdict(self.bg_writer)
        stat_database_dict = asdict(self.stat_database)
        db_conflicts_dict = asdict(self.db_conflicts)
        io_index_dict = asdict(self.io_index)
        io_table_dict = asdict(self.io_table)
        return {
            **archiver_dict,
            **wal_dict,
            **bg_writer_dict,
            **stat_database_dict,
            **db_conflicts_dict,
            **io_index_dict,
            **io_table_dict,
        }
