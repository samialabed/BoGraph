from dataclasses import dataclass

from sysgym.params.boxes import UnboundedBox

from autorocks.envs.postgres.schema.schema import PostgresParamsSpace
from autorocks.utils import converters as conv

OTTERTUNE_BEST_RESULT = {
    "shared_buffers": conv.short_size_to_bytes("4gb"),
    "max_wal_size": 540,  # this used to be called checkpoint_segment
    "effective_cache_size": conv.short_size_to_bytes("18gb"),
    "bgwriter_lru_maxpages": 1000,
    "bgwriter_delay": 213,
    "checkpoint_completion_target": 0.8,
    "deadlock_timeout": 6,
    "default_statistics_target": 78,
    "effective_io_concurrency": 3,
    "checkpoint_timeout": conv.convert_to_seconds("1h"),
}


@dataclass(frozen=True, init=False)
class OtterTuneParamSchema(PostgresParamsSpace):
    """Using the same parameters OtterTune tuned for a direct comparison"""

    shared_buffers: UnboundedBox = UnboundedBox(default=4294967296)
    max_wal_size: UnboundedBox = UnboundedBox(default=540)
    effective_cache_size: UnboundedBox = UnboundedBox(default=19327352832)
    bgwriter_lru_maxpages: UnboundedBox = UnboundedBox(default=1000)
    bgwriter_delay: UnboundedBox = UnboundedBox(default=213)
    checkpoint_completion_target: UnboundedBox = UnboundedBox(default=0.8)
    deadlock_timeout: UnboundedBox = UnboundedBox(default=6)
    default_statistics_target: UnboundedBox = UnboundedBox(default=78)
    effective_io_concurrency: UnboundedBox = UnboundedBox(default=3)
    checkpoint_timeout: UnboundedBox = UnboundedBox(default=3600)
