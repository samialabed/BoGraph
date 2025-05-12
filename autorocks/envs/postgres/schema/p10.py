from dataclasses import dataclass

from sysgym.params.boxes import ContinuousBox, DiscreteBox

from autorocks.envs.postgres.schema.schema import PostgresParamsSpace
from autorocks.envs.postgres.schema.utils import maximum_upper_bound


@dataclass(init=False, frozen=True)
class PostgresParametersCollection10(PostgresParamsSpace):
    """Using the same parameters OtterTune tuned for a direct comparison"""

    shared_buffers: DiscreteBox = DiscreteBox(
        lower_bound=16,
        upper_bound=maximum_upper_bound(
            param_max_valid_setting=1073741823, param_unit="8kB"
        ),
        default=16384,
    )

    max_wal_size: DiscreteBox = DiscreteBox(
        lower_bound=2,
        upper_bound=maximum_upper_bound(
            param_max_valid_setting=2147483647, param_unit="1MB"
        ),
        default=1024,
    )

    effective_cache_size: DiscreteBox = DiscreteBox(
        lower_bound=1,
        upper_bound=maximum_upper_bound(
            param_max_valid_setting=2147483647, param_unit="8kB"
        ),
        default=524288,
    )

    bgwriter_lru_maxpages: DiscreteBox = DiscreteBox(
        lower_bound=0,
        upper_bound=1073741823,
        default=100,
    )

    bgwriter_delay: DiscreteBox = DiscreteBox(
        lower_bound=10, upper_bound=10000, default=200
    )

    checkpoint_completion_target: ContinuousBox = ContinuousBox(
        lower_bound=0, upper_bound=1, default=0.9
    )

    deadlock_timeout: DiscreteBox = DiscreteBox(
        lower_bound=1, upper_bound=2147483647, default=1000
    )

    default_statistics_target: DiscreteBox = DiscreteBox(
        lower_bound=1, upper_bound=1000, default=100
    )

    effective_io_concurrency: DiscreteBox = DiscreteBox(
        lower_bound=0, upper_bound=1000, default=0
    )

    checkpoint_timeout: DiscreteBox = DiscreteBox(
        lower_bound=30, upper_bound=86400, default=300
    )
