import os

from autorocks.utils import converters as conv

os_getenv = os.getenv("SYSTEM_MEMORY")
assert os_getenv, "Please set the environment variable SYSTEM_MEMORY"
SYSTEM_MEMORY_IN_BYTE = conv.short_size_to_bytes(os_getenv)


def maximum_upper_bound(param_max_valid_setting: int, param_unit: str) -> int:
    """Return the minimum of either of the available system resources or parameter's
    maximum setting"""
    resource_unit = conv.short_size_to_bytes(param_unit)

    return min(param_max_valid_setting, int(SYSTEM_MEMORY_IN_BYTE // resource_unit))
