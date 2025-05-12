import math


def short_size_to_base2(size_str: str) -> int:
    """ Go from 16kB -> 10 + 4  (kb = 2^10, 16 = 2 ^4)"""
    return int(math.log2(short_size_to_bytes(size_str)))


def short_size_to_bytes(size_str: str) -> int:
    """ Go from 16kB -> 16384. """
    size_str = size_str.lower()
    if size_str.endswith("kb"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("mb"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("gb"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    elif size_str.endswith("b"):
        return int(size_str[:-2])
    else:
        raise ValueError(f'Size "{size_str}" cannot be converted into bytes.')


def bytes_to_gigabytes(b):
  return b / (1024 ** 3)


def byte_to_short(size: int) -> str:
    """ Go from 16384 -> 16kB. """
    if size < 1024:
        return str(size)
    else:
        return "%dkB" % (size / 1024)


seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def convert_to_seconds(s: str):
    """ Go from 1h to 1 * 3600 """
    return int(s[:-1]) * seconds_per_unit[s[-1]]
