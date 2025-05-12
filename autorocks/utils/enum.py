from enum import Enum
from typing import List, Set


class ExtendedEnum(Enum):
    @classmethod
    def list(cls) -> List[str]:
        return list(map(lambda c: c.value, cls))

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self):
        return str(self.value)
