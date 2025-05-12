from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class UpdateStrategy(ABC):
    @abstractmethod
    def eval(self) -> bool:
        pass


class EpsilonUpdateStrategy(UpdateStrategy):
    def __init__(self, init_update_freq: int, eps: float, minimum: int = 1):
        """
            Updater that
            starting at init_update_freq with a budget and reduce it by eps
        Args:
            init_update_freq:
            eps: the magnitude to lower the update freq by. should be (0, 1]
            minimum: the lowest possible frequency (lower more often)
        """
        assert (
            0 < eps <= 1
        ), f"Frequency should be reduced by the bounds of (0, 1], got {eps}"

        self._eps = eps
        self._update_freq = init_update_freq
        self._min = minimum
        self._obs = 0

    def eval(self) -> bool:
        self._obs += 1
        if self._obs % self._update_freq == 0:
            self._update_freq = max(self._update_freq * self._eps, self._min)
            return True


class StaticUpdateStrategy(UpdateStrategy):
    def __init__(self, update_freq_iter: int = 10):
        # every X iterations perform score-search-mutate rotation
        self.update_freq_iter = update_freq_iter
        self._obs = 0

    def eval(self) -> bool:
        self._obs += 1
        return self._obs % self.update_freq_iter == 0
