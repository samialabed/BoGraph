from typing import Optional, Set

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class VarianceThresholdPreprocessor(DataPreprocessor):
    """Cull columns that variance does not change between experiments.
    Works only on intermediate column and not parameters nor objectives.
    """

    def __init__(self, known_nodes: Optional[Set[str]] = None, threshold: float = 0.1):
        self._cols_selector = VarianceThreshold(threshold=threshold)
        self._known_nodes = known_nodes or {}

    def fit(self, data: BoGraphDataPandas):
        self._cols_selector.fit(data.intermediate)

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        drop_col = set(
            data.intermediate.columns[np.invert(self._cols_selector.get_support())]
        )
        # make sure known nodes aren't removed
        for known_node in self._known_nodes:
            if known_node in drop_col:
                drop_col.remove(known_node)

        data.intermediate = data.intermediate.drop(columns=drop_col)
        return data
