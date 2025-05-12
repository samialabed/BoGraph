from typing import Set

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class ColRemoverProcessor(DataPreprocessor):
    def __init__(self, col_to_remove: Set[str]):
        """Normalize the parameters to [0, 1] range according to their bounds"""
        self.col_to_remove = col_to_remove

    def fit(self, data: BoGraphDataPandas):
        pass

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.intermediate = data.intermediate[
            data.intermediate.columns - self.col_to_remove
        ]
        return data
