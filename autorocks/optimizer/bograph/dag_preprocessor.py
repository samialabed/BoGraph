from typing import List, Optional

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class PreprocessingPipeline(object):
    """Mutating pipeline."""

    def __init__(self, preprocessors: Optional[List[DataPreprocessor]] = None):
        self.preprocessors = preprocessors

    def fit(self, data: BoGraphDataPandas):
        if self.preprocessors is None:
            return
        for preprocessor in self.preprocessors:
            preprocessor.fit(data)
            data = preprocessor.transform(data)

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        if self.preprocessors is None:
            return data
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)
        return data

    def fit_transform(
        self, data: BoGraphDataPandas, inplace: bool = False
    ) -> BoGraphDataPandas:
        if self.preprocessors is None:
            return data
        if not inplace:
            data = data.copy()
        for preprocessor in self.preprocessors:
            data = preprocessor.fit_transform(data)
        return data

    def inverse_transform(self, data: BoGraphDataPandas, inplace: bool = False):
        if self.preprocessors is None:
            return data
        if not inplace:
            data = data.copy()
        for preprocessor in self.preprocessors:
            data = preprocessor.inverse_transform(data)
        return data
