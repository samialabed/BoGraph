import numpy as np

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class ParamNormalizerProcessor(DataPreprocessor):
    def __init__(self, bounds: np.ndarray):
        """Normalize the parameters to [0, 1] range according to their bounds"""
        self.bounds = bounds

    def fit(self, data: BoGraphDataPandas):
        # TODO: Take paramspace and actually map
        #  between col in data to param_space_bounds
        pass

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.params.iloc[:] = (data.params - self.bounds[0]) / (
            self.bounds[1] - self.bounds[0]
        )
        return data

    def inverse_transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.params.iloc[:] = (
            data.params * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        )
        return data
