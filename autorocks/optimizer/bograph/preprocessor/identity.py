from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class IdentityProcessor(DataPreprocessor):
    def fit(self, data: BoGraphDataPandas):
        # identity processor does not do anything
        return

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        return data
