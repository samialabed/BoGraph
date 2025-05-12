from abc import ABC, abstractmethod

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas


class DataPreprocessor(ABC):
    @abstractmethod
    def fit(self, data: BoGraphDataPandas):
        pass

    @abstractmethod
    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        pass

    def inverse_transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        return data

    def fit_transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        self.fit(data=data)
        return self.transform(data=data)
