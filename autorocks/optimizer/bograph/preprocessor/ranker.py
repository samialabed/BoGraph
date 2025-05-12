from sklearn.feature_selection import SelectKBest, f_regression

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class RankerProcessor(DataPreprocessor):
    """Pick K top highest ranking stats."""

    def __init__(self, top_k: int):
        self.ranker = SelectKBest(f_regression, k=top_k)

    def fit(self, data: BoGraphDataPandas):
        y = data.objs.values
        x = data.intermediate
        self.ranker.fit(x, y)

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.intermediate = data.intermediate.loc[:, self.ranker.get_support()]
        return data
