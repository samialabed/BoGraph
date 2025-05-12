from sklearn.preprocessing import StandardScaler

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class MetricsStandardizerProcessor(DataPreprocessor):
    def __init__(self, standardize_params: bool = False):
        self.obj_scaler = StandardScaler()
        self.intermediate_scaler = StandardScaler()
        self._param_scaler = StandardScaler() if standardize_params else None

    def fit(self, data: BoGraphDataPandas):
        obj_val = data.objs.values
        if obj_val.ndim == 1:
            # Single objective, ensure shape is [n, 1]
            obj_val = obj_val.reshape(-1, 1)
        self.obj_scaler.fit(obj_val)
        self.intermediate_scaler.fit(data.intermediate)
        if self._param_scaler is not None:
            self._param_scaler.fit(data.params)

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.intermediate.iloc[:] = self.intermediate_scaler.transform(
            data.intermediate
        )
        obj_val = data.objs.values
        if obj_val.ndim == 1:
            # Single objective, ensure shape is [n, 1]
            obj_val = obj_val.reshape(-1, 1)
        data.objs.iloc[:] = self.obj_scaler.transform(obj_val).squeeze()
        if self._param_scaler is not None:
            data.params.iloc[:] = self._param_scaler.transform(data.params)

        return data

    def inverse_transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        data.intermediate.iloc[:] = self.intermediate_scaler.inverse_transform(
            data.intermediate
        )
        obj_val = data.objs.values
        if obj_val.ndim == 1:
            # Single objective, ensure shape is [n, 1]
            obj_val = obj_val.reshape(-1, 1)
        data.objs.iloc[:] = self.obj_scaler.inverse_transform(obj_val).squeeze()
        if self._param_scaler is not None:
            data.params.iloc[:] = self._param_scaler.inverse_transform(data.params)

        return data
