from typing import Callable, List, NamedTuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetColumns(NamedTuple):
    params: List[str]
    system_perfs: List[str]


class BOSystemDataset(Dataset):
    def __init__(
        self,
        historic_data_df: pd.DataFrame,
        parameters_name: List[str],
        objectives_name: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.model_input = historic_data_df[parameters_name]
        self.model_target = historic_data_df[objectives_name]

        self.transform = transform
        self.target_transform = target_transform

        self.params = parameters_name
        self.objs = objectives_name

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, idx):
        """Return x, y: Parameters against the system perf."""
        system_param = torch.tensor(self.model_input.iloc[idx].values)
        system_perf = torch.tensor(self.model_target.iloc[idx].values)

        if self.transform:
            system_param = self.transform(system_param)
        if self.target_transform:
            system_perf = self.target_transform(system_perf)

        return system_param, system_perf

    def columns(self) -> DatasetColumns:
        return DatasetColumns(
            params=list(self.model_input.columns),
            system_perfs=list(self.model_target.columns),
        )
