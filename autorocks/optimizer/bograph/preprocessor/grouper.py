import re
from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class Compressor(Enum):
    """
    Attributes:
        COMBINER: Combines multiple metric in one, used for combining sum/count to avg.

    """

    PCA = "PrincipleComponentAnalysis"
    FA = "FactorAnalysis"
    COMBINER = "Combiner"


class CombinerMatcher:
    """Look for strings to match against in order"""

    def fit_transform(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        s = None
        c = None
        for col in data.columns:
            # Do not remove statistics
            if col.endswith("p95") or col.endswith("p50"):
                return None
            if s is not None and c is not None:
                break
            if "sum" in col:
                s = data[col]
            elif "count" in col:
                c = data[col]
        if s is not None and c is not None:
            return (s / c).fillna(0).values

        return None


class GrouperProcessor(DataPreprocessor):
    """Creates groups by indexing into the logs."""

    def __init__(
        self,
        index_into_group: int,
        compressor: Compressor = Compressor.PCA,
        metric_regex: str = "([^.]+)",
    ):
        """

        Can be called multiple time to keep compressing the groups further and further.
        Args:
            index_into_group: The index into the parsed metric that forms the group.
            compressor: the algorithm used to decompose along the indexes.
            metric_regex: The regex to parse the columns into groups.
        """
        if compressor.value == Compressor.PCA.value:
            from sklearn.decomposition import PCA

            self._compressor = PCA(n_components=1)

        elif compressor.value == Compressor.FA.value:
            from sklearn.decomposition import FactorAnalysis

            self._compressor = FactorAnalysis(n_components=1)

        elif compressor.value == Compressor.COMBINER.value:
            self._compressor = CombinerMatcher()
        else:
            raise Exception(f"Unknown compressor: {compressor}")

        self._groups_extractor = re.compile(metric_regex, re.RegexFlag.IGNORECASE)
        self._group_indexer = index_into_group

        self._metric_groups = defaultdict(list)

    def fit(self, data: BoGraphDataPandas):
        """Captures the groups."""
        for full_metric_name in data.intermediate.columns:
            # Find the main groups for the data
            metric_name_parts = self._groups_extractor.findall(full_metric_name)
            if metric_name_parts and len(metric_name_parts) > 1:
                # Discovered the groups, choose which group to index into
                metric_name = metric_name_parts[self._group_indexer]
                # Map the "group": {all metrics that have that belong to that group.}
                self._metric_groups[metric_name].append(full_metric_name)

            else:
                self._metric_groups[full_metric_name].append(full_metric_name)

    def transform(self, data: BoGraphDataPandas) -> BoGraphDataPandas:
        """Compress metrics results"""
        new_groups = {}

        metric_data = data.intermediate

        for group, related_groups in self._metric_groups.items():
            # Get all the related groups
            group_vals_ = metric_data[related_groups]
            group_pruned_val = self._compressor.fit_transform(group_vals_)
            if group_pruned_val is None:
                # Nothing to compress, encode back the related groups
                for related_group in related_groups:
                    new_groups[related_group] = metric_data[related_group]
            elif isinstance(self._compressor, CombinerMatcher):
                new_groups[f"{group}.avg"] = group_pruned_val.squeeze()
            else:
                new_groups[group] = group_pruned_val.squeeze()
        # Replace the intermediates
        data.intermediate = pd.DataFrame(new_groups)
        return data
