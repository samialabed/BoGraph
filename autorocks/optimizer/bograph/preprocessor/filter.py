import re
from collections import defaultdict
from enum import Enum
from typing import List, Optional

import pandas as pd

from autorocks.optimizer.bograph.bograph_dao import BoGraphDataPandas
from autorocks.optimizer.bograph.preprocessor.preprocessor_abc import DataPreprocessor


class Matcher(Enum):
    """
    Attributes:
        ORDERED: The matching happens in order of appearance.
        REVERSE: Accepts anythign but what is specified.
    """

    ORDERED = "Ordered"
    REVERSE = "Reversed"


class OrderedStringMatcher:
    """Look for strings to match against in order"""

    def __init__(self, strings_to_match: List[str]):
        self._strings_to_match = strings_to_match

    def fit_transform(self, data: pd.DataFrame) -> Optional[str]:
        for matcher in self._strings_to_match:
            for col in data.columns:
                if col.endswith(matcher):
                    return col
        return None


class FilterProcessor(DataPreprocessor):
    """Filters and select specific columns only."""

    def __init__(
        self,
        index_into_group: int,
        matcher: Matcher = Matcher.ORDERED,
        metric_regex: str = "([^.]+)",
    ):
        """

        Can be called multiple time to keep compressing the groups further and further.
        Args:
            index_into_group: The index into the parsed metric that forms the group.
            matcher: the algorithm used to decompose along the indexes.
            metric_regex: The regex to parse the columns into groups.
        """
        if matcher.value == matcher.ORDERED.value:
            self._matcher = OrderedStringMatcher(["p95", "p50", "avg", "sum"])
        else:
            raise Exception(f"Unknown matcher: {matcher}")

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
            group_pruned_val = self._matcher.fit_transform(group_vals_)
            if group_pruned_val is None:
                # Nothing to compress, encode back the related groups
                for related_group in related_groups:
                    new_groups[related_group] = metric_data[
                        related_group
                    ].values.squeeze()
            else:
                new_groups[group_pruned_val] = metric_data[group_pruned_val].values
        # Replace the intermediates
        data.intermediate = pd.DataFrame(new_groups)
        unwanted = data.intermediate.columns[
            data.intermediate.columns.str.endswith("count")
        ]
        data.intermediate.drop(unwanted, axis=1, inplace=True)

        unwanted = data.intermediate.columns[
            data.intermediate.columns.str.startswith(
                "compaction.overall_compaction.interval_compaction"
            )
        ]
        data.intermediate.drop(unwanted, axis=1, inplace=True)

        return data
