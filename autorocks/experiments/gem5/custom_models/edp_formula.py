from typing import Dict

from torch import Tensor

from autorocks.experiments.gem5.custom_models.constants import AVG_POWER, SIM_SEC
from autorocks.optimizer.bograph.dag_dao.model_nodes.determinstic_node import (
    DeterminsticModelNode,
)


class EDPFormula(DeterminsticModelNode):
    def forward(self, parents_vals: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        powr = parents_vals[AVG_POWER]
        latency = (1 / parents_vals[SIM_SEC]) ** 2

        return powr * latency
