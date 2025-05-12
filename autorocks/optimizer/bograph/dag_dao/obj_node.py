from torch import Tensor

from autorocks.optimizer.bograph.dag_dao.node_abc import Node


class ObjectiveNode(Node):
    @property
    def observed_values(self) -> Tensor:
        return self._values
