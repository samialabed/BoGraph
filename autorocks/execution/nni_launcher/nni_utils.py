from typing import Dict

from sysgym.params import ParamsSpace
from sysgym.params.boxes import BooleanBox, CategoricalBox, ContinuousBox, DiscreteBox


def conver_params_to_nni_search_space(
    parameters_space: ParamsSpace,
) -> Dict[str, Dict[str, any]]:
    search_space = {}
    for parameter_space in parameters_space.parameters():
        if isinstance(parameter_space.box, ContinuousBox):
            search_space[parameter_space.name] = {
                "_type": "uniform",
                "_value": [
                    parameter_space.box.lower_bound,
                    parameter_space.box.upper_bound,
                ],
            }
        elif isinstance(parameter_space.box, DiscreteBox):
            search_space[parameter_space.name] = {
                "_type": "quniform",
                "_value": [
                    parameter_space.box.lower_bound,
                    parameter_space.box.upper_bound,
                    1,
                ],
            }
        elif isinstance(parameter_space.box, BooleanBox):
            search_space[parameter_space.name] = {
                "_type": "choice",
                "_value": ["false", "true"],
            }
        elif isinstance(parameter_space.box, CategoricalBox):
            search_space[parameter_space.name] = {
                "_type": "choice",
                "_value": parameter_space.box.categories,
            }
        else:
            raise ValueError(f"Failed to parse the value {parameter_space}")
    return search_space
