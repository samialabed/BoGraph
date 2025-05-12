from botorch.models.model import Model
from gpytorch.kernels import AdditiveStructureKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from torch import Tensor

from autorocks.optimizer.botorch_opt.models.botorch_model_abc import BoTorchModel
from autorocks.optimizer.botorch_opt.models.custom_model import CustomSurrogateModel


class AdditiveModel(BoTorchModel):
    name: str = "AdditiveModel"

    def __repr__(self) -> str:
        return (
            f"AdditiveModel(mean_module:{str(self.mean_module)}, "
            f"base_kernel:{str(self.covar_module)})"
        )

    def __init__(
        self,
        mean_module: Mean = ConstantMean(),
        base_kernel: Kernel = ScaleKernel(MaternKernel(nu=2.5)),
    ):
        self.mean_module = mean_module
        self.covar_module = AdditiveStructureKernel(
            base_kernel,
            num_dims=10,
        )

    def model(self, train_x: Tensor, train_y: Tensor) -> Model:
        additive_model = CustomSurrogateModel(
            num_outputs=1,
            train_x=train_x,
            train_y=train_y,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
        )
        additive_model.to(train_x)
        additive_model.fit(train_x=train_x)
        return additive_model
