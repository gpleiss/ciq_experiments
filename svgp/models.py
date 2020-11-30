import torch

import gpytorch
from collections import OrderedDict
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel


vs_classes = {
    "ciq": gpytorch.variational.CiqVariationalStrategy,
    "standard": gpytorch.variational.VariationalStrategy,
}


class ApproximateSingleLayerGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, vs_class="standard"):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(-2))
        variational_strategy = vs_classes[vs_class](
            self, inducing_points, variational_distribution, learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()

        # Create kernel
        input_dim = inducing_points.size(-1)
        base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=input_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
