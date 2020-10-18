import torch

import gpytorch
from collections import OrderedDict
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel


variational_distributions = {
    "delta": gpytorch.variational.DeltaVariationalDistribution,
    "mf": gpytorch.variational.MeanFieldVariationalDistribution,
    "mf_decoupled": gpytorch.variational.MeanFieldVariationalDistribution,
    "cholesky": gpytorch.variational.CholeskyVariationalDistribution,
    "natural_slow": gpytorch.variational.NaturalVariationalDistribution,
}

variational_strategies = {
    "standard": gpytorch.variational.VariationalStrategy,
    "ciq": gpytorch.variational.CIQVariationalStrategy,
    "eig": gpytorch.variational.EigVariationalStrategy,
    "eigqr": gpytorch.variational.EigQRVariationalStrategy,
}


class ApproximateSingleLayerGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, vs_class="standard", vd_class="delta", grid=None, orth_inducing_points=None):
        batch_shape = torch.Size([])
        if orth_inducing_points is not None:
            covar_variational_distribution = variational_distributions[vd_class](orth_inducing_points.size(-2))
            covar_variational_strategy = variational_strategies[vs_class](
                self, orth_inducing_points, covar_variational_distribution, learn_inducing_locations=(grid is None),
            )
            mean_variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
                covar_variational_strategy, inducing_points, mean_variational_distribution
            )

        else:
            if vd_class == "natural":
                variational_strategy = gpytorch.variational.NaturalVariationalStrategy(
                    self, inducing_points, mode=vs_class, learn_inducing_locations=(grid is None),
                )
            elif "decoupled" in vd_class:
                batch_shape = torch.Size([2])
                if vs_class == "standard":
                    variational_distribution = variational_distributions[vd_class](inducing_points.size(-2))
                    variational_strategy = gpytorch.variational.DecoupledVariationalStrategy(
                        self, inducing_points, variational_distribution, learn_inducing_locations=True,
                    )
                else:
                    raise RuntimeError
            else:
                variational_distribution = variational_distributions[vd_class](inducing_points.size(-2))
                variational_strategy = variational_strategies[vs_class](
                    self, inducing_points, variational_distribution, learn_inducing_locations=(grid is None),
                )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()

        # Create kernel
        input_dim = inducing_points.size(-1)
        base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=input_dim, batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
