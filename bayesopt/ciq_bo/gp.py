import torch
from botorch.fit import fit_gpytorch_model

import gpytorch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

# Hyper bounds settings
LENGTHSCALE_BOUNDS = Interval(0.01, 2.0)
SIGNAL_VAR_BOUNDS = Interval(0.02, 50.0)
NOISE_VAR_BOUNDS = Interval(1e-6, 1e-1)

# Default hyper settings
SIGNAL_VAR = "covar_module.outputscale"
LENGTHSCALE = "covar_module.base_kernel.lengthscale"
NOISE_VAR = "likelihood.noise"


# GP Model
class GP(ExactGP):
    """Helper class for a GP model in GPyTorch."""

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        ard_dims,
        use_keops=False,
        jitter=1e-6,
    ):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.jitter = jitter
        self.ard_dims = ard_dims
        kernel = KMaternKernel if use_keops else MaternKernel
        base_covar_module = kernel(
            lengthscale_constraint=LENGTHSCALE_BOUNDS,
            nu=2.5,
            ard_num_dims=ard_dims,
        )
        self.covar_module = ScaleKernel(base_covar_module, outputscale_constraint=SIGNAL_VAR_BOUNDS)

    def forward(self, x):
        """Compute kernel matrix at x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.add_jitter(self.jitter)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_keops):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2 and train_y.ndim == 1 and train_x.shape[0] == train_y.shape[0]

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=Interval(*NOISE_VAR_BOUNDS))
    likelihood = likelihood.to(dtype=train_x.dtype, device=train_x.device)

    ard_dims = train_x.shape[1]
    model = GP(train_x=train_x, train_y=train_y, likelihood=likelihood, ard_dims=ard_dims, use_keops=use_keops)
    model = model.to(dtype=train_x.dtype, device=train_x.device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Find a good starting point
    hypers = {SIGNAL_VAR: 1.0, LENGTHSCALE: 0.5, NOISE_VAR: 1e-3}
    model.initialize(**hypers)

    # Call fit_gpytorch_model
    fit_gpytorch_model(mll)

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model
