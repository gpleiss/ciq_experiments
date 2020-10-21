import torch
from torch.quasirandom import SobolEngine

import gpytorch

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, standardize, to_unit_cube


class ThompsonSampling:
    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        jitter=1e-4,
        n_discrete_points=5000,
        max_preconditioner_size=50,
        num_ciq_samples=15,
        max_minres_iterations=200,
        minres_tolerance=1e-3,
        use_ciq=True,
        use_keops=False,
        dtype=None,
        device=None,
        verbose=False,
    ):
        """Thompson sampling algorithm.

        You want to use CUDA + torch.float32 for CIQ as it's much faster on the GPU. If you want to consider a large
        number of discrete points then you need to enable KeOps.

        Parameters
        ----------
        f : function handle that returns torch.tensor that we can call .item() on.
        lb : Lower variable bounds, torch.Tensor, shape (d,).
        ub : Upper variable bounds, torch.Tensor, shape (d,).
        n_init : Number of initial points, int.
        max_evals : Total evaluation budget, int.
        batch_size : Number of points in each batch, int.
        jitter : jitter to be added to the covariance matrix for numerical stability, float.
        n_discrete_points : Number of points to sample on, int.
        max_preconditioner_size :  Largest size of the pre-conditioner, int.
        num_ciq_samples : Number of CIQ samples (15 is usually enough), int.
        max_minres_iterations : Maximum number of MINRES iterations (increase if you see convergence errors), int.
        minres_tolerance : MINRES tolerance, float.
        use_ciq : True to use CIQ, otherwise Cholesky is used, bool.
        use_keops : True if KeOps is used, bool.
        dtype : torch.float32 or torch.float64.
        device : torch.device("cuda") or torch.device("cpu").
        verbose : If you want to print information about the optimization progress, bool.

        Example usage:
            ts = ThompsonSampling(
                f=f, lb=lb, ub=ub, n_init=10, max_evals=100, batch_size=5, n_discrete_points=5000, use_ciq=True
            )
            fx = ts.optimize()  # Run optimization and return tensor of values
            X = ts.X  # If you want to get the evaluated points
        """
        self.f = f
        self.lb = lb.to(dtype=dtype, device=device)
        self.ub = ub.to(dtype=dtype, device=device)
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.jitter = jitter
        self.n_discrete_points = n_discrete_points
        self.max_preconditioner_size = max_preconditioner_size
        self.num_ciq_samples = num_ciq_samples
        self.max_minres_iterations = max_minres_iterations
        self.minres_tolerance = minres_tolerance
        self.use_ciq = use_ciq
        self.use_keops = use_keops
        self.dtype = dtype
        self.device = device
        self.verbose = verbose

        self.n_evals = 0
        self.dim = len(lb)

    def make_batch(self):
        # Map inputs to [0, 1]^d and standardize outputs
        X = to_unit_cube(self.X.clone(), self.lb, self.ub)
        fX = standardize(self.fX.clone().squeeze(-1))

        # Always fit the GP with Cholesky + float64 on the cpu
        with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves=False):
            with gpytorch.settings.max_cholesky_size(1000000):
                gp = train_gp(
                    X.to(dtype=torch.float64, device=torch.device("cpu")),
                    fX.to(dtype=torch.float64, device=torch.device("cpu")),
                    use_keops=self.use_keops,
                )
                gp = gp.to(dtype=self.dtype, device=self.device)

        # Draw some samples
        Xcand = SobolEngine(self.dim, scramble=True).draw(self.n_discrete_points)
        Xcand = Xcand.to(dtype=self.dtype, device=self.device)

        with torch.no_grad():  # We don't need gradients for Thompson sampling
            if self.use_ciq:
                with gpytorch.settings.ciq_samples(True), gpytorch.settings.eval_cg_tolerance(self.minres_tolerance):
                    with gpytorch.settings.min_preconditioning_size(1):
                        with gpytorch.settings.max_preconditioner_size(self.max_preconditioner_size):
                            mean = gp(Xcand).mean
                            covar = gp(Xcand).lazy_covariance_matrix
                            dist = gpytorch.distributions.MultivariateNormal(mean, covar.add_jitter(self.jitter))
                            with gpytorch.settings.max_cholesky_size(0):
                                ycand = dist.rsample(torch.Size([self.batch_size])).t()
            else:
                with gpytorch.settings.max_cholesky_size(1000000):
                    mean = gp(Xcand).mean
                    covar = gp(Xcand).lazy_covariance_matrix
                    dist = gpytorch.distributions.MultivariateNormal(mean, covar.add_jitter(self.jitter))
                    ycand = dist.rsample(torch.Size([self.batch_size])).t()

        # Pick the new points
        assert ycand.shape == (self.n_discrete_points, self.batch_size)  # Make sure we got the shapes right
        X_next = torch.zeros(self.batch_size, self.dim, dtype=self.dtype, device=self.device)
        for i in range(self.batch_size):
            ind_min = ycand[:, i].argmin().item()
            X_next[i, :] = Xcand[ind_min, :]
            ycand[ind_min, :] = float("inf")

        return from_unit_cube(X_next, self.lb, self.ub)

    def optimize(self):
        dtype, device = self.dtype, self.device

        # Generate and evalute initial design points
        self.X = self.lb + (self.ub - self.lb) * latin_hypercube(self.n_init, self.dim, dtype=dtype, device=device)
        self.fX = torch.tensor([self.f(x).item() for x in self.X], dtype=dtype, device=device).unsqueeze(-1)
        self.n_evals += self.n_init

        # Adaptive phase
        while self.n_evals < self.max_evals:
            # Create and evaluate batch
            X_next = self.make_batch()
            fX_next = torch.tensor([self.f(x).item() for x in X_next], dtype=dtype, device=device).unsqueeze(-1)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = torch.cat((self.X, X_next), dim=0)
            self.fX = torch.cat((self.fX, fX_next), dim=0)

            if self.verbose:
                print(f"Iteration: {len(self.fX)}, Best value = {self.fX.min().item():.3e}", flush=True)
        return self.fX
