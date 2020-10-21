import torch


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert torch.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert torch.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def standardize(y):
    """Standardize a 1-dimensional tensor."""
    assert y.ndim == 1
    return (y - y.mean()) / y.std()


def latin_hypercube(n_pts, dim, dtype=None, device=None):
    """Latin hypercube with center perturbation."""
    X = torch.zeros(n_pts, dim, dtype=dtype, device=device)
    centers = (1.0 + 2.0 * torch.arange(0.0, n_pts, dtype=dtype, device=device)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[torch.randperm(n_pts)]

    # Add some perturbations within each box
    pert = (-1.0 + 2.0 * torch.rand(n_pts, dim, dtype=dtype, device=device)) / float(2 * n_pts)
    X += pert
    return X
