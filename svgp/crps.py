import torch
import math


def phi(x, mu, sigma):
    return torch.distributions.Normal(0.0, 1.0).log_prob((x - mu) / sigma).exp()


def Phi(x, mu, sigma):
    return torch.distributions.Normal(0.0, 1.0).cdf((x - mu) / sigma)


# mean CRPS (continuous ranked probability score)
# lower is better.
# https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
def crps(x, mu, sigma):
    crps = -1.0 / math.sqrt(math.pi) + 2.0 * phi(x, mu, sigma)
    crps += ((x - mu) / sigma) * (2.0 * Phi(x, mu, sigma) - 1.0)
    crps = sigma * crps
    return crps.mean().item()


# mean CRPS (continuous ranked probability score) for mixtures of normals
# page 25: https://arxiv.org/pdf/1709.04743.pdf
def multi_crps(x, mu, sigma, weights=None):
    if weights is None:
        weights = torch.ones(mu.size(0)).float() / float(mu.size(0))
    weights = weights.unsqueeze(-1)
    delta = x - mu
    delta2 = mu - mu.unsqueeze(-2)
    sigma2 = (sigma.pow(2.0) + sigma.pow(2.0).unsqueeze(-2)).sqrt()
    A1 = delta * (2.0 * Phi(delta, 0.0, sigma) - 1.0) + 2.0 * sigma * phi(delta, 0.0, sigma)
    A2 = delta2 * (2.0 * Phi(delta2, 0.0, sigma2) - 1.0) + 2.0 * sigma2 * phi(delta2, 0.0, sigma2)
    crps = (weights * A1).sum(0) - 0.5 * (weights * weights.unsqueeze(-1) * A2).sum(0).sum(0)
    return crps.mean().item()
