import os
import sys
import math
import numpy as np
import time
import torch
from torch.distributions import Gamma
import cv2
import kornia
import gpytorch
import torchvision
from gpytorch.lazy import NonLazyTensor
from gpytorch.utils import linear_cg


# mvn sampling helper
@torch.no_grad()
def mvn_sample(prec, minres_tol=1.0e-5, num_quad=15, num_cg_iter=1000):
    epsilon = torch.randn(prec.size(0), dtype=prec.dtype, device=prec.device)
    with gpytorch.settings.minres_tolerance(minres_tol), gpytorch.settings.record_ciq_stats(), \
         gpytorch.settings.num_contour_quadrature(num_quad), gpytorch.settings.max_cg_iterations(num_cg_iter):
            return prec.sqrt_inv_matmul(epsilon)

# inverse matmul helper
@torch.no_grad()
def inv_matmul(rhs, prec, num_cg_iter=1000, cg_tol=1.0e-5):
    with gpytorch.settings.max_cg_iterations(num_cg_iter), gpytorch.settings.cg_tolerance(cg_tol):
        squeeze = False
        if rhs.dim() == 1:
            rhs = rhs.unsqueeze(-1)
            squeeze = True

        diag = prec.diag().unsqueeze(-1)
        solves = linear_cg(
            prec.matmul,
            rhs,
            n_tridiag=0,
            max_iter=num_cg_iter,
            tolerance=cg_tol,
            preconditioner=lambda x: x / diag,
        )

        if squeeze:
            solves = solves.squeeze(-1)
        return solves

# downsample image using one of two strategies
def downsample_img(img, binary=True, factor=2):
    if binary:
        return img[::factor, ::factor]
    else:
        result = 0.0
        Z = factor ** 2
        for i in range(factor):
            for j in range(factor):
                result += img[i::factor, j::factor] / Z
        return result


# get low res images from high res image
def get_low_res_data(image, K=5, blur_radius=1.0, obs_sigma=2.0, num_images=5, binary_ds=True, ds_factor=2):
    N = image.shape[0]

    # create blur filter
    blur = kornia.filters.GaussianBlur2d((K, K), (blur_radius, blur_radius))
    # blur input image
    blurred = blur(torch.Tensor(image).float().unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    # construct explicit representation of blur matrix
    fake = torch.zeros(N, N, N, N)
    for i in range(N):
        for j in range(N):
            fake[i, j, i, j] = 1.0
    blur_mat = torch.empty_like(fake)
    for k in range(16):
        blur_mat[k * N // 16:(k+1) * N // 16] = blur(fake[k * N // 16:(k+1) * N // 16].cuda()).cpu()
    blur_mat = blur_mat.reshape(N * N, N * N).t()

    # let's ensure blur_mat is correct
    blurred_explicit = torch.matmul(blur_mat, torch.Tensor(image).float().reshape(N * N))
    blurred_kornia = blurred.reshape(N * N)
    mean_delta = (blurred_explicit - blurred_kornia).abs().mean().item()
    assert mean_delta < 1.0e-4

    # downsample blurred image
    downsampled = downsample_img(blurred, binary=binary_ds, factor=ds_factor)
    M = downsampled.size(0)
    assert M == N // ds_factor
    downsampled = downsampled.reshape(M * M)
    downsampled = downsampled.repeat(num_images)
    # create num_images many downsampled images, each with iid observation noise
    downsampled += obs_sigma * torch.randn(downsampled.shape)

    return (blurred,      # N x N tensor
            downsampled,  # vector of size num_images x M x M
            blur_mat)     # matrix of shape N^2 x N^2


# construct an explicit downsampling matrix
def get_downsample_matrix(N, M, num_images=7, binary=True, factor=2):
    assert M == N // ds_factor
    mat = torch.zeros(M, M, N, N)
    if binary:
        for i in range(M):
            for j in range(M):
                mat[i, j, factor * i, factor * j] = 1.0
    else:
        for i in range(M):
            for j in range(M):
                for k in range(factor):
                    for l in range(factor):
                        mat[i, j, factor * i + k, factor * j + l] = (1.0 / factor) ** 2

    mat = mat.reshape(M * M, N * N)

    # let's ensure mat is correct
    image = torch.randn(N, N)
    downsampled = downsample_img(image, binary=binary, factor=factor).reshape(M * M)
    downsampled_explicit = torch.mv(mat, image.reshape(N * N))
    mean_delta = (downsampled - downsampled_explicit).abs().mean().item()
    assert mean_delta < 1.0e-6

    mat = mat.repeat(num_images, 1, 1)

    return mat.reshape(num_images * M * M, N * N)  # (num_images x M^2) x N^2 matrix


# construct explicit representation of laplace matrix (this is for the prior)
def get_laplace_mat(N, K=3):
    # create laplace filter
    assert K == 3
    laplace = kornia.filters.Laplacian(K)
    laplace.kernel[0, 0, 0] = 1.0 / 12.0
    laplace.kernel[0, 0, 2] = 1.0 / 12.0
    laplace.kernel[0, 2, 0] = 1.0 / 12.0
    laplace.kernel[0, 2, 2] = 1.0 / 12.0
    laplace.kernel[0, 0, 1] = 1.0 / 6.0
    laplace.kernel[0, 2, 1] = 1.0 / 6.0
    laplace.kernel[0, 1, 0] = 1.0 / 6.0
    laplace.kernel[0, 1, 2] = 1.0 / 6.0
    laplace.kernel[0, 1, 1] = -1.0

    # construct explicit representation of laplace matrix
    fake = torch.zeros(N, N, N, N)
    for i in range(N):
        for j in range(N):
            fake[i, j, i, j] = 1.0
    fake = fake.reshape(N * N, 1, N, N)
    laplace_mat = torch.empty_like(fake)
    for k in range(16):
        laplace_mat[k * (N * N) // 16:(k+1) * (N * N) // 16] = laplace(fake[k * (N * N) // 16:(k+1) * (N * N) // 16].cuda()).cpu()
    laplace_mat = laplace_mat.reshape(N * N, N * N).t()

    # let's ensure laplace_mat is correct
    image = torch.randn(N, N)
    laplace_explicit = torch.matmul(laplace_mat, image.reshape(N * N))
    laplace_kornia = laplace(image.unsqueeze(0).unsqueeze(0))[0, 0, :, :].reshape(N * N)
    mean_delta = (laplace_explicit - laplace_kornia).abs().mean().item()
    assert mean_delta < 1.0e-7

    return 0.1 * laplace_mat


# [0, 255] -> [-0.5, 0.5]
def normalize_img(x):
    return (x - 127.5) / 255.0

# [-0.5, 0.5] -> [0, 255]
def unnormalize_img(x):
    return 255.0 * x + 127.5

# compute quantiles etc
def quantile_stats(x):
    x_mean, x_min, x_max = np.mean(x), np.min(x), np.max(x)
    x_05, x_95 = np.percentile(x, [5.0, 95.0]).tolist()
    return "mean: {:.3g}  min: {:.3g}  max: {:.3g}  p05: {:.3g}  p95: {:.3g}".format(x_mean, x_min, x_max, x_05, x_95)


# run the gibbs sampler for N_steps
@torch.no_grad()
def do_gibbs(lr_imgs=None, ds_blur_mat=None, N=None, M=None,
             x_init=None, ds_blur_prec=None, laplace_mat=None, laplace_prec=None,
             gamma_pr_init=1.0, gamma_obs_init=1.0, N_steps=800, burnin=200):

    assert N_steps > burnin

    # move everything to gpu; normalizer images
    ds_blur_mat = ds_blur_mat.cuda()
    lr_imgs = normalize_img(lr_imgs.cuda())
    laplace_mat = laplace_mat.cuda()
    upsampled_image = torch.mv(ds_blur_mat.t(), lr_imgs).cuda()
    laplace_prec = laplace_prec.cuda()
    ds_blur_prec = ds_blur_prec.cuda()

    x_history = []
    gamma_obs_history = []
    gamma_pr_history = []

    # initialize gibbs sampler
    x_prev = normalize_img(x_init.cuda())

    ts = torch.zeros(N_steps)
    for step in range(N_steps):
        t0 = time.time()

        # sample gamma_obs from its posterior conditional
        downsampled_x = torch.mv(ds_blur_mat, x_prev)
        alpha_obs = 1.0 + 0.5 * lr_imgs.size(0)
        beta_obs = 2.0 / (lr_imgs - downsampled_x).pow(2.0).sum().cpu().double()
        gamma_obs = Gamma(alpha_obs, beta_obs).sample().item()
        gamma_obs_history.append(gamma_obs)

        # sample gamma_pr from its posterior conditional
        laplace_x = torch.mv(laplace_mat, x_prev)
        alpha_pr = 1.0 + 0.5 * (x_prev.size(0) - 1.0)
        beta_pr = 2.0 / laplace_x.pow(2.0).sum().cpu().double()
        gamma_pr = Gamma(alpha_pr, beta_pr).sample().item()
        gamma_pr_history.append(gamma_pr)

        # sample latent high res image from its posterior conditional
        prec = NonLazyTensor(ds_blur_prec + (gamma_pr / gamma_obs) * laplace_prec)
        mean = inv_matmul(upsampled_image, prec)
        x_new = mvn_sample(prec) / math.sqrt(gamma_obs) + mean
        minres_residual = gpytorch.settings.record_ciq_stats.minres_residual
        if minres_residual > 1.0e-3:
            print("minres_residual: {:.4f}".format(minres_residual))

        # keep x sample for next iteration
        x_history.append(x_new.cpu())
        x_prev = x_new

        t1 = time.time()
        ts[step] = t1 - t0
        if step % 10 == 0 or step == N_steps - 1:
            print("Doing Gibbs step {} . . .    [dt: {:.3f}]".format(step + 1, t1 - t0))

    posterior_mean = unnormalize_img(torch.stack(x_history)[burnin:].mean(0))
    posterior_stdv = unnormalize_img(torch.stack(x_history)[burnin:].std(0))
    print("[posterior mean image stats] mean, min, max:", posterior_mean.mean().item(),
          posterior_mean.min().item(), posterior_mean.max().item())

    gamma_obs_history = gamma_obs_history[burnin:]
    gamma_pr_history = gamma_pr_history[burnin:]
    print("[gamma_obs] ", quantile_stats(gamma_obs_history))
    print("[gamma_pr]  ", quantile_stats(gamma_pr_history))

    #posterior_mean2 = inv_matmul(upsampled_image, prec).cpu()
    #posterior_mean2 = unnormalize_img(posterior_mean2)

    return posterior_mean, posterior_stdv, ts.mean().item()



if __name__ == "__main__":
    with torch.no_grad():
        _, filename, K, blur_radius = sys.argv
        basename = os.path.splitext(os.path.basename(filename))[0]
        raw_img = cv2.imread(filename, 0)

        N = raw_img.shape[0]  # high resolution image size
        ds_factor = 2         # factor by which to downsample
        M = N // ds_factor    # low resolution image size
        num_images = 4        # number of low res images
        K = int(K)            # gaussian blur filter size
        binary_ds = True     # downsampling strategy
        blur_radius = float(blur_radius)
        print(blur_radius)
        obs_sigma = 1.0
        print("M, N = ", M, N)


        blurred, downsampled, blur_mat = get_low_res_data(raw_img, K=K, num_images=num_images,
                                                          binary_ds=binary_ds, blur_radius=blur_radius, obs_sigma=obs_sigma,
                                                          ds_factor=ds_factor)
        if not os.path.isdir("results"):
            os.makedirs("results")
        cv2.imwrite(f'results/blurred_{basename}.png', blurred.data.numpy())
        print("[blurred image stats] mean, min, max:", blurred.mean().item(), blurred.min().item(), blurred.max().item())

        laplace_mat = get_laplace_mat(N).cuda()
        laplace_prec = torch.mm(laplace_mat.t(), laplace_mat).cpu()

        ds_mat = get_downsample_matrix(N, M, num_images=num_images, binary=binary_ds, factor=ds_factor)
        ds_mat, blur_mat = ds_mat.cuda(), blur_mat.cuda()
        ds_blur_mat = torch.mm(ds_mat, blur_mat)
        ds_blur_prec = torch.mm(ds_blur_mat.t(), ds_blur_mat).cpu()
        del ds_mat
        del blur_mat


        # initialize with one of the downsampled images blown up by a factor of ds_factor
        x_init = downsampled.reshape(num_images, M, M)[0].repeat_interleave(ds_factor, 0).repeat_interleave(ds_factor, 1).reshape(N * N)
        to_save = torchvision.utils.make_grid(
            downsampled.reshape(num_images, M, M).unsqueeze(-3),
            nrow=int(math.sqrt(num_images)),
            padding=1,
        ).squeeze(-3)[0]
        cv2.imwrite(f'results/init_{basename}.png', to_save.data.numpy())

        post_mean, post_stdv, t_mean = do_gibbs(lr_imgs=downsampled, ds_blur_mat=ds_blur_mat, N=N, M=M,
                                                x_init=x_init, ds_blur_prec=ds_blur_prec,
                                                laplace_prec=laplace_prec, laplace_mat=laplace_mat,
                                                N_steps=1000, burnin=200)
        delta = post_mean - torch.tensor(raw_img).view(post_mean.shape)

        cv2.imwrite(f'results/posterior_{basename}.png', post_mean.reshape(N, N).data.numpy())
        cv2.imwrite(f'results/stdv_{basename}.png', post_stdv.reshape(N, N).data.numpy())
        cv2.imwrite(f'results/delta_{basename}.png', delta.reshape(N, N).data.numpy())
        print(f"Avg time per iter: {t_mean}")
        print(f"Delta mean: {delta.pow(2).mean().sqrt()}")
