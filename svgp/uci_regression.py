#######################################################################
###                                                                 ###
###       THIS CODE RUNS OFF THE ciq BRANCH                         ###
###                                                                 ###
###       make sure the UCITEST script runs error free before       ###
###       committing any changes to this experimental script        ###
###       or models.py                                              ###
###                                                                 ###
#######################################################################

import os
import tqdm
import argparse
import uuid
import pickle
import time
import numpy as np
import pandas as pd
import math
import torch
from scipy.cluster.vq import kmeans2
from collections import OrderedDict
import gpytorch

from logger import get_logger
from models import ApproximateSingleLayerGP
from crps import crps
from load_uci_data import load_uci_data, set_seed
from custom_loader import BatchDataloader
from util import AverageMeter


likelihoods = {
    "beta": gpytorch.likelihoods.BetaLikelihood,
    "bernoulli": gpytorch.likelihoods.BernoulliLikelihood,
    "gaussian": gpytorch.likelihoods.GaussianLikelihood,
    "laplace": gpytorch.likelihoods.LaplaceLikelihood,
    "studentt": gpytorch.likelihoods.StudentTLikelihood,
}


@torch.no_grad()
def do_eval(model, likelihood, loader, targets, args):
    model.eval()
    likelihood.eval()
    all_means, all_vars = [], []
    log_prob = 0.0

    with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.minres_tolerance(args["minres_tol"]):
        with gpytorch.settings.num_contour_quadrature(args["num_quad"]):
            with gpytorch.settings.max_preconditioner_size(0):
                with gpytorch.settings.eval_cg_tolerance(args["cg_tol"]), gpytorch.settings.cg_tolerance(args["cg_tol"]):
                    with gpytorch.settings.num_likelihood_samples(100):
                        for x_batch, y_batch in loader:
                            latent_dist = model(x_batch)
                            mean, var = latent_dist.mean.cpu(), latent_dist.variance.cpu()
                            all_means.append(mean), all_vars.append(var)
                            log_prob += likelihood.log_marginal(y_batch.type_as(x_batch), latent_dist).sum().item()

    # Aggregate results
    means = torch.cat(all_means)
    vars = torch.cat(all_vars)

    # Compute stats
    if args["likelihood"] == "bernoulli":
        rmse = (means.gt(0) != targets.cpu().bool()).float().mean().item()
    else:
        rmse = (means - targets.cpu()).pow(2.0).mean().clamp_min_(1e-5).sqrt().item()
    ll = log_prob / targets.size(0)
    if args["likelihood"] == "gaussian":
        var_ratio = (likelihood.noise.item() / (vars + likelihood.noise.cpu())).mean().item()
        mean_crps = crps(targets.cpu(), means, (vars + likelihood.noise.cpu()).clamp_min(1e-9).sqrt())
    else:
        var_ratio = 0
        mean_crps = 0

    # Reset for training
    model.train()
    likelihood.train()
    del means, vars

    return rmse, ll, var_ratio, mean_crps


def main(**args):
    if args["dataset"] == "houseelectric":
        args["milestones"] = [5, 50, 100]
        if args["num_epochs"] == 0:
            args["num_epochs"] = 150
    elif args["dataset"] == "robopush" or args["dataset"] == "airline" or args["dataset"] == "covtype":
        args["milestones"] = [1, 5, 10, 15]
        args["num_epochs"] = 20
    elif args["dataset"] == "3droad":
        args["milestones"] = [1, 5, 10, 15]
        args["num_epochs"] = 20
    elif args["dataset"] == "precip":
        args["milestones"] = [4, 20, 40, 60]
        args["num_epochs"] = 75
    else:
        args["milestones"] = [20, 150, 300]
        if args["num_epochs"] == 0:
            args["num_epochs"] = 400

    # NGD if we have a natural vd:
    log_file = (
        f"{args['dataset']}.{args['variational_strategy']}"
        f".lik_{args['likelihood']}"
        f".ni_{args['num_ind']}.bs_{args['batch_size']}.nmi_{args['num_minres_iter']}"
        f".cgtol_{args['cg_tol']}.nq_{args['num_quad']}"
        f".nepochs_{args['num_epochs']}.lr_{args['lr']}.vlr_{args['vlr']}.vg_{args['gamma']}.b1_{args['b1']}"
        f".seed_{args['seed']}.ns_{args['num_splits']}.nr_{args['num_restarts']}.{str(uuid.uuid4())[:4]}.log"
    )
    log = get_logger(args["log_dir"], log_file, use_local_logger=False)
    log(args)
    log("")

    state_dicts = []
    all_results = {"args": args}
    for split in range(args["num_splits"]):
        # Start logging
        all_results[split] = {}
        all_results[split]["seed"] = args["seed"] + split

        # Get datasets
        train_x, train_y, test_x, test_y, valid_x, valid_y, y_std = load_uci_data(
            args["data_dir"], args["dataset"], args["seed"] + split
        )
        N_train = train_x.size(0)
        all_results["N_train"] = N_train
        all_results["N_test"] = test_x.size(0)
        all_results["N_valid"] = valid_x.size(0)

        # Get dataloaders
        test_loader = BatchDataloader(test_x, test_y, args["batch_size"], shuffle=False)
        valid_loader = BatchDataloader(valid_x, valid_y, args["batch_size"], shuffle=False)
        train_loader = BatchDataloader(train_x, train_y, args["batch_size"], shuffle=True)
        train_loader_eval = BatchDataloader(train_x, train_y, args["batch_size"], shuffle=False)
        train_x_shape = train_x.shape
        del test_x, valid_x

        # Storage for restart information
        num_restarts = args["num_restarts"]
        all_results[split]["restart_lls"] = []
        all_results[split]["restart_seeds"] = []

        # Do multiple restarts
        for restart in range(num_restarts + 1):
            if restart < num_restarts:
                restart_seed = 11 * args["seed"] + 17 * split + restart * 277
                all_results[split]["restart_seeds"].append(restart_seed)
                set_seed(restart_seed)
            else:
                if num_restarts > 0:
                    best_seed = np.argmax(all_results[split]["restart_lls"])
                    set_seed(all_results[split]["restart_seeds"][best_seed])
                else:
                    set_seed(args["seed"])

            # Constuct inducing points from k_means
            inducing_points = train_x[torch.randperm(min(1000 * 100, N_train))[:args["num_ind"]], :]
            inducing_points = inducing_points.clone().data.cpu().numpy()
            inducing_points = torch.tensor(
                kmeans2(train_x.data.cpu().numpy(), inducing_points, minit="matrix")[0]
            ).cuda()
            inducing_points.add_(torch.rand_like(inducing_points).mul(0.05))

            # Construct likelihood and model
            likelihood = likelihoods[args["likelihood"]]().cuda()
            model = ApproximateSingleLayerGP(
                inducing_points,
                vs_class=args["variational_strategy"],
            ).cuda()
            log(model)

            # Adjust the initial lengthscale for low-D datasets
            if args["dataset"] in ["3droad", "precip"]:
                model.covar_module.base_kernel.initialize(lengthscale=0.01)
            elif args["dataset"] in ["covtype"]:
                model.covar_module.base_kernel.initialize(lengthscale=0.1)

            # Objective and optimizer
            objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0), combine_terms=False)
            variational_params = list(model.variational_strategy._variational_distribution.parameters())
            other_params = list(set(model.parameters()) - set(variational_params))
            for param in variational_params:
                print("vp", param.shape)
            for param in other_params:
                print("op", param.shape)

            optim = torch.optim.Adam(
                [{"params": other_params}, {"params": likelihood.parameters()}],
                lr=args["lr"], betas=(args["b1"], 0.999)
            )
            optim_var = gpytorch.optim.NGD(variational_params, lr=args["vlr"], num_data=train_y.size(0))
            sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args["milestones"], gamma=args["gamma"])

            # Prep for training
            ts = []
            results = []
            num_epochs = args["num_epochs"] if restart == num_restarts else args["num_epochs"] // 40
            epochs_iter = tqdm.tqdm(
                range(num_epochs),
                desc=f"{args['dataset']} Training (seed={args['seed']}, restart={restart + 1})"
            )
            model.train()
            likelihood.train()

            for i in epochs_iter:
                # For recording stats
                if args["likelihood"] == "bernoulli":
                    stats = ["enll", "kl", "err", "os"]
                else:
                    stats = ["enll", "kl", "rmse", "os", "noise", "nu"]
                meters = [AverageMeter() for _ in stats]
                tqdm_train_loader = tqdm.tqdm(train_loader, desc=f"Epoch {i + 1}", leave=False)
                start = time.time()

                # Training loop
                with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.minres_tolerance(args["minres_tol"]):
                    with gpytorch.settings.max_preconditioner_size(0):
                        with gpytorch.settings.num_contour_quadrature(args["num_quad"]):
                            with gpytorch.settings.cg_tolerance(args["cg_tol"]):
                                for x_batch, y_batch in tqdm_train_loader:
                                    optim.zero_grad()
                                    optim_var.zero_grad()
                                    output = model(x_batch)
                                    ell, kl, _ = objective(output, y_batch)
                                    loss = -ell + kl
                                    loss.backward()
                                    optim_var.step()

                                    optim.zero_grad()
                                    output = model(x_batch)
                                    ell, kl, _ = objective(output, y_batch)
                                    loss = -ell + kl
                                    loss.backward()
                                    optim.step()


                                    # Measure and record stats
                                    batch_size = x_batch.size(0)
                                    if args["likelihood"] == "bernoulli":
                                        # Compute RMSE
                                        with torch.no_grad():
                                            err = (output.mean.gt(0) != y_batch.bool()).float().mean()
                                        stat_vals = [
                                            -ell.item(),
                                            kl.item(),
                                            err.item(),
                                            model.covar_module.outputscale.mean().item(),
                                        ]
                                    else:
                                        # Compute RMSE
                                        with torch.no_grad():
                                            rmse = (y_batch - output.mean).pow(2).mean(0).sqrt()
                                        stat_vals = [
                                            -ell.item(),
                                            kl.item(),
                                            rmse.item(),
                                            model.covar_module.outputscale.mean().item(),
                                            likelihood.noise.item() if args["likelihood"] not in ["beta", "bernoulli"] else 0.,
                                            likelihood.deg_free.item() if args["likelihood"] == "studentt" else 0.,
                                        ]
                                    for stat_val, meter in zip(stat_vals, meters):
                                        meter.update(stat_val, batch_size)

                                    # Print stats
                                    res = dict((name, f"{meter.val:.3f}") for name, meter in zip(stats, meters))
                                    tqdm_train_loader.set_postfix(**res)

                # Wrap up epoch
                ts.append(time.time() - start)
                sched.step()
                res = dict((name, f"{meter.avg:.3f}") for name, meter in zip(stats, meters))
                epochs_iter.set_postfix(**res)

                # Record the training results into a CSV
                result = [("epoch", i + 1), ("time", sum(ts))]
                result += [(name, f"{meter.avg:.8f}") for name, meter in zip(stats, meters)]

                # Every 25 epochs - run data through test set
                if restart == num_restarts and not (i + 1) % (25):
                    train_rmse, train_ll, _, _ = do_eval(model, likelihood, train_loader_eval, train_y, args)
                    test_rmse, test_ll, var_ratio, _ = do_eval(model, likelihood, test_loader, test_y, args)
                    valid_rmse, valid_ll, _, _ = do_eval(model, likelihood, valid_loader, valid_y, args)
                    if args["likelihood"] == "bernoulli":
                        format_str = (
                            f"[epoch {i + 1:03d}]    "
                            f"train err: {train_rmse:.5f}  nll: {-train_ll:.5f}    "
                            f"test err: {test_rmse:.5f}  nll: {-test_ll:.5f}    "
                            f"valid err: {valid_rmse:.5f}  nll: {-valid_ll:.5f}    "
                        )
                        log(format_str)
                        result += [("train_err", train_rmse), ("train_nll", -train_ll)]
                        result += [("valid_err", valid_rmse), ("valid_nll", -valid_ll)]
                        result += [("test_err", test_rmse), ("test_nll", -test_ll)]
                    else:
                        format_str = (
                            f"[epoch {i + 1:03d}]    "
                            f"train rmse: {train_rmse:.5f}  nll: {-train_ll:.5f}    "
                            f"test rmse: {test_rmse:.5f}  nll: {-test_ll:.5f}    "
                            f"valid rmse: {valid_rmse:.5f}  nll: {-valid_ll:.5f}    "
                        )
                        log(format_str)
                        result += [("train_rmse", train_rmse), ("train_nll", -train_ll)]
                        result += [("valid_rmse", valid_rmse), ("valid_nll", -valid_ll)]
                        result += [("test_rmse", test_rmse), ("test_nll", -test_ll)]

                # Create a csv
                if restart == num_restarts:
                    results.append(OrderedDict(result))
                    pd.DataFrame(results).set_index("epoch").to_csv(
                        os.path.join(args["log_dir"], log_file.replace(".log", f".split{split + 1}.csv"))
                    )

            # Determine best seed for restart
            train_rmse, train_ll, _, _ = do_eval(model, likelihood, train_loader_eval, train_y, args)
            all_results[split]["restart_lls"].append(train_ll)
            all_results[split]["train_results"] = results
            if restart < num_restarts:
                continue

            # Get final test/valid stats
            test_rmse, test_ll, var_ratio, test_crps = do_eval(model, likelihood, test_loader, test_y, args)
            valid_rmse, valid_ll, _, _ = do_eval(model, likelihood, valid_loader, valid_y, args)

            if args["save_model"]:
                state_dicts.append({"model": model.state_dict(), "likelihood": likelihood.state_dict()})

            results = "[Split {}/{}] Test RMSE: {:.6f}  Test NLL: {:.6f}  Var Ratio: {:.6f}  CRPS: {:.4f}"
            results = results.format(split + 1, args["num_splits"], test_rmse, -test_ll, var_ratio, test_crps)
            log(results)
            results = "[Split {}/{}] Valid RMSE: {:.6f}  Valid NLL: {:.6f}"
            results = results.format(split + 1, args["num_splits"], valid_rmse, -valid_ll)
            log(results)
            results = "[Split {}/{}] Train RMSE: {:.6f}  Train NLL: {:.6f}"
            results = results.format(split + 1, args["num_splits"], train_rmse, -train_ll)
            log(results)
            results = "[Split {}/{}] Training Time: {:.6f}"
            results = results.format(split + 1, args["num_splits"], sum(ts))
            log(results)
            log("")

            all_results[split]["ts"] = sum(ts)
            all_results[split]["test_crps"] = test_crps
            if args["likelihood"] == "bernoulli":
                all_results[split]["test_err"] = test_rmse
                all_results[split]["train_err"] = train_rmse
                all_results[split]["valid_err"] = valid_rmse
            else:
                all_results[split]["test_rmse"] = test_rmse
                all_results[split]["train_rmse"] = train_rmse
                all_results[split]["valid_rmse"] = valid_rmse
            all_results[split]["test_ll"] = test_ll
            all_results[split]["train_ll"] = train_ll
            all_results[split]["valid_ll"] = valid_ll
            all_results[split]["var_ratio"] = var_ratio
            all_results[split]["outputscale"] = model.covar_module.outputscale.item()
            all_results[split]["likelihood.noise"] = likelihood.noise.item() if hasattr(likelihood, "noise") else 0.

    splits = range(args["num_splits"])
    if args["likelihood"] == "bernoulli":
        test_rmses = [all_results[s]["test_err"] for s in splits]
        valid_rmses = [all_results[s]["valid_err"] for s in splits]
        train_rmses = [all_results[s]["train_err"] for s in splits]
    else:
        test_rmses = [all_results[s]["test_rmse"] for s in splits]
        valid_rmses = [all_results[s]["valid_rmse"] for s in splits]
        train_rmses = [all_results[s]["train_rmse"] for s in splits]
    test_lls = [all_results[s]["test_ll"] for s in splits]
    valid_lls = [all_results[s]["valid_ll"] for s in splits]
    train_lls = [all_results[s]["train_ll"] for s in splits]
    var_ratios = [all_results[s]["var_ratio"] for s in splits]

    mean_test_rmse = np.mean(test_rmses)
    mean_test_ll = np.mean(test_lls)
    mean_valid_rmse = np.mean(valid_rmses)
    mean_valid_ll = np.mean(valid_lls)
    mean_train_rmse = np.mean(train_rmses)
    mean_train_ll = np.mean(train_lls)
    mean_var_ratio = np.mean(var_ratios)

    results = "[Summary] Test RMSE: {:.5f} NLL: {:.5f}  Valid RMSE: {:.5f} NLL: {:.5f}  Train RMSE: {:.5f} NLL: {:.5f}  "
    results += "Var Ratio: {:.5f}"
    results = results.format(
        mean_test_rmse, -mean_test_ll, mean_valid_rmse, -mean_valid_ll, mean_train_rmse, -mean_train_ll, mean_var_ratio
    )
    log(results)

    log(all_results)
    if args["save_model"]:
        torch.save(state_dicts, os.path.join(args["log_dir"], log_file.replace(".log", ".model")))

    with open(args["log_dir"] + "/" + log_file[:-4] + ".pkl", "wb") as f:
        pickle.dump(all_results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")

    # Directories for data/logs
    parser.add_argument("-ld", "--log-dir", type=str, default="./logs/")
    parser.add_argument("-dd", "--data-dir", type=str, default="./data/")
    parser.add_argument("-sm", "--save-model", type=bool, default=False)

    # Dataset and model type
    parser.add_argument("-d", "--dataset", type=str, default="3droad")
    parser.add_argument("-l", "--likelihood", type=str, default="gaussian", choices=["beta", "gaussian", "laplace", "studentt", "bernoulli"])
    parser.add_argument("-vs", "--variational_strategy", type=str, default="standard", choices=["standard", "ciq"])

    # Model args
    parser.add_argument("-ni", "--num-ind", type=int, default=1000)
    parser.add_argument("-bs", "--batch-size", type=int, default=256)
    parser.add_argument("-nmi", "--num-minres-iter", type=int, default=200)
    parser.add_argument("-mt", "--minres-tol", type=float, default=1e-4)
    parser.add_argument("-nq", "--num-quad", type=int, default=15)
    parser.add_argument("-cg", "--cg-tol", type=float, default=0.001)

    # Training args
    parser.add_argument("-n", "--num-epochs", type=int, default=0)
    parser.add_argument("-lr", "--lr", type=float, default=0.01)
    parser.add_argument("-b1", "--b1", type=float, default=0.90)
    parser.add_argument("-vlr", "--vlr", type=float, default=0.1)
    parser.add_argument("-gamma", "--gamma", type=float, default=0.1)

    # Seed/splits/restarts
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-ns", "--num-splits", type=int, default=1)
    parser.add_argument("-nr", "--num-restarts", type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))
