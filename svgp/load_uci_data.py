import torch
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from math import floor
import numpy as np
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_airline_data(data_dir, seed):
    data = pd.read_pickle(data_dir + "airline.pickle")
    # Convert time of day from hhmm to minutes since midnight
    data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
    data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)
    data = data.loc[data["Month"] <= 4]

    y = data["ArrDelay"].values > 0  # Classification
    names = ["Month", "DayofMonth", "DayOfWeek", "plane_age", "AirTime", "Distance", "ArrTime", "DepTime"]
    X = data[names].values

    set_seed(0)
    N = 800 * 1000
    shuffled_indices = torch.randperm(X.shape[0])
    X = torch.tensor(X[shuffled_indices, :]).float()[0:N]
    y = torch.tensor(y[shuffled_indices]).float()[0:N]
    print("Loaded airline data with X/Y = ", X.shape, y.shape)

    set_seed(seed)
    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    # y -= y.mean()
    # y_std = y.std().item()
    # print("YSTD", y_std)
    # y /= y_std

    train_n = 700 * 1000
    valid_n = 50 * 1000

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None


def load_covtype_data(data_dir, seed):
    data = torch.tensor(pd.read_csv(data_dir + "covtype.csv").values).float()

    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = (data[:, -1].eq(2) | data[:, -1].eq(3)).long()
    print(X.min(), X.max(), y, y.float().mean())

    set_seed(seed)
    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None


def load_robopush_data(data_dir, seed):
    X_rand, X_turbo, fX_rand, fX_turbo = torch.load(data_dir + "robopush.pt")
    set_seed(seed)

    X = torch.cat([X_rand, X_turbo], dim=-2).float()
    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = torch.cat([fX_rand, fX_turbo]).float().clamp(0.001, 0.999)

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None


def load_precip_data(data_dir, seed):
    data = torch.tensor(pd.read_csv(data_dir + "precip.csv").values).float()

    set_seed(seed)

    X = data[:, :-1]
    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        print("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None




def load_uci_data(data_dir, dataset, seed):
    if dataset == "airline":
        return load_airline_data(data_dir, seed)
    elif dataset == "covtype":
        return load_covtype_data(data_dir, seed)
    elif dataset == "robopush":
        return load_robopush_data(data_dir, seed)
    elif dataset == "precip":
        return load_precip_data(data_dir, seed)

    set_seed(seed)

    data = torch.Tensor(loadmat(data_dir + dataset + '.mat')['data'])
    X = data[:, :-1]

    # Strip off first dimension of 3droad
    if dataset == "3droad":
        X = X[:, 1:]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        print("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = torch.Tensor(SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy()))

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()

    print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None
