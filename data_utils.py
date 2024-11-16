import time
import random
import numpy as np
import pandas as pd

np.random.seed(42)  # ensure reproducibility, better to compare different models, plot learning rates
random.seed(42)

# read and split the financial data
split_date1 = "2021-01-01"  # user defined
split_date2 = "2022-01-01"
filename = "data/30features_csi500.parquet"  # csi500


def read_financial_data():
    """
    Read a financial data file (parquet), split into train and test according to split_date.
    The dataframe must contain the following columns: 
        "id" - representing stock id/name
        "date" - the date or datetime
        "target" - target returns
        and other features
    """
    print(f"Split financial data... {filename} at {split_date1}, {split_date2}")

    df = pd.read_parquet(filename)
    train_data = df[df["date"] < split_date1]
    valid_data = df[(df["date"] < split_date2) & (df["date"] >= split_date1)]
    test_data = df[df["date"] >= split_date2]
    return train_data, valid_data, test_data


train_set, valid_set, test_set = read_financial_data()  # global variables: loaded once


def get_dates_pair(dates, date_style="consecutive"):
    # pick two dates
    if date_style == "consecutive":
        dates_pair = list(zip(dates[:-1], dates[1:]))  # consecutive two dates

    elif date_style == "random":
        dates_pair = []
        remain_dates = dates.copy()  # only get random dates from remaining dates

        while len(remain_dates) > 1:
            pair = random.sample(remain_dates, 2)  # a list of random two dates, unsorted
            dates_pair.append(pair)
            remain_dates.remove(pair[1])  # only remove one date

    else:  # if config["date_style"]=="gap-2", then gap=2
        gap = int(date_style.split("-")[1])
        dates_pair = list(zip(dates[:-gap], dates[gap:]))  # two dates gap by n

    #     print(dates_pair)
    return dates_pair


def get_financial_data(train, **config):
    """
    Get all data samples of two dates from financial data
    :param train: either train or valid
                True - training set, by default during training
                False - validation set, if dataloader.validate(model)
    """
    start_time = time.time()  # record data processing time

    # pick train/valid set
    if train:
        data = train_set.copy()
    else:
        data = valid_set.copy()

    dates = data[
        "date"].drop_duplicates().sort_values().tolist()  # all dates available, sort_values before unique is slow
    dates_pair = get_dates_pair(dates, date_style=config["date_style"])

    """
    Process data: select features, process target
    """
    # fill a number of features with zeros if not enough features (zero padding)
    add_feats = config["num_features"] - (data.shape[1] - 3)
    for feat in range(add_feats):  # if add_feats < 0, do not add columns
        data["addfeat%s" % feat] = 0
    featnames = data.drop(["date", "id", "target"], axis=1).columns

    # select features
    if 'select_features' not in config:
        sample_feats = featnames[:config["num_features"]]  # first n features, fixed features
    else:
        sample_feats = config["select_features"]  # given features

    """
    Combine data of n dates
    """
    result_X, result_Y = [], []

    # for each dates data, L stocks per date
    for pair in dates_pair:
        data_date = data[data["date"].isin(pair)]

        # get L stocks to sample from
        stocks = data_date.groupby("id").apply(len)
        stocks = stocks[stocks > 1].index  # only select stock with two dates data available
        # slice stocks' data
        data_date = data_date.set_index("id").loc[stocks]

        # for entire train / validation
        # multiclass classfication: convert target (returns) into quantiles, rank before subsampling
        data_date["target"] = pd.qcut(x=data_date["target"], q=config["multiclass"], labels=range(config["multiclass"]))

        if train:  # training: random sampling
            count = int(
                np.ceil(len(stocks) / (config["seq_len"] // 2)) * 2)  # repeat whole samples twice (*2) = 500/50*2 = 20
            sample_stks = np.random.choice(stocks, size=(count, config["seq_len"] // 2),
                                           replace=True)  # with replacement, have repeats
        else:  # validation: fixed sample size seq_len//2
            count = len(stocks) // (config["seq_len"] // 2)  # drop last splitted stock group < seq_len//2
            sample_stks = np.split(stocks[: count * (config["seq_len"] // 2)], count)

        for stks in sample_stks:  # size = config["seq_len"]//2, iterate count times
            # sort so that the first half is train, the later half is test during bptt split
            # stk orders doesn't matter
            train_X_Y = data_date.loc[stks].reset_index().sort_values(["date", "id"])  # slice index

            # sample features, get target
            train_X = list(train_X_Y[sample_feats].values)  # slice columns
            train_Y = list(train_X_Y["target"].values)

            result_X.append(train_X)  # append data
            result_Y.append(train_Y)

    print("get financial data time...", time.time() - start_time, "s")

    # m stocks per port = seq_len//2
    # result_X.shape = (n dates*L stocks per date // m stocks, seq_len, features) or (n dates * count, seq_len, features), 
    # result_Y.shape = (n dates*L stocks per date // m stocks, seq_len, 1) or (n dates * count, seq_len, 1)
    return np.array(result_X), np.array(result_Y)  # return all datapoints
