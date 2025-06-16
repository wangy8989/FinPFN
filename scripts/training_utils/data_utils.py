from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 4213


def read_financial_data(filename, split_date1, split_date2, percentage=True):
    """
    Read a financial data file, split into train and test according to split_date.
    The dataframe must contain the following columns: 
        "date" - the date or datetime
        "id" - representing stock id/name
        "target" - target returns
        and other features
        
   split_date1, split_date2: split dataframe into train, validation and test(out-of-sample) data
   percentage: if True, make returns into percentage returns by multiplying 100, in order to match the bar distribution range
    """
    
    df = pd.read_parquet(filename)
    df["date"] = pd.to_datetime(df["date"])
    
    # make it percentage return
    if percentage:
        df["target"] = df["target"]*100
    
    train_data = df[df["date"] < split_date1]
    valid_data = df[(df["date"] < split_date2) & (df["date"] >= split_date1)]
    test_data = df[df["date"] >= split_date2]
    
    print(f"Split financial data... {filename} at {split_date1}, {split_date2}")
    
    return train_data, valid_data, test_data


def get_dates_pair(dates, date_style="consecutive"):
    """
    Get date pairs based on a list of dates.
    4 date pair styles:
        consecutive: [t-1, t]
        consecutive-n: [t-n, t-n+1, ..., t]
        random: [t1, t2] with t1<t2
        gap-n: [t-n, t] 
    """
    
    # pick two dates
    if date_style=="consecutive":
        dates_pair = list(zip(dates[:-1], dates[1:]))  # consecutive two dates
    
    elif "consecutive-" in date_style:
        n_past = int(date_style.split("-")[1])       # past n dates
        dates_pair = [dates[i - n_past: i + 1] for i in range(n_past, len(dates))]
        
    elif date_style=="random":
        dates_pair = []
        remain_dates = dates.copy()              # only get random dates from remaining dates
        
        while len(remain_dates) > 1:
            pair = random.sample(remain_dates, 2)    # a list of random two dates, unsorted
            dates_pair.append(pair)
            remain_dates.remove(pair[1])          # only remove one date
            
    elif "gap-" in date_style:
        gap = int(date_style.split("-")[1])         # if "gap-2", then gap=2
        dates_pair = list(zip(dates[:-gap], dates[gap:]))  # two dates gap by n
    
    else:
        raise ValueError("Incorrect datestyle.")
        
#     print(dates_pair)
    return dates_pair

    
def create_data(
    *,
    data_set: pd.DataFrame | None = None,
    seq_len: int = 1000,
    train: bool = False,
    date_style: str = "consecutive",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    
    """
    This function is used to create data batches from a dataframe.
    
    Arguments:
    ----------
    *
        all arguments after * must be passed as keyword arguments.
    data_set: pd.DataFrame
        Training, Validation or Test dataframe.
    seq_len: int
        Sequence length. The larger the slower (1000:50s, 200:6s, 1300:67s).
    train: bool
        Indicate whether the train set is passed it.

    Returns:
    --------
    Tuple of torch Tensors.

    """

    dates = data_set["date"].drop_duplicates().sort_values().tolist()  # all dates available, sort_values before unique is slow
    featnames = data_set.drop(["date", "id", "target"], axis=1).columns[:]
    
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    
#     if not train and "random" in date_style:
#         # if date style for train is random/gap dates, then for validation is still kept consecutive?
#         date_style = "consecutive"
    
    n_past = 1
    if "consecutive-" in date_style:
        n_past = int(date_style.split("-")[1])
    n_days = n_past + 1
    stocks_per_date = seq_len // n_days
    date_windows = get_dates_pair(dates, date_style=date_style)
    print(date_style, date_windows[0])

    for date_window in date_windows:  # list of [t-n_past, ..., t] date sets
        data_date = data_set[data_set["date"].isin(date_window)]

        # Keep stocks that appear in all dates in the window
        valid_stocks = data_date.groupby("id").count()["date"]
        stocks = valid_stocks[valid_stocks == n_days].index
        data_date = data_date.set_index("id").loc[stocks]

        if train:  # training: random sampling
            count = int(np.ceil(len(stocks) / stocks_per_date))  # repeat whole samples once
            sample_stks = np.random.choice(stocks, size=(count, stocks_per_date), replace=True)  # with replacement, have repeats
        else:    # validation: fixed sample size
            count = len(stocks) // stocks_per_date  # drop last splitted stock group < seq_len//2
            sample_stks = np.split(stocks[:count * stocks_per_date], count)

        for stks in sample_stks:  # each of size stocks_per_date
            # Sort by date then id to preserve order across dates
            group = data_date.loc[stks].reset_index().sort_values(["date", "id"])
            
            # Standardize target by date: zero mean and unit std per date
            group["target"] = group.groupby("date")["target"].transform(
                            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0
                        )

            # Convert to numpy
            X = group[featnames].to_numpy(dtype=np.float32)
            Y = group["target"].to_numpy(dtype=np.float32)

            # Each block of `stocks_per_date` corresponds to one date
            X_train_list.append(X[:n_past * stocks_per_date])
            y_train_list.append(Y[:n_past * stocks_per_date])

            X_val_list.append(X[n_past * stocks_per_date:])   # last block is test (today)
            y_val_list.append(Y[n_past * stocks_per_date:])

    # Convert lists to tensors: shape (batch_size, mid, ...)
    X_train = torch.tensor(np.stack(X_train_list), dtype=torch.float32)        # (batch_size, stks, n_features)
    y_train = torch.tensor(np.stack(y_train_list), dtype=torch.float32).unsqueeze(-1)  # (batch_size, stks, 1)
    X_val   = torch.tensor(np.stack(X_val_list), dtype=torch.float32)          # (batch_size, stks, n_features)
    y_val   = torch.tensor(np.stack(y_val_list), dtype=torch.float32).unsqueeze(-1)    # (batch_size, stks, 1)

    # Rearrange axes to match expected format: (n_samples, batch_size, ...)
    X_train = X_train.permute(1, 0, 2)  # (stks, batch_size, n_features)
    y_train = y_train.permute(1, 0, 2)  # (stks, batch_size, 1)
    X_val   = X_val.permute(1, 0, 2)    # (stks, batch_size, n_features)
    y_val   = y_val.permute(1, 0, 2)    # (stks, batch_size, 1)
    
    print(X_train.shape, X_val.shape)  # every epoch, validation

    return X_train, X_val, y_train, y_val


class DatePairDataset(Dataset):
    """
    PyTorch Dataset that generates batches from pairs of dates.
    Each item is a tuple: (X_train, y_train, X_test, y_test)
    """

    def __init__(self, 
                 dataset: tuple, 
                 max_steps: int,
                 batch_size: int,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        # every step has batch_size
#         self.num_steps = self.dataset[0].shape[1]
        self.num_steps = min(self.dataset[0].shape[1], max_steps*batch_size)  # limit the maximum step per epoch

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        X_train, X_val, y_train, y_val = self.dataset
        X_train, X_val, y_train, y_val = X_train[:, idx, :], X_val[:, idx, :], y_train[:, idx, :], y_val[:, idx, :]

        return {
            "X_train": X_train,  # (n_samples, n_features)
            "y_train": y_train,  # (n_samples, n_features)
            "X_test": X_val,    # (n_samples, 1)
            "y_test": y_val,    # (n_samples, 1)
        }


def get_data_loader(
    *,
    train_set: pd.DataFrame | None = None,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    num_workers: int,
    seq_len: int,
    date_style: str = "consecutive",
) -> DataLoader:
    """Get data loader.

    This function is used to get data loader.

    Arguments:
    ----------
    train_set: pd.DataFrame
        Input features and Target labels.
    max_steps: int
        Maximum number of steps per epoch (draws of the data).
    torch_rng: torch.Generator
        Torch random number generator for draws and similar.
    batch_size: int
        Batch size. How many draws to load at a time.
    num_workers: int
        Number of workers for data loader.
    seq_len: int
        Sequence length
    date_style: str
        Default: consecutive date pairs.

    Returns:
    --------
    DataLoader
        Data loader.
    """
    
    data_tuple = create_data(data_set=train_set, seq_len=seq_len, train=True, date_style=date_style)

    dataset = DatePairDataset(dataset=data_tuple, max_steps=max_steps, batch_size=batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,  # batch size of random indexes draw for every step, not repeated
        shuffle=True,  # True: Good for training to avoid learning patterns based on data(date) order.
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,  # True: to drop the last incomplete batch
        generator=torch_rng,
        persistent_workers=False,
    )
